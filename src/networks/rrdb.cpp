#include "network.h"
#include <algorithm>

namespace upscaler {
namespace {
int liveStripRows(int h, int w, const Config& config)
{
    // Fit the tail and its halos in the trunk's dense and transform buffers.
    const auto tiles = winogradTileCount(h, w);
    const int rows = std::min(h * 3 / 4 - 2, int(tiles * 9 / w) - 4);
    return std::clamp(rows / 16 * 16, 96, config.liveMaxStripRows);
}
}

class RRDBNet final : public Network
{
public:
    BufferPtr shallow, saved, dense, x, hi0, hi1, rgb, winIn, winOut;
    explicit RRDBNet(Gpu& gpu) : Network(gpu, Model::Live, "RRDBV3D!")
    {
    }
    static auto bufferSizes(int h, int w, int scale, int strip)
    {
        const size_t pixels = size_t(h) * w;
        const auto tiles = winogradTileCount(h, w);
        using Size = std::pair<BufferPtr RRDBNet::*, size_t>;
        return std::to_array<Size>({
            {&RRDBNet::input, pixels * 3},
            {&RRDBNet::shallow, pixels * 64},
            {&RRDBNet::saved, pixels * 64},
            {&RRDBNet::dense, std::max(pixels * 192, size_t(std::min(h * 4, strip + 2)) * w * 256)},
            {&RRDBNet::x, pixels * 64},
            {&RRDBNet::output, pixels * scale * scale * 3},
            // The forward Winograd transform stores 192 FP16 channels, or 96 32-bit words.
            {&RRDBNet::hi0, std::max(pixels * 256, tiles * 36 * 96)},
            {&RRDBNet::hi1, std::max(size_t(std::min(h * 4, strip + 4)) * w * 256, tiles * 36 * 64)},
        });
    }
    void allocate(int h, int w, int scale) override
    {
        H = h;
        W = w;
        N = h * w;
        const int strip = liveStripRows(H, W, gpu.config);
        // Bound descriptor usage by the trunk plus the six dispatches in each tail strip.
        gpu.reserveDispatches(1100 + 6 * ((H * 4 + strip - 1) / strip));
        winIn.reset();
        winOut.reset();
        rgb.reset();
        for (const auto& [buffer, elements] : bufferSizes(h, w, scale, strip))
            reserve(this->*buffer, elements);
        rgb = x; // The 64-channel trunk result dies after the first upsampling.
        // Transforms die before the upsampling tail reuses these buffers.
        winIn = hi0;
        winOut = hi1;
    }
    void conv(BufferPtr in, BufferPtr out, const std::string& name, int h, int w, int ic, int oc, int stride, int outstride, int offset, int mode = 0, int up = 1, BufferPtr add = {}, int top = 0, int rows = 0, int inputTop = 0, int outputTop = 0)
    {
        Params p{};
        p[0] = h;
        p[1] = w;
        p[2] = ic;
        p[3] = oc;
        p[4] = up;
        p[6] = tensors.at(name + ".bias").first;
        p[7] = mode;
        p[8] = stride;
        p[9] = outstride;
        p[10] = offset;
        p[11] = bool(add);
        p[16] = top;
        p[17] = rows ? rows : h;
        p[18] = inputTop;
        p[19] = outputTop;
        if (name.rfind("body.", 0) == 0 || name == "conv_body")
        {
            int tiles = ((h + 3) / 4) * ((w + 3) / 4);
            p[12] = (tiles + 3) / 4 * 4;
            p[13] = (w + 3) / 4;
            p[22] = packedTensors.at(name + ".winograd").first;
            // Linear input transforms for earlier dense features remain valid.
            // Transform only the newly appended 32 channels at each growth step.
            p[14] = ic == 64 ? 0 : ic - 32;
            p[15] = ic == 64 ? 64 : 32;
            p[21] = ic == 192 && oc == 64;
            const bool paired = h > 4;
            const int inputTiles = paired ? ((h + 7) / 8) * p[13] : p[12];
            const char* forward = paired ? "conv_win_in_pair" : "conv_win_in_single";
            gpu.dispatchLinear(forward, {in, {}, winIn, weights, {}, {}}, p, inputTiles * p[15]);
            gpu.dispatch("conv_win_mul_half",
                         {winIn, {}, winOut, weights, {}, halfWeights}, p, (oc + 15) / 16, 3 * ((p[12] + 15) / 16), 1);
            gpu.dispatchLinear("conv_win_out",
                               {winOut, {}, out, weights, add, mode == 3 ? saved : BufferPtr{}}, p, tiles * oc);
            return;
        }
        if (up == 2 && tensors.contains(name + ".polyphase"))
        {
            // Four parity-specific 2x2 convolutions replace nearest-neighbour
            // expansion followed by a 3x3 convolution. All weights stay FP32.
            for (int phase = 0; phase < 4; ++phase)
            {
                p[5] = at(name + ".polyphase") + phase * 4 * ic * oc;
                p[20] = phase;
                gpu.dispatch("conv_poly", {in, {}, out, weights, add, {}}, p,
                             (oc + 15) / 16, (p[17] * w / 4 + 191) / 192, 1);
            }
            return;
        }
        p[5] = tensors.at(name + ".weight").first;
        gpu.dispatch("conv_fast", {in, {}, out, weights, add, {}}, p,
                     (oc + 15) / 16, (p[17] * w + 191) / 192, 1);
    }
    void pack(BufferPtr in, BufferPtr out, int mode)
    {
        Params p{};
        p[0] = N;
        p[1] = mode;
        gpu.dispatchLinear("pack", {in, {}, out, {}, saved, {}}, p, N * 64);
    }
    void execute(int scale) override
    {
        gpu.begin();
        conv(input, shallow, "conv_first", H, W, 3, 64, 3, 64, 0);
        pack(shallow, dense, 0);
        for (int block = 0; block < 23; block++)
        {
            // Save the outer residual; growth features occupy disjoint channel slices.
            pack(dense, saved, 1);
            for (int rdb = 1; rdb <= 3; rdb++)
            {
                std::string base = "body." + std::to_string(block) + ".rdb" + std::to_string(rdb);
                for (int c = 1; c <= 4; c++) conv(dense, dense, base + ".conv" + std::to_string(c), H, W, 64 + (c - 1) * 32, 32, 192, 192, 64 + (c - 1) * 32, 1);
                // Fold both residual updates into the inverse Winograd transform.
                conv(dense, dense, base + ".conv5", H, W, 192, 64, 192, 192, 0, rdb == 3 ? 3 : 2, 1, dense);
            }
        }
        conv(dense, x, "conv_body", H, W, 64, 64, 192, 64, 0, 0, 1, shallow);
        conv(x, hi0, "conv_up1", H * 2, W * 2, 64, 64, 64, 64, 0, 1, 2);
        // Stream the 4x tail in strips, retaining its complete two-row context.
        // Coordinates remain global, so only true image edges receive padding.
        const int strip = liveStripRows(H, W, gpu.config);
        for (int top = 0; top < H * 4; top += strip)
        {
            const int end = std::min(top + strip, H * 4);
            const int upTop = std::max(top - 2, 0), upEnd = std::min(end + 2, H * 4);
            const int hrTop = std::max(top - 1, 0), hrEnd = std::min(end + 1, H * 4);
            conv(hi0, hi1, "conv_up2", H * 4, W * 4, 64, 64, 64, 64, 0, 1, 2, {}, upTop, upEnd - upTop);
            conv(hi1, dense, "conv_hr", H * 4, W * 4, 64, 64, 64, 64, 0, 1, 1, {}, hrTop, hrEnd - hrTop, upTop);
            conv(dense, rgb, "conv_last", H * 4, W * 4, 64, 3, 64, 3, 0, 0, 1, {}, top, end - top, hrTop, top);
        }
        recordOutput(scale);
        gpu.finish();
    }
    void recordOutput(int scale)
    {
        downscale.record(rgb, output, H, W, scale, Model::Live);
    }
};

MemoryEstimate estimateLiveBuffers(int h, int w, int scale, const Config& config)
{
    return estimateBuffers(RRDBNet::bufferSizes(h, w, scale, liveStripRows(h, w, config)));
}

std::unique_ptr<Network> makeLiveNetwork(Gpu& gpu)
{ return std::make_unique<RRDBNet>(gpu); }
} // namespace upscaler
