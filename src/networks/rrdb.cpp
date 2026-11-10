#include "network.h"
#include <algorithm>

namespace upscaler {
class RRDBNet final : public Network
{
public:
    BufferPtr shallow, saved, dense, x, hi0, hi1, rgb, winIn, winOut;
    explicit RRDBNet(Gpu& gpu) : Network(gpu, Model::Live, "RRDBV3D!")
    {
    }
    void allocate(int h, int w, int scale) override
    {
        if (estimatedMemory(Model::Live, h, w, scale) > uint64_t(3) * 1024 * 1024 * 1024) throw std::runtime_error("Working set exceeds 3 GiB; use --tile 256");
        H = h;
        W = w;
        N = h * w;
        const int strip = liveStripRows(H, W);
        winIn.reset();
        winOut.reset();
        rgb.reset();
        reserve(input, size_t(N) * 3);
        reserve(shallow, size_t(N) * 64);
        reserve(saved, size_t(N) * 64);
        reserve(dense, std::max(size_t(N) * 192, size_t(std::min(H * 4, strip + 2)) * W * 4 * 64));
        reserve(x, size_t(N) * 64);
        rgb = x; // The 64-channel trunk result dies after the first upsampling.
        reserve(output, size_t(N) * scale * scale * 3);
        size_t tiles = (((H + 3) / 4) * ((W + 3) / 4) + 3) / 4 * 4;
        {
            // Transforms die before the upsampling tail reuses these buffers.
            reserve(hi0, std::max(size_t(N) * 4 * 64, tiles * 36 * 192));
            reserve(hi1, std::max(size_t(std::min(H * 4, strip + 4)) * W * 4 * 64, tiles * 36 * 64));
            winIn = hi0;
            winOut = hi1;
        }
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
            p[5] = tensors.at(name + ".winograd").first;
            // Linear input transforms for earlier dense features remain valid.
            // Transform only the newly appended 32 channels at each growth step.
            p[14] = ic == 64 ? 0 : ic - 32;
            p[15] = ic == 64 ? 64 : 32;
            int groups = (p[12] * p[15] + 191) / 192, gx = std::min(groups, 65535);
            gpu.dispatch("conv_win_in", {in, {}, winIn, weights, {}, {}}, p, gx, (groups + gx - 1) / gx, 1);
            gpu.dispatch("conv_win_mul", {winIn, {}, winOut, weights, {}, {}}, p, (oc + 15) / 16, 3 * ((p[12] + 15) / 16), 1);
            groups = (tiles * oc + 191) / 192;
            gx = std::min(groups, 65535);
            gpu.dispatch("conv_win_out", {winOut, {}, out, weights, add, mode == 3 ? saved : BufferPtr{}}, p, gx, (groups + gx - 1) / gx, 1);
            return;
        }
        p[5] = tensors.at(name + ".weight").first;
        gpu.dispatch("conv_fast", {in, {}, out, weights, add, {}}, p, (oc + 15) / 16, (p[17] * w + 191) / 192, 1);
    }
    void pack(BufferPtr in, BufferPtr out, int mode)
    {
        Params p{};
        p[0] = N;
        p[1] = mode;
        int groups = (N * 64 + 191) / 192, gx = std::min(groups, 65535);
        gpu.dispatch("pack", {in, {}, out, {}, saved, {}}, p, gx, (groups + gx - 1) / gx, 1);
    }
    void execute(int scale) override
    {
        gpu.begin();
        conv(input, shallow, "conv_first", H, W, 3, 64, 3, 64, 0);
        pack(shallow, dense, 0);
        dump(shallow, "shallow", N * 64);
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
            if (!trace.empty())
            {
                pack(dense, x, 1);
                dump(x, "block" + std::to_string(block), N * 64);
            }
        }
        conv(dense, x, "conv_body", H, W, 64, 64, 192, 64, 0, 0, 1, shallow);
        conv(x, hi0, "conv_up1", H * 2, W * 2, 64, 64, 64, 64, 0, 1, 2);
        // Stream the 4x tail in strips, retaining its complete two-row context.
        // Coordinates remain global, so only true image edges receive padding.
        const int strip = liveStripRows(H, W);
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
    void rescale(int scale) override
    {
        gpu.begin();
        recordOutput(scale);
        gpu.finish();
    }
    void recordOutput(int scale)
    {
        downscale.record(rgb, output, H, W, scale, Model::Live);
    }
};

std::unique_ptr<Network> makeLiveNetwork(Gpu& gpu)
{ return std::make_unique<RRDBNet>(gpu); }
} // namespace upscaler
