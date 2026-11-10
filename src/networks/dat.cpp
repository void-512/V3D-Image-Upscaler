#include "network.h"
#include <algorithm>

namespace upscaler {
class DatNetwork final : public Network
{
public:
    int chunks = 0;
    BufferPtr shallow, residual, x, norm, qkv, conv, wide, gate, mixed, partial, small0, small1, channelMap, spatialMap, channelScores, norms;
    explicit DatNetwork(Gpu& gpu) : Network(gpu, Model::Anime, "APISRDAT")
    {
    }
    void allocate(int h, int w, int scale) override
    {
        H = h;
        W = w;
        N = h * w;
        chunks = (N + 255) / 256;
        if (estimatedMemory(Model::Anime, h, w, scale) > uint64_t(3) * 1024 * 1024 * 1024) throw std::runtime_error("Full-frame working set exceeds 3 GiB; use --tile 256 (changes global attention)");
        wide.reset();
        gate.reset();
        reserve(input, size_t(N) * 3);
        reserve(shallow, size_t(N) * 180);
        reserve(residual, size_t(N) * 180);
        reserve(x, size_t((N + 3) / 4 * 4) * 180);
        reserve(norm, size_t((N + 3) / 4 * 4) * 180);
        const size_t tiles = (size_t((H + 3) / 4) * ((W + 3) / 4) + 3) / 4 * 4;
        reserve(qkv, std::max({size_t(N) * 540, size_t(5400), std::min(tiles, size_t(animeWinogradTiles)) * 36 * 180}));
        reserve(conv, size_t(N) * 180);
        // QKV dies before spatial interaction/FFN; gate dies before convolution.
        wide = qkv;
        gate = conv;
        reserve(mixed, size_t((N + 3) / 4 * 4) * 180);
        const size_t windows = size_t((H + 15) / 16 * 16) * ((W + 15) / 16 * 16) / 128;
        gpu.reserveDispatches(826 + 24 * ((tiles + animeWinogradTiles - 1) / animeWinogradTiles - 1)
                                  + 108 * ((windows + animeAttentionWindows - 1) / animeAttentionWindows - 1));
        reserve(partial, std::max({size_t(chunks) * 5400, std::min(tiles, size_t(animeWinogradTiles)) * 36 * 180, std::min(windows, size_t(animeAttentionWindows)) * 49152}));
        reserve(small0, 180);
        reserve(small1, 180);
        reserve(channelMap, 180);
        reserve(spatialMap, N);
        reserve(channelScores, 5400);
        reserve(norms, 360);
        reserve(output, size_t(N) * scale * scale * 3);
    }
    void op(int kind, BufferPtr a, BufferPtr b, BufferPtr out, Params p, int groups, BufferPtr aux = {}, BufferPtr extra = {})
    {
        int gx = std::min(groups, 65535), gy = (groups + gx - 1) / gx;
        gpu.dispatch("op" + std::to_string(kind), {a, b, out, weights, aux, extra}, p, gx, gy, 1);
    }
    void linear(BufferPtr in, BufferPtr out, const std::string& name, int ic, int oc, int kernel = 1, int activation = 0, BufferPtr add = {}, int count = -1, int stride = 0, int offset = 0, bool packed = false)
    {
        Params p{};
        p[0] = count < 0 ? H : 1;
        p[1] = count < 0 ? W : count;
        p[2] = ic;
        p[3] = oc;
        p[4] = kernel;
        if (kernel == 3 && ic == 180)
        {
            p[5] = at(name + ".winograd");
            p[6] = at(name + ".bias");
            p[8] = ic;
            p[9] = oc;
            p[11] = bool(add);
            const int tiles = ((H + 3) / 4) * ((W + 3) / 4);
            p[13] = (W + 3) / 4;
            p[15] = ic;
            // Bound transform scratch; each batch still reads the full image.
            for (int first = 0; first < tiles; first += animeWinogradTiles)
            {
                p[16] = first;
                p[17] = std::min(animeWinogradTiles, tiles - first);
                p[12] = (p[17] + 3) / 4 * 4;
                int groups = (p[12] * ic + 191) / 192, gx = std::min(groups, 65535);
                gpu.dispatch("conv_win_in", {in, {}, qkv, weights, {}, {}}, p, gx, (groups + gx - 1) / gx, 1);
                gpu.dispatch("conv_win_mul", {qkv, {}, partial, weights, {}, {}}, p, (oc + 15) / 16, 3 * ((p[12] + 15) / 16), 1);
                groups = (p[17] * oc + 191) / 192;
                gx = std::min(groups, 65535);
                gpu.dispatch("conv_win_out", {partial, {}, out, weights, add, {}}, p, gx, (groups + gx - 1) / gx, 1);
            }
            return;
        }
        p[5] = at(name + ".weight");
        p[6] = at(name + ".bias");
        p[7] = activation;
        p[8] = stride ? stride : ic;
        p[9] = offset;
        p[10] = bool(add);
        p[11] = packed;
        int tile = 192;
        gpu.dispatch("gemm_fast",
                     {in, {}, out, weights, add, {}}, p, (oc + 15) / 16, (p[0] * p[1] + tile - 1) / tile, 1);
    }
    void layernorm(BufferPtr in, BufferPtr out, const std::string& name, int stride = 180, int offset = 0, bool packed = false)
    {
        Params p{};
        p[0] = N;
        p[1] = 180;
        p[2] = stride;
        p[3] = offset;
        p[4] = at(name + ".weight");
        p[5] = at(name + ".bias");
        p[6] = packed;
        gpu.dispatch("norm", {in, {}, out, weights, {}, {}}, p, (N + 11) / 12, 1, 1);
    }
    void depthwise(BufferPtr in, BufferPtr out, const std::string& name, bool activation, int stride, int offset, BufferPtr multiply = {})
    {
        Params p{};
        p[0] = H;
        p[1] = W;
        p[2] = 180;
        p[3] = stride;
        p[4] = offset;
        p[5] = at(name + ".weight");
        p[6] = at(name + ".bias");
        p[7] = activation;
        p[8] = bool(multiply);
        p[9] = bool(multiply);
        int groups = (N * 45 + 191) / 192, gx = std::min(groups, 65535);
        gpu.dispatch("depthwise", {in, multiply, out, weights, {}, {}}, p, gx, (groups + gx - 1) / gx, 1);
    }
    void poolMean(BufferPtr in)
    {
        Params p{};
        p[0] = N;
        p[1] = chunks;
        op(2, in, {}, partial, p, (chunks * 180 + 63) / 64);
        op(3, partial, {}, small0, p, 3);
    }
    void execute(int scale) override
    {
        gpu.begin();
        linear(input, shallow, "conv_first", 3, 180, 3);
        dump(shallow, "shallow", N * 180);
        layernorm(shallow, x, "before_RG.1");
        for (int g = 0; g < 6; g++)
        {
            // Save the group residual before updating x in place.
            Params copyParams{};
            copyParams[0] = N * 45;
            const int groups = (copyParams[0] + 191) / 192, gx = std::min(groups, 65535);
            gpu.dispatch("copy", {x, {}, residual, {}, {}, {}}, copyParams,
                         gx, (groups + gx - 1) / gx, 1);
            std::string group = "layers." + std::to_string(g);
            for (int block = 0; block < 6; block++)
            {
                std::string base = group + ".blocks." + std::to_string(block), att = base + ".attn";
                layernorm(x, norm, base + ".norm1", 180, 0, true);
                linear(norm, qkv, att + ".qkv", 180, 540, 1, false, {}, -1, 0, 0, true);
                // Normalized input is dead after QKV; attention dies before projection.
                const auto attention = norm;
                depthwise(qkv, conv, att + ".dwconv.0", true, 540, 360);
                if (block % 2 == 0)
                {
                    bool shift = (g % 2 == 0 && block == 2) || (g % 2 == 1 && block % 4 == 0);
                    for (int branch = 0; branch < 2; branch++)
                    {
                        Params p{};
                        p[0] = H;
                        p[1] = W;
                        p[2] = (H + 15) / 16 * 16;
                        p[3] = (W + 15) / 16 * 16;
                        p[4] = branch ? 16 : 8;
                        p[5] = branch ? 8 : 16;
                        p[6] = shift ? p[4] / 2 : 0;
                        p[7] = shift ? p[5] / 2 : 0;
                        p[8] = branch * 90;
                        p[9] = at(att + ".attns." + std::to_string(branch) + ".relative_bias");
                        {
                            int windows = p[2] * p[3] / 128;
                            // Windows are independent; keep score storage in a bounded batch.
                            for (int first = 0; first < windows; first += animeAttentionWindows)
                            {
                                p[10] = first;
                                p[11] = std::min(animeAttentionWindows, windows - first);
                                gpu.dispatch("window_qk", {qkv, {}, partial, weights, {}, {}}, p, 8, 2, p[11]);
                                gpu.dispatch("softmax", {partial, {}, partial, weights, {}, {}}, p, p[11] * 32, 1, 1);
                                gpu.dispatch("window_av", {qkv, partial, attention, weights, {}, {}}, p, 2, 2, p[11]);
                            }
                        }
                    }
                }
                else
                {
                    Params p{};
                    p[0] = N;
                    p[1] = chunks;
                    op(6, qkv, {}, partial, p, (chunks * 360 + 63) / 64);
                    op(7, partial, {}, norms, p, 6);
                    gpu.dispatch("channel_fast", {qkv, {}, partial, weights, {}, {}}, p, 2, chunks, 1);
                    p = {};
                    p[0] = chunks;
                    p[1] = at(att + ".temperature");
                    op(8, partial, norms, channelScores, p, (5400 + 63) / 64);
                    op(9, channelScores, {}, channelScores, p, 3);
                    p = {};
                    p[0] = N;
                    gpu.dispatch("channel_av", {channelScores, qkv, attention, weights, {}, {}}, p, 2, (N + 191) / 192, 6);
                }
                if (g == 0 && block < 2) dump(attention, "attention" + std::to_string(block), N * 180);
                poolMean(block % 2 == 0 ? conv : attention);
                linear(small0, small1, att + ".channel_interaction.1", 180, 22, 1, true, {}, 1);
                linear(small1, channelMap, att + ".channel_interaction.4", 22, 180, 1, 2, {}, 1);
                linear(block % 2 == 0 ? attention : conv, wide, att + ".spatial_interaction.0", 180, 11, 1, true);
                linear(wide, spatialMap, att + ".spatial_interaction.3", 11, 1, 1, 2);
                Params p{};
                p[0] = N;
                p[1] = block % 2;
                op(4, attention, conv, mixed, p, (((N + 3) / 4) * 180 + 63) / 64, channelMap, spatialMap);
                linear(mixed, norm, att + ".proj", 180, 180, 1, false, x, -1, 0, 0, true);
                std::swap(x, norm);
                layernorm(x, norm, base + ".norm2", 180, 0, true);
                linear(norm, wide, base + ".ffn.fc1", 180, 360, 1, true, {}, -1, 0, 0, true);
                layernorm(wide, gate, base + ".ffn.sg.norm", 360, 180);
                depthwise(gate, mixed, base + ".ffn.sg.conv", false, 180, 0, wide);
                linear(mixed, norm, base + ".ffn.fc2", 180, 180, 1, false, x, -1, 0, 0, true);
                std::swap(x, norm);
                dump(x, "block" + std::to_string(g * 6 + block), N * 180);
            }
            linear(x, norm, group + ".conv", 180, 180, 3, false, residual);
            std::swap(x, norm);
        }
        layernorm(x, norm, "norm");
        linear(norm, x, "conv_after_body", 180, 180, 3, false, shallow);
        // Keep RGB features outside QKV, which the batched transforms reuse.
        linear(x, norm, "upsample.0", 180, 48, 3);
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
        downscale.record(norm, output, H, W, scale, Model::Anime);
    }
};

std::unique_ptr<Network> makeAnimeNetwork(Gpu& gpu)
{ return std::make_unique<DatNetwork>(gpu); }
} // namespace upscaler
