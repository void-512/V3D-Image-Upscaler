#include "upscale.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace upscaler {
void upload(Network& dat, Model model, const unsigned char* pixels, int width, int left, int top)
{
    float* dst = (float*)dat.input->mapped;
    const float mean[3] = {0.4488f, 0.4371f, 0.4040f};

    for (int y = 0; y < dat.H; y++)
        for (int x = 0; x < dat.W; x++)
            for (int c = 0; c < 3; c++) dst[(y * dat.W + x) * 3 + c] = pixels[((y + top) * width + x + left) * 3 + c] / 255.f - (model == Model::Anime ? mean[c] : 0.f);
}

namespace {
std::vector<int> cuts(int length, int tile)
{
    const int count = (length + tile - 1) / tile;
    const int step = ((length + count - 1) / count + 15) / 16 * 16;
    std::vector<int> result;
    for (int x = 0; x < length; x += step) result.push_back(x);
    result.push_back(length);
    return result;
}
struct Band
{
    const ScaleSink& sink;
    int width, first = 0, last = 0;
    std::vector<float> values, weights;
    Band(const ScaleSink& output, int w, int rows) : sink(output), width(w * output.scale)
    {
        values.reserve(size_t(rows) * output.scale * width * 3);
        weights.reserve(size_t(rows) * output.scale * width);
    }
    void extend(int end)
    {
        last = end;
        values.resize(size_t(last - first) * width * 3, 0.f);
        weights.resize(size_t(last - first) * width, 0.f);
    }
    void flush(int end)
    {
        const size_t count = size_t(end - first) * width;
        for (size_t i = 0; i < count; ++i)
            for (int c = 0; c < 3; ++c) values[i * 3 + c] /= weights[i];
        sink.rows(values.data(), first, end - first);
        const size_t remaining = size_t(last - end) * width;
        std::memmove(values.data(), values.data() + count * 3, remaining * 3 * sizeof(float));
        std::memmove(weights.data(), weights.data() + count, remaining * sizeof(float));
        values.resize(remaining * 3);
        weights.resize(remaining);
        first = end;
    }
};
}

void upscale(Network& network, const unsigned char* pixels, int width, int height, const Options& options, const std::vector<ScaleSink>& sinks)
{
    if (!options.tile)
    {
        network.execute(options.scale);
        int current = options.scale;
        for (const auto& sink : sinks)
        {
            if (sink.scale != current) network.rescale(sink.scale);
            current = sink.scale;
            sink.rows(static_cast<const float*>(network.output->mapped), 0, height * sink.scale);
        }
        return;
    }
    const int halo = options.halo;
    const int blend = options.model == Model::Live && halo >= 352 ? 0 : halo / 2;
    const auto xs = cuts(width, options.tile), ys = cuts(height, options.tile);
    std::vector<Band> bands;
    for (const auto& sink : sinks) bands.emplace_back(sink, width, std::min(height, options.tile + 2 * blend));
    for (size_t ty = 0; ty + 1 < ys.size(); ++ty)
    {
        const int top = ys[ty], bottom = ys[ty + 1];
        const int y0 = std::max(top - halo, 0), y1 = std::min(bottom + halo, height);
        const int ky0 = std::max(top - blend, 0), ky1 = std::min(bottom + blend, height);
        for (auto& band : bands) band.extend(ky1 * band.sink.scale);
        for (size_t tx = 0; tx + 1 < xs.size(); ++tx)
        {
            const int left = xs[tx], right = xs[tx + 1];
            const int x0 = std::max(left - halo, 0), x1 = std::min(right + halo, width);
            const int kx0 = std::max(left - blend, 0), kx1 = std::min(right + blend, width);
            network.allocate(y1 - y0, x1 - x0, options.scale);
            upload(network, options.model, pixels, width, x0, y0);
            network.execute(options.scale);
            int current = options.scale;
            for (auto& band : bands)
            {
                const int scale = band.sink.scale;
                if (scale != current) network.rescale(scale);
                current = scale;
                const auto* src = static_cast<const float*>(network.output->mapped);
                for (int y = ky0 * scale; y < ky1 * scale; ++y)
                    for (int x = kx0 * scale; x < kx1 * scale; ++x)
                    {
                        const float xx = (x + .5f) / scale, yy = (y + .5f) / scale;
                        const float wx = xx < left ? (xx - kx0) / blend : xx > right ? (kx1 - xx) / blend : 1.f;
                        const float wy = yy < top ? (yy - ky0) / blend : yy > bottom ? (ky1 - yy) / blend : 1.f;
                        const float weight = wx * wy;
                        const size_t dst = size_t(y - band.first) * band.width + x;
                        const size_t source = size_t(y - y0 * scale) * (network.W * scale) + x - x0 * scale;
                        band.weights[dst] += weight;
                        for (int c = 0; c < 3; ++c) band.values[dst * 3 + c] += src[source * 3 + c] * weight;
                    }
            }
            std::cerr << "Tile " << ty * (xs.size() - 1) + tx + 1 << '/' << (ys.size() - 1) * (xs.size() - 1)
                      << " at " << left << ',' << top << " (" << network.W << 'x' << network.H << ")\n";
        }
        const int safe = bottom == height ? height : std::max(bottom - blend, 0);
        for (auto& band : bands) band.flush(safe * band.sink.scale);
    }
}
} // namespace upscaler
