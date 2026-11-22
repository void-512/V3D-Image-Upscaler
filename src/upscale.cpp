#include "upscale.h"
#include "runtime.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <optional>

namespace upscaler {
void upload(Network& dat, Model model, const unsigned char* pixels, int width, int left, int top)
{
    float* dst = (float*)dat.input->mapped;
    const float mean[3] = {0.4488f, 0.4371f, 0.4040f};

    for (int y = 0; y < dat.H; y++)
    {
        checkInterrupted();
        for (int x = 0; x < dat.W; x++)
            for (int c = 0; c < 3; c++) dst[(y * dat.W + x) * 3 + c] = pixels[((y + top) * width + x + left) * 3 + c] / 255.f - (model == Model::Anime ? mean[c] : 0.f);
    }
}

namespace {
int tileStep(int length, int tile)
{
    const int count = (length + tile - 1) / tile;
    return ((length + count - 1) / count + 15) / 16 * 16;
}
std::vector<int> cuts(int length, int step)
{
    std::vector<int> result;
    for (int x = 0; x < length; x += step) result.push_back(x);
    result.push_back(length);
    return result;
}
int largestTileExtent(int length, int step, int halo)
{
    // The core containing the midpoint loses the least halo to the image edges.
    const int start = length / (2 * step) * step;
    return std::min(start + step + halo, length) - std::max(start - halo, 0);
}
struct Band
{
    const OutputRows& output;
    int width, first = 0, last = 0;
    std::vector<float> values, weights;
    Band(const OutputRows& output, int w, int rows) : output(output), width(w)
    {
        values.reserve(size_t(rows) * width * 3);
        weights.reserve(size_t(rows) * width);
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
        for (size_t row = 0; row < count; row += width)
        {
            checkInterrupted();
            for (size_t i = row; i < row + width; ++i)
                for (int c = 0; c < 3; ++c) values[i * 3 + c] /= weights[i];
        }
        output(values.data(), first, end - first);
        const size_t remaining = size_t(last - end) * width;
        std::memmove(values.data(), values.data() + count * 3, remaining * 3 * sizeof(float));
        std::memmove(weights.data(), weights.data() + count, remaining * sizeof(float));
        values.resize(remaining * 3);
        weights.resize(remaining);
        first = end;
    }
};
}

int resolveTile(const Task& task, const Config& config, int width, int height,
                uint64_t storageBufferLimit, uint64_t heapLimit)
{
    int tile = task.tile;
    if (tile == 0) return tile;

    if (tile == -1)
    {
        const auto full = estimateMemory(task.model, height, width, task.scale, config);
        if (full.largestBufferBytes <= storageBufferLimit && full.totalBytes <= heapLimit) return 0;
        tile = config.tiling(task.model).tileSize;
    }

    const auto memoryLimit = uint64_t(config.tileMemoryThresholdMiB) * 1024 * 1024;
    const auto fits = [&](int core) {
        const int h = largestTileExtent(height, tileStep(height, core), task.halo);
        const int w = largestTileExtent(width, tileStep(width, core), task.halo);
        return estimateMemory(task.model, h, w, task.scale, config).totalBytes <= memoryLimit;
    };
    if (fits(tile)) return tile;

    const auto error = "Tile core size " + std::to_string(tile) + " exceeds the configured RAM limit of "
        + std::to_string(config.tileMemoryThresholdMiB) + " MiB. ";
    // Rebalancing the cuts can shrink the largest tile when the core size increases.
    for (int core = maxTileDimension; core >= 16; core -= 16)
        if (fits(core))
            throw std::runtime_error(error + "Maximum allowed tile core size is " + std::to_string(core)
                                     + " pixels (--tile " + std::to_string(core) + ").");
    throw std::runtime_error(error + "No positive tile size fits with the current settings.");
}

void upscale(Network& network, const unsigned char* pixels, int width, int height, const Task& options,
             const OutputRows& output)
{
    const int scale = options.scale;
    if (!options.tile)
    {
        network.execute(scale);
        if (output)
            output(static_cast<const float*>(network.output->mapped), 0, height * scale);
        return;
    }

    const int halo = options.halo;
    const int blend = options.model == Model::Live && halo >= 352 ? 0 : halo / 2;
    const int xStep = tileStep(width, options.tile), yStep = tileStep(height, options.tile);
    const auto xs = cuts(width, xStep), ys = cuts(height, yStep);
    network.allocate(largestTileExtent(height, yStep, halo), largestTileExtent(width, xStep, halo), scale);
    std::optional<Band> band;
    if (output)
        band.emplace(output, width * scale, std::min(height, options.tile + 2 * blend) * scale);
    for (size_t ty = 0; ty + 1 < ys.size(); ++ty)
    {
        const int top = ys[ty], bottom = ys[ty + 1];
        const int y0 = std::max(top - halo, 0), y1 = std::min(bottom + halo, height);
        const int ky0 = std::max(top - blend, 0), ky1 = std::min(bottom + blend, height);
        if (band) band->extend(ky1 * scale);
        for (size_t tx = 0; tx + 1 < xs.size(); ++tx)
        {
            checkInterrupted();
            const int left = xs[tx], right = xs[tx + 1];
            const int x0 = std::max(left - halo, 0), x1 = std::min(right + halo, width);
            network.allocate(y1 - y0, x1 - x0, scale);
            upload(network, options.model, pixels, width, x0, y0);
            network.execute(scale);
            if (!band) continue;

            const int kx0 = std::max(left - blend, 0), kx1 = std::min(right + blend, width);
            const auto* src = static_cast<const float*>(network.output->mapped);
            for (int y = ky0 * scale; y < ky1 * scale; ++y)
            {
                checkInterrupted();
                for (int x = kx0 * scale; x < kx1 * scale; ++x)
                {
                    const float xx = (x + .5f) / scale, yy = (y + .5f) / scale;
                    const float wx = xx < left ? (xx - kx0) / blend : xx > right ? (kx1 - xx) / blend : 1.f;
                    const float wy = yy < top ? (yy - ky0) / blend : yy > bottom ? (ky1 - yy) / blend : 1.f;
                    const float weight = wx * wy;
                    const size_t dst = size_t(y - band->first) * band->width + x;
                    const size_t source = size_t(y - y0 * scale) * (network.W * scale) + x - x0 * scale;
                    band->weights[dst] += weight;
                    for (int c = 0; c < 3; ++c) band->values[dst * 3 + c] += src[source * 3 + c] * weight;
                }
            }
            std::cerr << "Tile " << ty * (xs.size() - 1) + tx + 1 << '/' << (ys.size() - 1) * (xs.size() - 1)
                      << " at " << left << ',' << top << " (" << network.W << 'x' << network.H << ")\n";
        }
        if (band) band->flush((bottom == height ? height : std::max(bottom - blend, 0)) * scale);
    }
}
} // namespace upscaler
