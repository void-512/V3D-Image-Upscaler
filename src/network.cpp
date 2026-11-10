#include "network.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace upscaler {

Network::Network(Gpu& device, Model model, const char* magic) : gpu(device), downscale(device)
{
    const Resource blob = embeddedResource(std::string(modeName(model)) + "/model");
    size_t offset = 0;
    auto take = [&](size_t count) {
        if (count > blob.size - offset) throw std::runtime_error("Truncated embedded model");
        const unsigned char* data = blob.data + offset;
        offset += count;
        return data;
    };
    auto read32 = [&]() {
        uint32_t value;
        std::memcpy(&value, take(4), 4);
        return value;
    };
    if (std::memcmp(take(8), magic, 8) || read32() != 1) throw std::runtime_error("Invalid embedded model");
    const uint32_t count = read32(), floats = read32();
    if (count > 4096 || floats > 70000000) throw std::runtime_error("Invalid embedded model size");
    for (uint32_t i = 0; i < count; ++i)
    {
        const uint32_t length = read32(), start = read32(), size = read32();
        if (length > 512 || uint64_t(start) + size > floats) throw std::runtime_error("Invalid embedded tensor");
        std::string name(reinterpret_cast<const char*>(take(length)), length);
        tensors.emplace(std::move(name), std::make_pair(start, size));
    }
    const auto* data = take(size_t(floats) * 4);
    if (offset != blob.size) throw std::runtime_error("Unexpected embedded model data");
    weights = gpu.buffer(floats);
    std::memcpy(weights->mapped, data, size_t(floats) * 4);
}

void Network::reserve(BufferPtr& buffer, size_t floats)
{
    if (!buffer || buffer->bytes < floats * 4)
    {
        buffer.reset();
        buffer = gpu.buffer(floats);
    }
}

int Network::at(const std::string& name) const
{
    return tensors.at(name).first;
}

void Network::dump(BufferPtr buffer, const std::string& name, int count)
{
    if (trace.empty()) return;
    gpu.finish();
    std::filesystem::create_directories(trace);
    std::ofstream file(std::filesystem::path(trace) / (name + ".f32"), std::ios::binary);
    if (!file.write(static_cast<char*>(buffer->mapped), size_t(count) * 4)) throw std::runtime_error("Cannot write trace");
    gpu.begin();
}

int liveStripRows(int h, int w)
{
    // Fit the tail and its halos in the trunk's dense and transform buffers.
    const int tiles = (((h + 3) / 4) * ((w + 3) / 4) + 3) / 4 * 4;
    const int rows = std::min(h * 3 / 4 - 2, int(int64_t(tiles) * 9 / w) - 4);
    return std::clamp(rows / 16 * 16, 96, 256);
}

uint64_t estimatedMemory(Model model, int h, int w, int scale)
{
    // 4H * 2W * RGBA32: reserve for the largest Lanczos intermediate, also
    // when initial 4x inference is followed by additional output scales.
    constexpr int outputScratch = 32;
    if (model == Model::Anime)
    {
        const uint64_t n = uint64_t(h) * w, padded = (n + 3) / 4 * 4;
        const uint64_t tiles = (uint64_t((h + 3) / 4) * ((w + 3) / 4) + 3) / 4 * 4;
        const uint64_t win = std::min(tiles, uint64_t(animeWinogradTiles)) * 36 * 180;
        const uint64_t windows = uint64_t((h + 15) / 16 * 16) * ((w + 15) / 16 * 16) / 128;
        return (n * (544 + scale * scale * 3 + outputScratch) + padded * 360
                + std::max({n * 540, uint64_t(5400), win})
                + padded * 180
                + std::max({(n + 255) / 256 * 5400, win, std::min(windows, uint64_t(animeAttentionWindows)) * 49152})) * 4
               + 96 * 1024 * 1024;
    }
    const uint64_t pixels = uint64_t(h) * w;
    const uint64_t tiles = (uint64_t((h + 3) / 4) * ((w + 3) / 4) + 3) / 4 * 4;
    const int strip = liveStripRows(h, w);
    return pixels * (3 + 3 * 64 + scale * scale * 3 + outputScratch) * 4
           + std::max(pixels * 192, uint64_t(std::min(h * 4, strip + 2)) * w * 256) * 4
           + (std::max(pixels * 256, tiles * 36 * 192)
              + std::max(uint64_t(std::min(h * 4, strip + 4)) * w * 256, tiles * 36 * 64)) * 4
           + 272 * 1024 * 1024;
}

} // namespace upscaler
