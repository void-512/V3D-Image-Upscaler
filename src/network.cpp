#include "network.h"
#include <cstring>

namespace upscaler {
namespace {
Resource modelResource(Model model)
{
    return embeddedResource(model == Model::Anime ? "anime/model" : "live/model");
}
}

Network::Network(Gpu& device, Model model, const char* magic) : gpu(device), downscale(device)
{
    const auto blob = modelResource(model);
    size_t offset = 0;
    auto take = [&](size_t count) {
        if (count > blob.size - offset) throw std::runtime_error("Truncated model asset");
        const unsigned char* data = blob.data + offset;
        offset += count;
        return data;
    };
    auto read32 = [&]() {
        uint32_t value;
        std::memcpy(&value, take(4), 4);
        return value;
    };
    if (std::memcmp(take(8), magic, 8)) throw std::runtime_error("Invalid model asset");
    if (read32() != 3) throw std::runtime_error("Packed model format version 3 is required");
    const uint32_t count = read32(), floats = read32();
    const uint32_t halves = read32();
    if (count > 4096 || floats > 70000000 || halves > 70000000 ||
        halves == 0 || halves % 4)
        throw std::runtime_error("Invalid model size");
    for (uint32_t i = 0; i < count; ++i)
    {
        const uint32_t length = read32(), start = read32(), size = read32();
        const uint32_t kind = read32();
        if (length == 0 || length > 512 || kind > 1 ||
            uint64_t(start) + size > (kind == 0 ? floats : halves) ||
            start % 4 || (kind && size % 4))
            throw std::runtime_error("Invalid tensor descriptor");
        std::string name(reinterpret_cast<const char*>(take(length)), length);
        if (tensors.contains(name) || packedTensors.contains(name))
            throw std::runtime_error("Duplicate model tensor " + name);
        (kind == 0 ? tensors : packedTensors).emplace(std::move(name), std::make_pair(start, size));
    }
    const auto* data = take(size_t(floats) * 4);
    const auto* packed = take(size_t(halves) * 2);
    if (offset != blob.size) throw std::runtime_error("Unexpected model data");
    weights = gpu.buffer(floats);
    std::memcpy(weights->mapped, data, size_t(floats) * 4);
    halfWeights = gpu.buffer(halves / 2);
    std::memcpy(halfWeights->mapped, packed, size_t(halves) * 2);
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

MemoryEstimate estimateMemory(Model model, int h, int w, int scale, const Config& config)
{
    auto estimate = model == Model::Anime ? estimateAnimeBuffers(h, w, scale, config)
                                         : estimateLiveBuffers(h, w, scale, config);
    // The model asset includes both weight buffers and a small tensor directory.
    const auto weights = modelResource(model).size;
    // Scales below 4x render a 4H by scale*W RGBA32 Lanczos intermediate.
    const uint64_t texels = scale == 4 ? 0 : uint64_t(h) * w * 4 * scale;
    estimate.totalBytes += weights + texels * 4 * sizeof(float);
    return estimate;
}

} // namespace upscaler
