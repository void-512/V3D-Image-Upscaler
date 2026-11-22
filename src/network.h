#pragma once

#include "downscale.h"

namespace upscaler {

class Network
{
public:
    virtual ~Network() = default;
    virtual void allocate(int height, int width, int scale) = 0;
    virtual void execute(int scale) = 0;

    BufferPtr input, output;
    int H = 0, W = 0, N = 0;

protected:
    Network(Gpu& gpu, Model model, const char* magic);
    void reserve(BufferPtr& buffer, size_t floats);
    int at(const std::string& name) const;

    Gpu& gpu;
    Downscale downscale;
    BufferPtr weights, halfWeights;
    std::map<std::string, std::pair<int, int> > tensors;
    std::map<std::string, std::pair<int, int> > packedTensors;
};

inline size_t winogradTileCount(int height, int width)
{
    const size_t tiles = size_t((height + 3) / 4) * ((width + 3) / 4);
    return (tiles + 3) / 4 * 4;
}

struct MemoryEstimate
{
    uint64_t totalBytes = 0;
    uint64_t largestBufferBytes = 0;
};

template<class BufferSizes>
MemoryEstimate estimateBuffers(const BufferSizes& sizes)
{
    MemoryEstimate estimate;
    for (const auto& [buffer, elements] : sizes)
    {
        const uint64_t bytes = elements * sizeof(float);
        estimate.totalBytes += bytes;
        if (bytes > estimate.largestBufferBytes) estimate.largestBufferBytes = bytes;
    }
    return estimate;
}

MemoryEstimate estimateMemory(Model model, int height, int width, int scale, const Config& config);
MemoryEstimate estimateAnimeBuffers(int height, int width, int scale, const Config& config);
MemoryEstimate estimateLiveBuffers(int height, int width, int scale, const Config& config);
std::unique_ptr<Network> makeAnimeNetwork(Gpu& gpu);
std::unique_ptr<Network> makeLiveNetwork(Gpu& gpu);

} // namespace upscaler
