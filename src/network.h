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

int liveStripRows(int height, int width, const Config& config);
uint64_t estimatedMemory(Model model, int height, int width, int scale, const Config& config);
std::unique_ptr<Network> makeAnimeNetwork(Gpu& gpu);
std::unique_ptr<Network> makeLiveNetwork(Gpu& gpu);

} // namespace upscaler
