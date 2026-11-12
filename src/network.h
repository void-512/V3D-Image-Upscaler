#pragma once

#include "downscale.h"

namespace upscaler {

inline constexpr int animeWinogradTiles = 512;
inline constexpr int animeAttentionWindows = 64;

class Network
{
public:
    virtual ~Network() = default;
    virtual void allocate(int height, int width, int scale) = 0;
    virtual void execute(int scale) = 0;
    virtual void rescale(int scale) = 0;

    BufferPtr input, output;
    int H = 0, W = 0, N = 0;
    std::string trace;

protected:
    Network(Gpu& gpu, Model model, const char* magic);
    void reserve(BufferPtr& buffer, size_t floats);
    int at(const std::string& name) const;
    void dump(BufferPtr buffer, const std::string& name, int count);

    Gpu& gpu;
    uint32_t modelVersion = 0;
    Downscale downscale;
    BufferPtr weights;
    std::map<std::string, std::pair<int, int> > tensors;
};

int liveStripRows(int height, int width);
uint64_t estimatedMemory(Model model, int height, int width, int scale);
std::unique_ptr<Network> makeAnimeNetwork(Gpu& gpu);
std::unique_ptr<Network> makeLiveNetwork(Gpu& gpu);

} // namespace upscaler
