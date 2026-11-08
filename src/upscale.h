#pragma once
#include "network.h"
#include "options.h"
#include <functional>

namespace upscaler {
void upload(Network& network, Model model, const unsigned char* pixels, int width, int left, int top);
struct ScaleSink
{
    int scale;
    std::function<void(const float*, int, int)> rows;
};
// Output sinks receive contiguous rows at their requested scale.
void upscale(Network& network, const unsigned char* pixels, int width, int height, const Options& options, const std::vector<ScaleSink>& sinks);
} // namespace upscaler
