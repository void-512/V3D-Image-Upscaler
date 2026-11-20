#pragma once
#include "network.h"
#include "task.h"
#include <functional>

namespace upscaler {
void upload(Network& network, Model model, const unsigned char* pixels, int width, int left, int top);
struct ScaleSink
{
    int scale;
    std::function<void(const float*, int, int)> rows;
};
struct RunTimings
{
    double decode = 0, driver = 0, model = 0, allocate = 0, upload = 0, blend = 0, encode = 0;
};
// Output sinks receive contiguous rows at their requested scale.
void upscale(Network& network, const unsigned char* pixels, int width, int height, const Task& options,
             const std::vector<ScaleSink>& sinks, RunTimings* timings = nullptr);
} // namespace upscaler
