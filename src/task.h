#pragma once

#include "resources.h"
#include <cstdint>
#include <string>

namespace upscaler {
struct Task
{
    Model model = Model::Live;
    int scale = 4, tile = -1, halo = -1;
    std::string input, output;
};

constexpr bool validScale(int scale)
{
    return scale == 1 || scale == 2 || scale == 4;
}

constexpr bool validImageDimensions(int width, int height)
{
    // A 4x RGB output must fit in the shaders' 32-bit signed indexing.
    return width > 0 && height > 0 && int64_t(width) * height <= INT32_MAX / 48;
}
} // namespace upscaler
