#pragma once

#include "resources.h"
#include <string>

namespace upscaler {
struct Task
{
    Model model = Model::Live;
    int scale = 4, tile = -1, halo = -1;
    bool referenceGraph = false;
    std::string input, output, raw, decoded, trace, profile;
};

constexpr bool validScale(int scale)
{
    return scale == 1 || scale == 2 || scale == 4;
}
} // namespace upscaler
