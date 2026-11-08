#pragma once
#include "resources.h"
#include <string>

namespace upscaler {
struct Options
{
    Model model = Model::Live;
    int scale = 4, tile = -1, halo = -1;
    bool help = false, version = false;
    std::string input, output, raw, decoded, trace;
};
Options parseOptions(int argc, char** argv);
const char* usage();
} // namespace upscaler
