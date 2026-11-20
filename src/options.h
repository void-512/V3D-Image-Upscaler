#pragma once
#include "resources.h"
#include <string>

namespace upscaler {
struct Options
{
    Model model = Model::Live;
    int scale = 4, tile = -1, halo = -1;
    bool help = false, version = false;
    bool systemDriver = false, fp32 = false, driverInfo = false, referenceGraph = false;
    std::string input, output, raw, decoded, trace;
    std::string profile;
};
Options parseOptions(int argc, char** argv);
const char* usage();
} // namespace upscaler
