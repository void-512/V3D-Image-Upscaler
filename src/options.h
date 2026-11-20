#pragma once
#include "config.h"

namespace upscaler {
struct Options
{
    bool help = false, version = false;
    bool benchmark = false;
    int warmup = 1, loops = 3;
    Config config;
    std::vector<Task> tasks;
};
Options parseOptions(int argc, char** argv);
const char* usage();
} // namespace upscaler
