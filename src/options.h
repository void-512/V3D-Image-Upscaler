#pragma once
#include "config.h"

namespace upscaler {
struct Options
{
    bool help = false, version = false;
    bool benchmark = false;
    int benchmarkWidth = 0, benchmarkHeight = 0;
    int warmup = 1, loops = 3;
    Config config;
    std::vector<Task> tasks;
};
Options parseOptions(int argc, char** argv);
const char* usage();
} // namespace upscaler
