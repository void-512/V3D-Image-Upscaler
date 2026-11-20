#pragma once
#include "config.h"

namespace upscaler {
struct Options
{
    bool help = false, version = false;
    bool driverInfo = false;
    Model driverModel = Model::Live;
    Config config;
    std::vector<Task> tasks;
};
Options parseOptions(int argc, char** argv);
const char* usage();
} // namespace upscaler
