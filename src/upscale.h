#pragma once
#include "network.h"
#include "task.h"
#include <functional>

namespace upscaler {
int resolveTile(const Task& task, const Config& config, int width, int height,
                uint64_t storageBufferLimit, uint64_t heapLimit);
void upload(Network& network, Model model, const unsigned char* pixels, int width, int left, int top);
using OutputRows = std::function<void(const float*, int, int)>;
void upscale(Network& network, const unsigned char* pixels, int width, int height, const Task& options,
             const OutputRows& output = {});
} // namespace upscaler
