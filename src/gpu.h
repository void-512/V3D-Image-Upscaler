#pragma once
#include "resources.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace upscaler {
void check(VkResult result);
struct Buffer
{
    VkDevice device{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    size_t bytes{};
    ~Buffer()
    {
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};
using BufferPtr = std::shared_ptr<Buffer>;
using Params = std::array<int, 24>;

class Gpu
{
public:
    uint32_t dispatchCapacity = 0, descriptorSetsUsed = 0;
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memories{};
    VkCommandPool pool{};
    VkCommandBuffer cmd{};
    VkDescriptorSetLayout setLayout{};
    VkPipelineLayout layout{};
    VkDescriptorPool descriptors{};
    std::map<std::string, VkPipeline> pipelines;
    BufferPtr dummy;
    std::string shaderPrefix;
    uint32_t queueFamily = 0;
    bool uniformInputAlias = false;
    // Optional benchmark instrumentation; empty in normal inference.
    std::function<void(const std::string&, const Params&, bool)> observeDispatch;

    explicit Gpu(Model model, bool performanceQueries = false, bool uniformAlias = false);

    ~Gpu();

    BufferPtr buffer(size_t floats);

    VkPipeline pipeline(const std::string& name, const Params& p);

    void reserveDispatches(uint32_t count);

    void begin();

    void dispatch(const std::string& name, std::array<BufferPtr, 6> buffers, const Params& p, uint32_t x, uint32_t y = 1, uint32_t z = 1);

    void finish();
};

} // namespace upscaler
