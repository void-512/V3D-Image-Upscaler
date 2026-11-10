#include "gpu.h"
#include <algorithm>
#include <iostream>
#include <bit>

namespace upscaler {
void check(VkResult r)
{
    if (r != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r));
}
Gpu::Gpu(Model model) : shaderPrefix(std::string(modeName(model)) + "/")
{
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Pi 5 Upscaler";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    check(vkCreateInstance(&ci, nullptr, &instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    for (auto d : devices)
    {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(d, &pr);
        if (pr.vendorID == 0x14e4 && std::string(pr.deviceName).find("V3D") != std::string::npos)
        {
            physical = d;
            properties = pr;
            break;
        }
    }
    if (!physical) throw std::runtime_error("A hardware V3D Vulkan device is required; no CPU fallback is permitted");
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 pr2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    pr2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(physical, &pr2);
    if (sg.subgroupSize != 16 || (sg.supportedOperations & (VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)) != (VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)) throw std::runtime_error("This build requires Pi 5 V3DV with 16-lane shuffle/arithmetic subgroups");
    vkGetPhysicalDeviceMemoryProperties(physical, &memories);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    uint32_t family = 0;
    constexpr auto queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
    while (family < count && (families[family].queueFlags & queueFlags) != queueFlags) family++;
    if (family == count) throw std::runtime_error("No graphics/compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(physical, &features);
    if (!features13.dynamicRendering || !features.features.fragmentStoresAndAtomics)
        throw std::runtime_error("Lanczos requires dynamic rendering and fragment storage writes");
    features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures enabled{};
    enabled.fragmentStoresAndAtomics = VK_TRUE;
    di.pNext = &features13;
    di.pEnabledFeatures = &enabled;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    check(vkCreateDevice(physical, &di, nullptr, &device));
    vkGetDeviceQueue(device, family, 0, &queue);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = family;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device, &pi, nullptr, &pool));
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device, &ca, &cmd));
    std::array<VkDescriptorSetLayoutBinding, 8> binds{};
    for (int i = 0; i < 8; i++)
    {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = 8;
    si.pBindings = binds.data();
    check(vkCreateDescriptorSetLayout(device, &si, nullptr, &setLayout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &setLayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device, &li, nullptr, &layout));
    reserveDispatches(2048);
    dummy = buffer(4);
    std::cerr << "GPU: " << properties.deviceName << "; compute workgroup limit " << properties.limits.maxComputeWorkGroupInvocations << "\n";
}

Gpu::~Gpu()
{
    if (device) vkDeviceWaitIdle(device);
    dummy.reset();
    for (auto& [n, p] : pipelines) vkDestroyPipeline(device, p, nullptr);
    if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
}

BufferPtr Gpu::buffer(size_t floats)
{
    size_t bytes = std::max<size_t>(floats * 4, 16);
    if (bytes > properties.limits.maxStorageBufferRange) throw std::runtime_error("Tensor exceeds V3D storage-buffer range; use --tile");
    auto b = std::make_shared<Buffer>();
    b->device = device;
    b->bytes = bytes;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = bytes;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device, &ci, nullptr, &b->buffer));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, b->buffer, &mr);
    uint32_t type = 0;
    for (; type < memories.memoryTypeCount; type++)
        if ((mr.memoryTypeBits & (1u << type)) && (memories.memoryTypes[type].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    if (type == memories.memoryTypeCount) throw std::runtime_error("V3D coherent unified memory is required");
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    check(vkAllocateMemory(device, &ai, nullptr, &b->memory));
    check(vkBindBufferMemory(device, b->buffer, b->memory, 0));
    check(vkMapMemory(device, b->memory, 0, bytes, 0, &b->mapped));
    return b;
}

VkPipeline Gpu::pipeline(const std::string& name, const Params& p)
{
    std::string key = name;
    std::vector<VkSpecializationMapEntry> entries;
    if (name == "output4")
    {
        key += "_" + std::to_string(p[3]);
        entries.push_back({3, 12, 4});
    }
    if (name == "gemm_fast")
        for (int i : {0, 1, 2, 3, 4, 7, 8, 9, 10, 11})
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (name == "window_qk" || name == "window_av")
        for (int i = 0; i <= 8; i++)
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (name == "depthwise")
        for (int i : {0, 1, 3, 4, 7, 8, 9})
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (name.rfind("conv", 0) == 0)
        for (int i : {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15})
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (pipelines.count(key)) return pipelines.at(key);
    const bool common = name == "output4";
    const Resource code = embeddedResource((common ? "common/" : shaderPrefix) + name);
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size;
    mi.pCode = reinterpret_cast<const uint32_t*>(code.data);
    VkShaderModule module;
    check(vkCreateShaderModule(device, &mi, nullptr, &module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.layout = layout;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    VkSpecializationInfo spec{uint32_t(entries.size()), entries.data(), sizeof(Params), p.data()};
    if (!entries.empty()) ci.stage.pSpecializationInfo = &spec;
    VkPipeline result;
    VkResult status = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &result);
    vkDestroyShaderModule(device, module, nullptr);
    check(status);
    pipelines[key] = result;
    return result;
}

void Gpu::reserveDispatches(uint32_t count)
{
    if (count <= dispatchCapacity) return;
    if (count > 16384) throw std::runtime_error("Too many dispatches; use smaller tiles");
    // Called between completed inferences, never while commands are pending.
    if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    descriptors = VK_NULL_HANDLE;
    dispatchCapacity = std::bit_ceil(count);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 * dispatchCapacity};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = dispatchCapacity;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(device, &dp, nullptr, &descriptors));
}

void Gpu::begin()
{
    check(vkResetCommandPool(device, pool, 0));
    check(vkResetDescriptorPool(device, descriptors, 0));
    descriptorSetsUsed = 0;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &bi));
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Gpu::dispatch(const std::string& name, std::array<BufferPtr, 6> buffers, const Params& p, uint32_t x, uint32_t y, uint32_t z)
{
    if (descriptorSetsUsed >= dispatchCapacity) throw std::runtime_error("Too many dispatches");
    if (x > properties.limits.maxComputeWorkGroupCount[0] || y > properties.limits.maxComputeWorkGroupCount[1] || z > properties.limits.maxComputeWorkGroupCount[2]) throw std::runtime_error("Dispatch exceeds V3D limits; use --tile");
    VkPipeline pipe = pipeline(name, p);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descriptors;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &setLayout;
    VkDescriptorSet ds;
    check(vkAllocateDescriptorSets(device, &ai, &ds));
    std::array<VkDescriptorBufferInfo, 8> info;
    std::array<VkWriteDescriptorSet, 8> writes{};
    for (int i = 0; i < 8; i++)
    {
        int src = i < 6 ? i : i == 6 ? 0
                                     : 3;
        BufferPtr b = buffers[src] ? buffers[src] : dummy;
        info[i] = {b->buffer, 0, b->bytes};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(device, 8, writes.data(), 0, nullptr);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), p.data());
    vkCmdDispatch(cmd, x, y, z);
    ++descriptorSetsUsed;
}

void Gpu::finish()
{
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(queue));
}
} // namespace upscaler
