#include "gpu.h"
#include "runtime.h"
#include <algorithm>
#include <bit>
#include <dlfcn.h>

namespace upscaler {
void check(VkResult r)
{
    if (r != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r));
}
namespace {
PFN_vkGetInstanceProcAddrLUNARG loadDriver()
{
    static const auto library = [] {
        const auto path = runtimeRoot() / "libvulkan_broadcom.so";
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(dlerror());
        return std::unique_ptr<void, int(*)(void*)>(handle, dlclose);
    }();
    auto entry = reinterpret_cast<PFN_vkGetInstanceProcAddrLUNARG>(
        dlsym(library.get(), "vk_icdGetInstanceProcAddr"));
    if (!entry) throw std::runtime_error("The bundled V3DV driver has no ICD entry point");
    return entry;
}
}

Gpu::Gpu(Model model, const Config& config)
    : shaderPrefix(std::string(modeName(model)) + "/"), config(config)
{
    const char* extension = VK_LUNARG_DIRECT_DRIVER_LOADING_EXTENSION_NAME;
    VkDirectDriverLoadingInfoLUNARG driver{VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_INFO_LUNARG};
    driver.pfnGetInstanceProcAddr = loadDriver();
    VkDirectDriverLoadingListLUNARG drivers{VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_LIST_LUNARG};
    drivers.mode = VK_DIRECT_DRIVER_LOADING_MODE_EXCLUSIVE_LUNARG;
    drivers.driverCount = 1;
    drivers.pDrivers = &driver;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Upscaler";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pNext = &drivers;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = &extension;
    const auto status = vkCreateInstance(&ci, nullptr, &instance);
    if (status == VK_ERROR_EXTENSION_NOT_PRESENT)
        throw std::runtime_error("The Vulkan loader does not support " + std::string(extension));
    check(status);
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    if (!count) throw std::runtime_error("The bundled V3DV driver found no Vulkan device");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    physical = devices.front();
    vkGetPhysicalDeviceProperties(physical, &properties);
    if (const auto size = uint32_t(config.linearWorkgroupSize);
        size > std::min(properties.limits.maxComputeWorkGroupInvocations, properties.limits.maxComputeWorkGroupSize[0]))
        throw std::runtime_error("performance.linear_workgroup_size exceeds the Vulkan device limit");
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
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    features13.pNext = &features12;
    features13.dynamicRendering = VK_TRUE;
    features12.pNext = &features11;
    features12.shaderFloat16 = VK_TRUE;
    features11.storageBuffer16BitAccess = VK_TRUE;
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
    for (uint32_t i = 0; i < binds.size(); ++i)
    {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = binds.size();
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

uint32_t Gpu::bufferMemoryType(uint32_t memoryTypeBits) const
{
    constexpr auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t type = 0; type < memories.memoryTypeCount; ++type)
        if ((memoryTypeBits & (1u << type)) && (memories.memoryTypes[type].propertyFlags & flags) == flags)
            return type;
    throw std::runtime_error("V3D coherent unified memory is required");
}

VkDeviceSize Gpu::bufferHeapSize() const
{
    // The dummy buffer uses the same creation flags and usage as the inference buffers.
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, dummy->buffer, &requirements);
    const auto type = bufferMemoryType(requirements.memoryTypeBits);
    return memories.memoryHeaps[memories.memoryTypes[type].heapIndex].size;
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
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = bufferMemoryType(mr.memoryTypeBits);
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
    if (name == "gemm_fast" || name == "gemm_fast_half")
        for (int i : {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12})
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (name.rfind("window_", 0) == 0)
        for (int i = 0; i <= 8; i++)
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (name == "depthwise" || name == "depthwise_half")
        for (int i : {0, 1, 3, 4, 7, 8, 9})
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (name.rfind("conv", 0) == 0)
        for (int i : {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21})
        {
            key += "_" + std::to_string(p[i]);
            entries.push_back({uint32_t(i), uint32_t(i * 4), 4});
        }
    if (pipelines.count(key)) return pipelines.at(key);
    Params constants = p;
    const bool transform = name.starts_with("conv_win_in") || name.starts_with("conv_win_out");
    if (transform || name == "copy" || name == "depthwise" || name == "depthwise_half" || name == "pack")
    {
        constexpr uint32_t localSizeId = 23;
        // Transforms retain a 16-wide X dimension and specialize the Y dimension.
        constants[localSizeId] = transform ? config.linearWorkgroupSize / 16 : config.linearWorkgroupSize;
        entries.push_back({localSizeId, localSizeId * sizeof(int), sizeof(int)});
    }
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
    VkSpecializationInfo spec{uint32_t(entries.size()), entries.data(), sizeof(constants), constants.data()};
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
    checkInterrupted();
    check(vkResetCommandPool(device, pool, 0));
    check(vkResetDescriptorPool(device, descriptors, 0));
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
    checkInterrupted();
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
    for (uint32_t i = 0; i < writes.size(); ++i)
    {
        const auto src = i < buffers.size() ? i : i == 6 ? 0 : 3;
        BufferPtr b = buffers[src] ? buffers[src] : dummy;
        info[i] = {b->buffer, 0, b->bytes};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), p.data());
    vkCmdDispatch(cmd, x, y, z);
}

void Gpu::dispatchLinear(const std::string& name, std::array<BufferPtr, 6> buffers, const Params& p, uint32_t elements)
{
    const uint32_t size = config.linearWorkgroupSize;
    const uint32_t groups = (elements + size - 1) / size;
    const uint32_t x = std::min(groups, properties.limits.maxComputeWorkGroupCount[0]);
    dispatch(name, std::move(buffers), p, x, (groups + x - 1) / x);
}

void Gpu::finish()
{
    checkInterrupted();
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    checkInterrupted();
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    // Buffer owners may unwind as soon as cancellation is observed.
    check(vkQueueWaitIdle(queue));
    checkInterrupted();
}
} // namespace upscaler
