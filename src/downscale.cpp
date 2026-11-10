#include "downscale.h"
#include <algorithm>

namespace upscaler {
namespace {
constexpr VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
VkShaderModule module(Gpu& gpu, const char* name)
{
    const auto code = embeddedResource(name);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size;
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data);
    VkShaderModule result;
    check(vkCreateShaderModule(gpu.device, &ci, nullptr, &result));
    return result;
}
}
Downscale::~Downscale()
{
    for (const auto& [key, pipe] : pipelines) vkDestroyPipeline(gpu.device, pipe, nullptr);
    if (pool) vkDestroyDescriptorPool(gpu.device, pool, nullptr);
    if (layout) vkDestroyPipelineLayout(gpu.device, layout, nullptr);
    if (descriptors) vkDestroyDescriptorSetLayout(gpu.device, descriptors, nullptr);
    if (sampler) vkDestroySampler(gpu.device, sampler, nullptr);
    if (view) vkDestroyImageView(gpu.device, view, nullptr);
    if (image) vkDestroyImage(gpu.device, image, nullptr);
    if (memory) vkFreeMemory(gpu.device, memory, nullptr);
}
void Downscale::initialize()
{
    if (layout) return;
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(gpu.physical, format, &properties);
    constexpr auto required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((properties.optimalTilingFeatures & required) != required) throw std::runtime_error("FP32 Lanczos render/texture format unavailable");
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    di.bindingCount = 3; di.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(gpu.device, &di, nullptr, &descriptors));
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &descriptors;
    check(vkCreatePipelineLayout(gpu.device, &li, nullptr, &layout));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 2; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    check(vkCreateDescriptorPool(gpu.device, &pi, nullptr, &pool));
    VkDescriptorSetLayout layouts[] = {descriptors, descriptors};
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = 2; ai.pSetLayouts = layouts;
    check(vkAllocateDescriptorSets(gpu.device, &ai, sets));
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(gpu.device, &si, nullptr, &sampler));
}
void Downscale::reserveImage(int width, int height)
{
    if (width <= imageWidth && height <= imageHeight) return;
    if (view) vkDestroyImageView(gpu.device, view, nullptr);
    if (image) vkDestroyImage(gpu.device, image, nullptr);
    if (memory) vkFreeMemory(gpu.device, memory, nullptr);
    view = VK_NULL_HANDLE; image = VK_NULL_HANDLE; memory = VK_NULL_HANDLE;
    // Avoid a large cross-product when a wide tile follows a tall tile.
    imageWidth = width; imageHeight = height;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = format;
    ci.extent = {uint32_t(width), uint32_t(height), 1};
    ci.mipLevels = ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    check(vkCreateImage(gpu.device, &ci, nullptr, &image));
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(gpu.device, image, &requirements);
    uint32_t type = 0;
    while (!(requirements.memoryTypeBits & (1u << type))) ++type;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = requirements.size; ai.memoryTypeIndex = type;
    check(vkAllocateMemory(gpu.device, &ai, nullptr, &memory));
    check(vkBindImageMemory(gpu.device, image, memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(gpu.device, &vi, nullptr, &view));
}
VkPipeline Downscale::pipeline(const Params& params, int pass)
{
    Params key = params; key[6] = pass;
    if (auto found = pipelines.find(key); found != pipelines.end()) return found->second;
    auto vertex = module(gpu, "common/fullscreen"), fragment = module(gpu, pass == 1 ? "common/lanczos_h" : "common/lanczos_v");
    VkSpecializationMapEntry entries[6];
    for (uint32_t i = 0; i < 6; ++i) entries[i] = {i, i * 4, 4};
    VkSpecializationInfo spec{6, entries, sizeof(Params), params.data()};
    VkPipelineShaderStageCreateInfo stages[] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", &spec}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attachment{}; attachment.colorWriteMask = 15;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = pass == 1 ? 1 : 0; blend.pAttachments = &attachment;
    VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2; dynamic.pDynamicStates = states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = blend.attachmentCount; rendering.pColorAttachmentFormats = &format;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &rendering; ci.stageCount = 2; ci.pStages = stages;
    ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia; ci.pViewportState = &vp;
    ci.pRasterizationState = &rs; ci.pMultisampleState = &ms; ci.pColorBlendState = &blend;
    ci.pDynamicState = &dynamic; ci.layout = layout;
    VkPipeline result;
    const auto status = vkCreateGraphicsPipelines(gpu.device, VK_NULL_HANDLE, 1, &ci, nullptr, &result);
    vkDestroyShaderModule(gpu.device, vertex, nullptr); vkDestroyShaderModule(gpu.device, fragment, nullptr);
    check(status);
    pipelines.emplace(key, result);
    return result;
}
void Downscale::record(BufferPtr source, BufferPtr output, int height, int width, int scale, Model model)
{
    Params p{};
    p[0] = height * 4; p[1] = width * 4; p[2] = 4 / scale; p[3] = model == Model::Anime;
    if (scale == 4)
    {
        const uint32_t groups = uint32_t((size_t(height) * width * 48 + 63) / 64), x = std::min(groups, 65535u);
        gpu.dispatch("output4", {source, {}, output, {}, {}, {}}, p, x, (groups + x - 1) / x);
        return;
    }
    initialize();
    const auto& limits = gpu.properties.limits;
    const int limit = int(std::min({limits.maxImageDimension2D, limits.maxFramebufferWidth, limits.maxFramebufferHeight}));
    const int ow = width * scale;
    // Flatten extreme aspect ratios to fit framebuffer limits without a fallback.
    auto rasterWidth = [&](int rows) {
        const int result = std::max(std::min(ow, limit), int((int64_t(ow) * rows + limit - 1) / limit));
        if (result > limit) throw std::runtime_error("Lanczos intermediate exceeds image limits; use --tile");
        return result;
    };
    p[4] = rasterWidth(height * 4);
    const int imageRows = (int64_t(ow) * height * 4 + p[4] - 1) / p[4];
    reserveImage(p[4], imageRows);
    VkDescriptorBufferInfo buffers[] = {{source->buffer, 0, source->bytes}, {output->buffer, 0, output->bytes}};
    VkDescriptorImageInfo texture{sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets[0], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffers[0], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets[1], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &texture, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets[1], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffers[1], nullptr}};
    // record() is called once per completed inference/rescale submission.
    vkUpdateDescriptorSets(gpu.device, 3, writes, 0, nullptr);
    VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(gpu.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &ready, 0, nullptr, 0, nullptr);
    VkImageMemoryBarrier imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // Every required pixel is overwritten.
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image; imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(gpu.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    for (int pass = 1; pass <= 2; ++pass)
    {
        const int rows = pass == 1 ? height * 4 : height * scale;
        p[5] = pass == 1 ? p[4] : rasterWidth(rows);
        VkExtent2D extent{uint32_t(p[5]), uint32_t((int64_t(ow) * rows + p[5] - 1) / p[5])};
        const auto pipe = pipeline(p, pass);
        VkRenderingAttachmentInfo attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        attachment.imageView = view; attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea.extent = extent; rendering.layerCount = 1;
        rendering.colorAttachmentCount = pass == 1 ? 1 : 0; rendering.pColorAttachments = &attachment;
        vkCmdBeginRendering(gpu.cmd, &rendering);
        vkCmdBindPipeline(gpu.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        VkViewport viewport{0, 0, float(extent.width), float(extent.height), 0, 1};
        VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(gpu.cmd, 0, 1, &viewport); vkCmdSetScissor(gpu.cmd, 0, 1, &scissor);
        vkCmdBindDescriptorSets(gpu.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &sets[pass - 1], 0, nullptr);
        vkCmdDraw(gpu.cmd, 3, 1, 0, 0);
        vkCmdEndRendering(gpu.cmd);
        if (pass == 1)
        {
            imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; imageBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(gpu.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
        }
    }
}
} // namespace upscaler
