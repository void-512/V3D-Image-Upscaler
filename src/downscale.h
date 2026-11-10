#pragma once
#include "gpu.h"

namespace upscaler {
// Separable Lanczos3 in the inference command buffer, with no CPU round trip.
class Downscale
{
public:
    explicit Downscale(Gpu& device) : gpu(device) {}
    ~Downscale();
    Downscale(const Downscale&) = delete;
    Downscale& operator=(const Downscale&) = delete;
    void record(BufferPtr source, BufferPtr output, int height, int width, int scale, Model model);

private:
    void initialize();
    void reserveImage(int width, int height);
    VkPipeline pipeline(const Params& params, int pass);
    Gpu& gpu;
    VkImage image{};
    VkImageView view{};
    VkDeviceMemory memory{};
    int imageWidth = 0, imageHeight = 0;
    VkSampler sampler{};
    VkDescriptorSetLayout descriptors{};
    VkPipelineLayout layout{};
    VkDescriptorPool pool{};
    VkDescriptorSet sets[2]{};
    std::map<Params, VkPipeline> pipelines;
};
} // namespace upscaler
