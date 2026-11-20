#pragma once

#include "gpu.h"

#include <fstream>
#include <iostream>

namespace upscaler {
class DispatchProfile {
public:
    DispatchProfile(Gpu& gpu, const std::string& path) : gpu(gpu)
    {
        if (path.empty())
            return;
        file.open(path);
        if (!file)
            throw std::runtime_error("Could not open the dispatch timing file");
        file << "shader,gpu_ms,h,w,ic,oc,kernel,input_packed,tiles,rows,batch\n";
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &count, families.data());
        uint32_t bits = families.at(gpu.queueFamily).timestampValidBits;
        if (!bits)
            throw std::runtime_error("No compute timestamps");
        mask = bits == 64 ? UINT64_MAX : (uint64_t(1) << bits) - 1;
        VkQueryPoolCreateInfo create{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        create.queryType = VK_QUERY_TYPE_TIMESTAMP;
        create.queryCount = capacity;
        check(vkCreateQueryPool(gpu.device, &create, nullptr, &pool));
        gpu.observeDispatch = [this](const std::string& name, const Params& p, bool before) {
            if (next >= capacity)
                throw std::runtime_error("Dispatch timestamp pool exhausted");
            if (next == 0)
                vkCmdResetQueryPool(this->gpu.cmd, pool, 0, capacity);
            if (before)
                dispatches.push_back({name, p});
            vkCmdWriteTimestamp(this->gpu.cmd, before ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
                                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, next++);
        };
        gpu.observeBatch = [this](double recordMs, double waitMs) { writeBatch(recordMs, waitMs); };
    }

    ~DispatchProfile()
    {
        gpu.observeDispatch = {};
        gpu.observeBatch = {};
        if (pool)
            vkDestroyQueryPool(gpu.device, pool, nullptr);
    }

private:
    void writeBatch(double recordMs, double waitMs)
    {
        if (!pool || !next)
            return;
        std::vector<uint64_t> stamps(next);
        check(vkGetQueryPoolResults(gpu.device, pool, 0, next, stamps.size()*sizeof(uint64_t),
               stamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
        ++batch;
        const auto milliseconds = [&](uint64_t first, uint64_t last) {
            return double((last - first) & mask) * gpu.properties.limits.timestampPeriod / 1e6;
        };
        for (size_t i = 0; i < dispatches.size(); i++) {
            const auto& [name, p] = dispatches[i];
            const double ms = milliseconds(stamps[2*i], stamps[2*i+1]);
            file << name << ',' << ms << ',' << p[0] << ',' << p[1] << ',' << p[2] << ',' << p[3]
                 << ',' << p[4] << ',' << p[11] << ',' << p[12] << ',' << p[17] << ',' << batch << '\n';
        }
        file.flush();
        if (!file) throw std::runtime_error("Could not write dispatch timings");
        std::cerr << "Profile batch," << batch << ',' << recordMs << ',' << waitMs << ','
                  << milliseconds(stamps.front(), stamps.back()) << '\n';
        dispatches.clear();
        next = 0;
    }

    static constexpr uint32_t capacity = 32768;
    Gpu& gpu;
    VkQueryPool pool{};
    uint64_t mask{};
    uint32_t next{};
    uint32_t batch{};
    std::ofstream file;
    std::vector<std::pair<std::string, Params>> dispatches;
};
}
