#include "config.h"
#include "runtime.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <toml++/toml.hpp>

namespace upscaler {
namespace {
int integer(const toml::node& node, std::string_view key, int min, int max, int step = 1, bool allowAuto = false)
{
    const auto* value = node.as_integer();
    if (value && allowAuto && value->get() == -1)
        return -1;
    if (value && value->get() >= min && value->get() <= max && value->get() % step == 0)
        return static_cast<int>(value->get());
    std::string expected = " must be an integer from " + std::to_string(min) + " to " + std::to_string(max);
    if (step > 1) expected += " in multiples of " + std::to_string(step);
    if (allowAuto) expected += ", or -1 for automatic tiling";
    throw std::invalid_argument(std::string(key) + expected);
}
}

Config loadConfig(const std::filesystem::path& requested)
{
    Config config;
    auto file = requested;
    if (file.empty())
    {
        file = "config.toml";
        if (!std::filesystem::exists(file))
            file = runtimeRoot() / file;
        if (!std::filesystem::exists(file))
            return config;
    }

    try
    {
        const auto root = toml::parse_file(file.string());
        for (const auto& [section, node] : root)
        {
            if (section != "performance" && section != "anime" && section != "live")
                throw std::invalid_argument("Unknown section " + std::string(section.str()));
            const auto* table = node.as_table();
            if (!table)
                throw std::invalid_argument(std::string(section.str()) + " must be a table");
            const bool anime = section == "anime";
            auto& tiling = anime ? config.anime : config.live;
            for (const auto& [name, value] : *table)
            {
                const std::string key = std::string(section.str()) + "." + std::string(name.str());
                if (key == "performance.tile_memory_threoshold_mib")
                    config.tileMemoryThresholdMiB = integer(value, key, 1, 3072);
                else if (key == "performance.linear_workgroup_size")
                    config.linearWorkgroupSize = integer(value, key, 16, 256, 16);
                else if (key == "anime.winograd_tiles")
                    config.animeWinogradTiles = integer(value, key, 16, 1024, 16);
                else if (key == "anime.attention_windows")
                    config.animeAttentionWindows = integer(value, key, 1, 256);
                else if (key == "live.max_strip_rows")
                    config.liveMaxStripRows = integer(value, key, 96, 256, 16);
                else if (section != "performance" && name == "tile")
                    tiling.tile = integer(value, key, 0, maxTileDimension, 16, true);
                else if (section != "performance" && name == "tile_size")
                    tiling.tileSize = integer(value, key, 16, maxTileDimension, 16);
                else if (section != "performance" && name == "halo")
                    tiling.halo = integer(value, key, 16, maxTileDimension, 16);
                else
                    throw std::invalid_argument("Unknown setting " + key);
            }
        }
    }
    catch (const toml::parse_error& error)
    {
        throw std::invalid_argument("Config " + file.string() + ": " + std::string(error.description()));
    }
    catch (const std::invalid_argument& error)
    {
        throw std::invalid_argument("Config " + file.string() + ": " + error.what());
    }
    return config;
}
} // namespace upscaler
