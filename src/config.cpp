#include "config.h"
#include "runtime.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <toml++/toml.hpp>
#include <utility>

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

std::string taskString(const toml::node_view<const toml::node>& node, const std::string& key)
{
    const auto* value = node.as_string();
    if (!value || value->get().empty() || value->get().find('\0') != std::string::npos)
        throw std::invalid_argument(key + " must be a nonempty string");
    return value->get();
}

void appendTasks(const toml::node& node, const std::filesystem::path& directory, std::vector<Task>& tasks)
{
    const auto* entries = node.as_array();
    if (!entries) throw std::invalid_argument("tasks must be an array of tables");
    size_t index = 0;
    for (const auto& entry : *entries)
    {
        const std::string key = "tasks[" + std::to_string(++index) + "]";
        const auto* table = entry.as_table();
        if (!table) throw std::invalid_argument(key + " must be a table");
        for (const auto& [name, value] : *table)
            if (name != "input" && name != "output" && name != "type" && name != "scale")
                throw std::invalid_argument("Unknown setting " + key + "." + std::string(name.str()));

        Task task;
        const auto type = taskString((*table)["type"], key + ".type");
        if (type == "anime") task.model = Model::Anime;
        else if (type == "live") task.model = Model::Live;
        else throw std::invalid_argument(key + ".type must be anime or live");
        if (const auto* scale = table->get("scale"))
        {
            task.scale = integer(*scale, key + ".scale", 1, 4);
            if (!validScale(task.scale)) throw std::invalid_argument(key + ".scale must be 1, 2 or 4");
        }

        const auto input = directory / taskString((*table)["input"], key + ".input");
        task.input = input.string();
        if (const auto output = (*table)["output"])
            task.output = (directory / taskString(output, key + ".output")).string();
        else
            task.output = (input.parent_path() / (input.stem().string() + "-" + std::to_string(task.scale) + "x.png")).string();
        tasks.push_back(std::move(task));
    }
}
}

Config loadConfig(const std::filesystem::path& requested, std::vector<Task>& tasks)
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
            if (section == "tasks")
            {
                appendTasks(node, file.parent_path(), tasks);
                continue;
            }
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
