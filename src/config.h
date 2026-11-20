#pragma once

#include "resources.h"
#include <filesystem>

namespace upscaler {
struct TileConfig
{
    int tile = -1;
    int tileSize;
    int halo;
};

struct Config
{
    int tileMemoryThresholdMiB = 2048;
    int linearWorkgroupSize = 192;
    TileConfig anime{-1, 448, 48};
    TileConfig live{-1, 576, 96};
    int animeWinogradTiles = 512;
    int animeAttentionWindows = 64;
    int liveMaxStripRows = 256;

    const TileConfig& tiling(Model model) const { return model == Model::Anime ? anime : live; }
};

inline constexpr int maxTileDimension = 1048576;
Config loadConfig(const std::filesystem::path& requested);
} // namespace upscaler
