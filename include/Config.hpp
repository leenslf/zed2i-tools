#pragma once
// Runtime configuration and feature toggles.

#include <cstdint>
#include <optional>
#include <string>

namespace zedapp {

enum class DepthMode {
    Performance,
    Quality,
    Ultra,
    Neural
};

struct ConfigOverrides {
    std::optional<bool> enable_frames;
    std::optional<bool> enable_imu;
    std::optional<bool> enable_odometry;
    std::optional<bool> enable_point_cloud;

    std::optional<DepthMode> depth_mode;
    std::optional<uint64_t> serial_number;
    std::optional<std::string> config_path;

    static ConfigOverrides fromArgs(int argc, char** argv);
};

struct Config {
    bool enable_frames = true;
    bool enable_imu = true;
    bool enable_odometry = true;
    bool enable_point_cloud = false;

    DepthMode depth_mode = DepthMode::Quality;

    std::optional<uint64_t> serial_number;

    static Config fromFile(const std::string& path);

    void applyOverrides(const ConfigOverrides& overrides);
};

} // namespace zedapp
