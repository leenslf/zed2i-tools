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

enum class ImageFormat {
    Png,
    Jpg
};

enum class PointCloudFormat {
    Ply,
    Svo
};

struct ConfigOverrides {
    std::optional<bool> enable_frames;
    std::optional<bool> enable_imu;
    std::optional<bool> enable_odometry;
    std::optional<bool> enable_point_cloud;

    std::optional<DepthMode> depth_mode;
    std::optional<uint64_t> serial_number;
    std::optional<std::string> config_path;

    std::optional<bool> enable_recording;
    std::optional<int> recording_duration_sec;
    std::optional<int> recording_frame_limit;
    std::optional<int> recording_frame_stride;
    std::optional<ImageFormat> recording_image_format;
    std::optional<PointCloudFormat> recording_point_cloud_format;
    std::optional<std::string> recording_root;
    std::optional<bool> recording_keyboard_toggle;

    static ConfigOverrides fromArgs(int argc, char** argv);
};

struct Config {
    bool enable_frames = true;
    bool enable_imu = true;
    bool enable_odometry = true;
    bool enable_point_cloud = false;

    DepthMode depth_mode = DepthMode::Quality;

    std::optional<uint64_t> serial_number;

    bool enable_recording = false;
    int recording_duration_sec = 0;
    int recording_frame_limit = 0;
    int recording_frame_stride = 1;
    ImageFormat recording_image_format = ImageFormat::Png;
    PointCloudFormat recording_point_cloud_format = PointCloudFormat::Ply;
    std::string recording_root = "recordings";
    bool recording_keyboard_toggle = false;

    static Config fromFile(const std::string& path);

    void applyOverrides(const ConfigOverrides& overrides);
};

} // namespace zedapp
