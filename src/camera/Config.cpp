#include "camera/Config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "camera/Logger.hpp"

namespace zedapp {
namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string trim(const std::string& value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

bool parseBool(const std::string& value) {
    const auto lower = toLower(trim(value));
    return lower == "true" || lower == "1" || lower == "yes" || lower == "on";
}

DepthMode parseDepthMode(const std::string& value) {
    const auto upper = toLower(trim(value));
    if (upper == "performance") {
        return DepthMode::Performance;
    }
    if (upper == "quality") {
        return DepthMode::Quality;
    }
    if (upper == "ultra") {
        return DepthMode::Ultra;
    }
    if (upper == "neural") {
        return DepthMode::Neural;
    }
    throw std::runtime_error("Unknown depth mode: " + value);
}

CameraResolution parseCameraResolution(const std::string& value) {
    const auto upper = toLower(trim(value));
    if (upper == "hd2k") {
        return CameraResolution::Hd2k;
    }
    if (upper == "hd1080") {
        return CameraResolution::Hd1080;
    }
    if (upper == "hd720") {
        return CameraResolution::Hd720;
    }
    if (upper == "vga") {
        return CameraResolution::Vga;
    }
    throw std::runtime_error("Unknown camera resolution: " + value);
}


} // namespace

Config Config::fromFile(const std::string& path) {
    Config config;

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        const auto trimmed = trim(line);
        if (trimmed.empty() || trimmed.rfind("#", 0) == 0) {
            continue;
        }
        const auto eq = trimmed.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const auto key = toLower(trim(trimmed.substr(0, eq)));
        const auto value = trim(trimmed.substr(eq + 1));

        if (key == "enable_frames") {
            config.enable_frames = parseBool(value);
        } else if (key == "enable_imu") {
            config.enable_imu = parseBool(value);
        } else if (key == "enable_odometry") {
            config.enable_odometry = parseBool(value);
        } else if (key == "enable_point_cloud") {
            config.enable_point_cloud = parseBool(value);
        } else if (key == "enable_recording") {
            config.enable_recording = parseBool(value);
        } else if (key == "recording_duration_sec") {
            config.recording_duration_sec = std::stoi(value);
        } else if (key == "recording_root") {
            config.recording_root = value;
        } else if (key == "recording_keyboard_toggle") {
            config.recording_keyboard_toggle = parseBool(value);
        } else if (key == "depth_mode") {
            config.depth_mode = parseDepthMode(value);
        } else if (key == "camera_resolution") {
            config.camera_resolution = parseCameraResolution(value);
        } else if (key == "serial_number") {
            config.serial_number = static_cast<uint64_t>(std::stoull(value));
        } else if (key == "log_level") {
            const auto level = toLower(value);
            if (level == "debug") {
                Logger::setMinLevel(LogLevel::Debug);
            } else if (level == "info") {
                Logger::setMinLevel(LogLevel::Info);
            } else if (level == "warn") {
                Logger::setMinLevel(LogLevel::Warn);
            } else if (level == "error") {
                Logger::setMinLevel(LogLevel::Error);
            }
        }
    }

    return config;
}

ConfigOverrides ConfigOverrides::fromArgs(int argc, char** argv) {
    ConfigOverrides overrides;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            overrides.config_path = argv[++i];
        } else if (arg == "--enable-frames") {
            overrides.enable_frames = true;
        } else if (arg == "--disable-frames") {
            overrides.enable_frames = false;
        } else if (arg == "--enable-imu") {
            overrides.enable_imu = true;
        } else if (arg == "--disable-imu") {
            overrides.enable_imu = false;
        } else if (arg == "--enable-odometry") {
            overrides.enable_odometry = true;
        } else if (arg == "--disable-odometry") {
            overrides.enable_odometry = false;
        } else if (arg == "--enable-point-cloud") {
            overrides.enable_point_cloud = true;
        } else if (arg == "--disable-point-cloud") {
            overrides.enable_point_cloud = false;
        } else if (arg == "--record") {
            overrides.enable_recording = true;
        } else if (arg == "--no-record") {
            overrides.enable_recording = false;
        } else if (arg == "--record-toggle") {
            overrides.recording_keyboard_toggle = true;
        } else if (arg.rfind("--record-duration=", 0) == 0) {
            overrides.recording_duration_sec = std::stoi(arg.substr(std::string("--record-duration=").size()));
        } else if (arg.rfind("--record-root=", 0) == 0) {
            overrides.recording_root = arg.substr(std::string("--record-root=").size());
        } else if (arg.rfind("--depth-mode=", 0) == 0) {
            overrides.depth_mode = parseDepthMode(arg.substr(std::string("--depth-mode=").size()));
        } else if (arg.rfind("--camera-resolution=", 0) == 0) {
            overrides.camera_resolution =
                parseCameraResolution(arg.substr(std::string("--camera-resolution=").size()));
        } else if (arg.rfind("--serial=", 0) == 0) {
            overrides.serial_number = static_cast<uint64_t>(
                std::stoull(arg.substr(std::string("--serial=").size())));
        }
    }

    return overrides;
}

void Config::applyOverrides(const ConfigOverrides& overrides) {
    if (overrides.enable_frames.has_value()) {
        enable_frames = *overrides.enable_frames;
    }
    if (overrides.enable_imu.has_value()) {
        enable_imu = *overrides.enable_imu;
    }
    if (overrides.enable_odometry.has_value()) {
        enable_odometry = *overrides.enable_odometry;
    }
    if (overrides.enable_point_cloud.has_value()) {
        enable_point_cloud = *overrides.enable_point_cloud;
    }
    if (overrides.enable_recording.has_value()) {
        enable_recording = *overrides.enable_recording;
    }
    if (overrides.recording_duration_sec.has_value()) {
        recording_duration_sec = *overrides.recording_duration_sec;
    }
    if (overrides.recording_root.has_value()) {
        recording_root = *overrides.recording_root;
    }
    if (overrides.recording_keyboard_toggle.has_value()) {
        recording_keyboard_toggle = *overrides.recording_keyboard_toggle;
    }
    if (overrides.depth_mode.has_value()) {
        depth_mode = *overrides.depth_mode;
    }
    if (overrides.camera_resolution.has_value()) {
        camera_resolution = *overrides.camera_resolution;
    }
    if (overrides.serial_number.has_value()) {
        serial_number = overrides.serial_number;
    }
}

} // namespace zedapp
