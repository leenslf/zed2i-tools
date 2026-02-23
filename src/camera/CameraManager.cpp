#include "camera/CameraManager.hpp"

#include <memory>
#include <sstream>

#include <sl/Camera.hpp>

#include "camera/Logger.hpp"
#include "camera/RealZedCamera.hpp"

namespace zedapp {
namespace {

sl::DEPTH_MODE toSdkDepthMode(DepthMode mode) {
    switch (mode) {
        case DepthMode::Performance:
            return sl::DEPTH_MODE::PERFORMANCE;
        case DepthMode::Quality:
            return sl::DEPTH_MODE::QUALITY;
        case DepthMode::Ultra:
            return sl::DEPTH_MODE::ULTRA;
        case DepthMode::Neural:
            return sl::DEPTH_MODE::NEURAL;
    }
    return sl::DEPTH_MODE::QUALITY;
}

} // namespace

std::unique_ptr<IZedCamera> CameraManager::openCamera(const Config& config) {
    auto camera = std::make_unique<RealZedCamera>();

    sl::InitParameters init_params;
    init_params.depth_mode = toSdkDepthMode(config.depth_mode);
    init_params.coordinate_units = sl::UNIT::METER;

    if (config.serial_number.has_value()) {
        Logger::log(LogLevel::Warn,
                    "Serial number provided but not applied. "
                    "TODO: set InitParameters input serial after verifying SDK API.");
    }

    const auto open_status = camera->open(init_params);
    if (open_status != sl::ERROR_CODE::SUCCESS) {
        std::ostringstream oss;
        oss << "Failed to open ZED camera: error code " << static_cast<int>(open_status);
        Logger::log(LogLevel::Error, oss.str());
        return nullptr;
    }

    if (config.enable_odometry) {
        sl::PositionalTrackingParameters tracking_params;
        const auto tracking_status = camera->enablePositionalTracking(tracking_params);
        if (tracking_status != sl::ERROR_CODE::SUCCESS) {
            std::ostringstream oss;
            oss << "Failed to enable positional tracking: error code " << static_cast<int>(tracking_status);
            Logger::log(LogLevel::Error, oss.str());
            camera->close();
            return nullptr;
        }
    }

    Logger::log(LogLevel::Info, "ZED camera opened successfully.");
    return camera;
}

} // namespace zedapp
