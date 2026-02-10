#pragma once
// Camera abstraction for real and mock ZED cameras.

#include <sl/Camera.hpp>

namespace zedapp {

class IZedCamera {
public:
    virtual ~IZedCamera() = default;

    virtual sl::ERROR_CODE open(const sl::InitParameters& params) = 0;
    virtual void close() = 0;

    virtual sl::ERROR_CODE grab(const sl::RuntimeParameters& params) = 0;
    virtual sl::ERROR_CODE retrieveImage(sl::Mat& image, sl::VIEW view) = 0;
    virtual sl::ERROR_CODE retrieveMeasure(sl::Mat& measure, sl::MEASURE measure_type) = 0;

    virtual sl::ERROR_CODE getSensorsData(sl::SensorsData& data, sl::TIME_REFERENCE reference) = 0;
    virtual sl::ERROR_CODE enablePositionalTracking(const sl::PositionalTrackingParameters& params) = 0;
    virtual void disablePositionalTracking() = 0;
    virtual sl::POSITIONAL_TRACKING_STATE getPosition(sl::Pose& pose, sl::REFERENCE_FRAME reference) = 0;
};

} // namespace zedapp
