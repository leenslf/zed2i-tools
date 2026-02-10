#pragma once
// RAII wrapper over sl::Camera.

#include "IZedCamera.hpp"

namespace zedapp {

class RealZedCamera : public IZedCamera {
public:
    RealZedCamera() = default;
    ~RealZedCamera() override;

    sl::ERROR_CODE open(const sl::InitParameters& params) override;
    void close() override;

    sl::ERROR_CODE grab(const sl::RuntimeParameters& params) override;
    sl::ERROR_CODE retrieveImage(sl::Mat& image, sl::VIEW view) override;
    sl::ERROR_CODE retrieveMeasure(sl::Mat& measure, sl::MEASURE measure_type) override;

    sl::ERROR_CODE getSensorsData(sl::SensorsData& data, sl::TIME_REFERENCE reference) override;
    sl::ERROR_CODE enablePositionalTracking(const sl::PositionalTrackingParameters& params) override;
    void disablePositionalTracking() override;
    sl::POSITIONAL_TRACKING_STATE getPosition(sl::Pose& pose, sl::REFERENCE_FRAME reference) override;

private:
    sl::Camera camera_;
};

} // namespace zedapp
