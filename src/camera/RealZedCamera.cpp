#include "camera/RealZedCamera.hpp"

namespace zedapp {

RealZedCamera::~RealZedCamera() {
    close();
}

sl::ERROR_CODE RealZedCamera::open(const sl::InitParameters& params) {
    return camera_.open(params);
}

void RealZedCamera::close() {
    camera_.close();
}

sl::ERROR_CODE RealZedCamera::grab(const sl::RuntimeParameters& params) {
    return camera_.grab(params);
}

sl::ERROR_CODE RealZedCamera::retrieveImage(sl::Mat& image, sl::VIEW view) {
    return camera_.retrieveImage(image, view);
}

sl::ERROR_CODE RealZedCamera::retrieveMeasure(sl::Mat& measure, sl::MEASURE measure_type) {
    return camera_.retrieveMeasure(measure, measure_type);
}

sl::ERROR_CODE RealZedCamera::getSensorsData(sl::SensorsData& data, sl::TIME_REFERENCE reference) {
    return camera_.getSensorsData(data, reference);
}

sl::Timestamp RealZedCamera::getTimestamp(sl::TIME_REFERENCE reference) {
    return camera_.getTimestamp(reference);
}

sl::ERROR_CODE RealZedCamera::enableRecording(const sl::RecordingParameters& params) {
    return camera_.enableRecording(params);
}

void RealZedCamera::disableRecording() {
    camera_.disableRecording();
}

sl::ERROR_CODE RealZedCamera::enablePositionalTracking(const sl::PositionalTrackingParameters& params) {
    return camera_.enablePositionalTracking(params);
}

void RealZedCamera::disablePositionalTracking() {
    camera_.disablePositionalTracking();
}

sl::POSITIONAL_TRACKING_STATE RealZedCamera::getPosition(sl::Pose& pose, sl::REFERENCE_FRAME reference) {
    return camera_.getPosition(pose, reference);
}

} // namespace zedapp
