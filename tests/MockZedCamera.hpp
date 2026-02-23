#pragma once

#include <sl/Camera.hpp>

#include "IZedCamera.hpp"

namespace zedapp {

class MockZedCamera : public IZedCamera {
public:
    sl::ERROR_CODE open(const sl::InitParameters&) override {
        opened = true;
        return open_result;
    }

    void close() override { opened = false; }

    sl::ERROR_CODE grab(const sl::RuntimeParameters&) override { return grab_result; }

    sl::ERROR_CODE retrieveImage(sl::Mat&, sl::VIEW) override { return image_result; }

    sl::ERROR_CODE retrieveMeasure(sl::Mat&, sl::MEASURE) override { return measure_result; }

    sl::ERROR_CODE getSensorsData(sl::SensorsData& data, sl::TIME_REFERENCE) override {
        data = sensors_data;
        return sensors_result;
    }

    sl::Timestamp getTimestamp(sl::TIME_REFERENCE) override { return timestamp; }

    sl::ERROR_CODE enableRecording(const sl::RecordingParameters& params) override {
        recording_enabled = true;
        last_recording_params = params;
        return recording_result;
    }

    void disableRecording() override { recording_enabled = false; }

    sl::ERROR_CODE enablePositionalTracking(const sl::PositionalTrackingParameters&) override {
        tracking_enabled = true;
        return tracking_result;
    }

    void disablePositionalTracking() override { tracking_enabled = false; }

    sl::POSITIONAL_TRACKING_STATE getPosition(sl::Pose& pose, sl::REFERENCE_FRAME) override {
        pose = pose_data;
        return tracking_state;
    }

    bool opened = false;
    bool tracking_enabled = false;
    bool recording_enabled = false;
    sl::RecordingParameters last_recording_params{};

    sl::ERROR_CODE open_result = sl::ERROR_CODE::SUCCESS;
    sl::ERROR_CODE grab_result = sl::ERROR_CODE::SUCCESS;
    sl::ERROR_CODE image_result = sl::ERROR_CODE::SUCCESS;
    sl::ERROR_CODE measure_result = sl::ERROR_CODE::SUCCESS;
    sl::ERROR_CODE sensors_result = sl::ERROR_CODE::SUCCESS;
    sl::ERROR_CODE tracking_result = sl::ERROR_CODE::SUCCESS;
    sl::ERROR_CODE recording_result = sl::ERROR_CODE::SUCCESS;

    sl::POSITIONAL_TRACKING_STATE tracking_state = sl::POSITIONAL_TRACKING_STATE::OK;
    sl::SensorsData sensors_data{};
    sl::Pose pose_data{};
    sl::Timestamp timestamp{};
};

} // namespace zedapp
