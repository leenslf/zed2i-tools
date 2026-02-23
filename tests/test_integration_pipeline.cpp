#include <cstdlib>

#include <gtest/gtest.h>

#include "camera/CameraManager.hpp"
#include "camera/Config.hpp"
#include "camera/DataRetriever.hpp"

namespace zedapp {

TEST(IntegrationPipelineTest, RetrievesDataWithLiveCamera) {
    // End-to-end grab of frames and IMU from a live camera.
    if (std::getenv("ZED_TEST_LIVE") == nullptr) {
        GTEST_SKIP() << "ZED_TEST_LIVE not set";
    }

    Config config;
    config.enable_frames = true;
    config.enable_imu = true;
    config.enable_odometry = false;
    config.enable_point_cloud = false;

    CameraManager manager;
    auto camera = manager.openCamera(config);
    if (!camera) {
        GTEST_SKIP() << "Camera not available";
    }

    DataRetriever retriever(*camera, config);

    DataSnapshot snapshot;
    const auto status = retriever.retrieve(snapshot);

    EXPECT_EQ(status, sl::ERROR_CODE::SUCCESS);
    EXPECT_TRUE(snapshot.frames.has_value());
    EXPECT_TRUE(snapshot.imu.has_value());

    camera->close();
}

} // namespace zedapp
