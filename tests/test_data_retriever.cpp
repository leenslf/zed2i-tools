#include <gtest/gtest.h>

#include "Config.hpp"
#include "DataRetriever.hpp"
#include "MockZedCamera.hpp"

namespace zedapp {

TEST(DataRetrieverTest, RetrievesEnabledData) {
    // Ensures enabled data paths populate the snapshot.
    Config config;
    config.enable_frames = true;
    config.enable_imu = true;
    config.enable_odometry = true;
    config.enable_point_cloud = false;

    MockZedCamera camera;
    DataRetriever retriever(camera, config);

    DataSnapshot snapshot;
    const auto status = retriever.retrieve(snapshot);

    EXPECT_EQ(status, sl::ERROR_CODE::SUCCESS);
    EXPECT_TRUE(snapshot.frames.has_value());
    EXPECT_TRUE(snapshot.imu.has_value());
    EXPECT_TRUE(snapshot.odometry.has_value());
    EXPECT_FALSE(snapshot.point_cloud.has_value());
}

} // namespace zedapp
