#include <limits>

#include <gtest/gtest.h>

#include "camera/Validator.hpp"

namespace zedapp {

TEST(ValidatorTest, ValidImuSample) {
    // Accepts finite IMU values.
    ImuSample sample;
    sample.linear_accel = {0.0f, 0.1f, -0.2f};
    sample.angular_vel = {0.01f, 0.02f, 0.03f};
    sample.orientation = {0.0f, 0.0f, 0.0f, 1.0f};

    EXPECT_TRUE(Validator::validateImu(sample));
}

TEST(ValidatorTest, InvalidImuSample) {
    // Rejects IMU data containing NaNs.
    ImuSample sample;
    sample.linear_accel = {0.0f, 0.1f, -0.2f};
    sample.angular_vel = {0.01f, 0.02f, 0.03f};
    sample.orientation = {0.0f, 0.0f, 0.0f, std::numeric_limits<float>::quiet_NaN()};

    EXPECT_FALSE(Validator::validateImu(sample));
}

TEST(ValidatorTest, PointCloudSamplesValidation) {
    // Accepts point clouds with at least one finite sample.
    std::vector<PointSample> samples;
    samples.push_back(PointSample{std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 1.0f});
    samples.push_back(PointSample{1.0f, 2.0f, 3.0f, 1.0f});

    EXPECT_TRUE(Validator::validatePointCloudSamples(samples));
}

TEST(ValidatorTest, PointCloudSamplesInvalid) {
    // Rejects point clouds with only invalid samples.
    std::vector<PointSample> samples;
    samples.push_back(PointSample{std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 1.0f});

    EXPECT_FALSE(Validator::validatePointCloudSamples(samples));
}

} // namespace zedapp
