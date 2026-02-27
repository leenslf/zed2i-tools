#include "traversability/tilt_compensate.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace traversability {
namespace {

float yaw_from_quaternion_xyzw(const Eigen::Quaternionf& q_xyzw) {
    const float x = q_xyzw.x();
    const float y = q_xyzw.y();
    const float z = q_xyzw.z();
    const float w = q_xyzw.w();

    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

} // namespace

TiltCompensate::TiltCompensate(const TiltCompensateConfig& config) : config_(config) {}

Eigen::MatrixXf TiltCompensate::process(const Eigen::MatrixXf& points,
                                        const Eigen::Vector4f& quaternion) const {
    if (points.cols() < 3) {
        throw std::invalid_argument(
            "`points` must be 2D with at least 3 columns, got cols=" +
            std::to_string(points.cols()));
    }

    const float norm = quaternion.norm();
    if (norm == 0.0f) {
        throw std::invalid_argument("`quaternion` must be non-zero.");
    }

    // Python input ordering is [x, y, z, w].
    Eigen::Quaternionf q_world_cam(
        quaternion[3] / norm,
        quaternion[0] / norm,
        quaternion[1] / norm,
        quaternion[2] / norm);

    // Mirror Python logic:
    // yaw = _yaw_from_quaternion_xyzw(q_world_cam)
    // q_pr = q_world_cam * Rz(yaw)^-1
    const float yaw = yaw_from_quaternion_xyzw(q_world_cam);
    const Eigen::Quaternionf q_yaw(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    const Eigen::Quaternionf q_pr = q_world_cam * q_yaw.inverse();

    Eigen::MatrixXf corrected(points);
    corrected.leftCols(3) = (q_pr.toRotationMatrix() * points.leftCols(3).transpose()).transpose();
    return corrected;
}

} // namespace traversability
