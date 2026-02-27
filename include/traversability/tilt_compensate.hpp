#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "traversability/config.hpp"

namespace traversability {

// TODO: define PointCloud type placeholder if a named alias is introduced later ?

// Points is (N, C) where C >= 3; first three columns are XYZ.
// Quaternion is [qx, qy, qz, qw].
// Returns a copy of points with XYZ rotated by the pitch/roll-only component.
// Remaining columns (if any) are passed through unchanged.
// Throws std::invalid_argument on malformed input or zero quaternion.
class TiltCompensate {
public:
    explicit TiltCompensate(const TiltCompensateConfig& config);
    Eigen::MatrixXf process(const Eigen::MatrixXf& points, const Eigen::Vector4f& quaternion) const;

private:
    TiltCompensateConfig config_;
};

} // namespace traversability
