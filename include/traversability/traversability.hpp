#pragma once

#include <Eigen/Core>

#include "traversability/config.hpp"

namespace traversability {

struct TraversabilityResult {
    Eigen::MatrixXf danger_grid;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> nontraversable;
    Eigen::VectorXf r_edges;
    Eigen::VectorXf theta_edges;
};

// Points is (N, 3) polar [r, theta, z].
// Throws std::invalid_argument on malformed input or invalid config values.
class Traversability {
public:
    explicit Traversability(const TraversabilityConfig& config);
    TraversabilityResult process(const Eigen::MatrixXf& points) const;

private:
    TraversabilityConfig config_;
};

} // namespace traversability
