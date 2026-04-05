#pragma once

#include <Eigen/Core>

#include "traversability/config.hpp"

namespace traversability {

struct TraversabilityResult {
    Eigen::MatrixXf trav_grid;   // NaN = unknown, 0 = traversable, 1 = non-traversable
    Eigen::MatrixXf height_map;  // max-height per bin; unobserved bins filled with -0.3
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
