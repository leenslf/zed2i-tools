#include "traversability/traversability.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace traversability {
namespace {

constexpr float kPi = 3.14159265358979323846f;

int reflect_index(int idx, int size) {
    if (size <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= size) {
        if (idx < 0) {
            idx = -idx - 1;
        } else {
            idx = 2 * size - idx - 1;
        }
    }
    return idx;
}

Eigen::VectorXf arange_with_step(float start, float stop, float step) {
    if (step <= 0.0f) {
        return Eigen::VectorXf(0);
    }
    std::vector<float> values;
    float v = start;
    while (v < stop) {
        values.push_back(v);
        v += step;
    }
    Eigen::VectorXf out(static_cast<Eigen::Index>(values.size()));
    for (Eigen::Index i = 0; i < out.size(); ++i) {
        out(i) = values[static_cast<std::size_t>(i)];
    }
    return out;
}

int find_bin(float value, const Eigen::VectorXf& edges) {
    const Eigen::Index num_bins = edges.size() - 1;
    if (num_bins <= 0) {
        return -1;
    }
    if (value < edges(0) || value > edges(edges.size() - 1)) {
        return -1;
    }
    if (value == edges(edges.size() - 1)) {
        return static_cast<int>(num_bins - 1);
    }
    for (Eigen::Index i = 0; i < num_bins; ++i) {
        if (edges(i) <= value && value < edges(i + 1)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::pair<Eigen::MatrixXf, Eigen::MatrixXf> gradient(
    const Eigen::MatrixXf& terrain,
    float spacing_r,
    float spacing_theta) {
    const Eigen::Index rows = terrain.rows();
    const Eigen::Index cols = terrain.cols();

    Eigen::MatrixXf grad_r(rows, cols);
    Eigen::MatrixXf grad_theta(rows, cols);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            float dr = 0.0f;
            if (rows > 1) {
                if (i == 0) {
                    dr = (terrain(i + 1, j) - terrain(i, j)) / spacing_r;
                } else if (i == rows - 1) {
                    dr = (terrain(i, j) - terrain(i - 1, j)) / spacing_r;
                } else {
                    dr = (terrain(i + 1, j) - terrain(i - 1, j)) / (2.0f * spacing_r);
                }
            }
            grad_r(i, j) = dr;

            float dtheta = 0.0f;
            if (cols > 1) {
                if (j == 0) {
                    dtheta = (terrain(i, j + 1) - terrain(i, j)) / spacing_theta;
                } else if (j == cols - 1) {
                    dtheta = (terrain(i, j) - terrain(i, j - 1)) / spacing_theta;
                } else {
                    dtheta = (terrain(i, j + 1) - terrain(i, j - 1)) / (2.0f * spacing_theta);
                }
            }
            grad_theta(i, j) = dtheta;
        }
    }

    return {grad_r, grad_theta};
}

Eigen::MatrixXf std_filter_3x3_reflect(const Eigen::MatrixXf& terrain) {
    const Eigen::Index rows = terrain.rows();
    const Eigen::Index cols = terrain.cols();
    Eigen::MatrixXf out(rows, cols);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int count = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ri = reflect_index(static_cast<int>(i) + di, static_cast<int>(rows));
                    const int cj = reflect_index(static_cast<int>(j) + dj, static_cast<int>(cols));
                    const float v = terrain(ri, cj);
                    sum += v;
                    sum_sq += v * v;
                    ++count;
                }
            }
            const float mean = sum / static_cast<float>(count);
            const float var = std::max(0.0f, sum_sq / static_cast<float>(count) - mean * mean);
            out(i, j) = std::sqrt(var);
        }
    }
    return out;
}

Eigen::MatrixXf max_filter_5x5_reflect(const Eigen::MatrixXf& terrain) {
    const Eigen::Index rows = terrain.rows();
    const Eigen::Index cols = terrain.cols();
    Eigen::MatrixXf out(rows, cols);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            float v_max = -std::numeric_limits<float>::infinity();
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    const int ri = reflect_index(static_cast<int>(i) + di, static_cast<int>(rows));
                    const int cj = reflect_index(static_cast<int>(j) + dj, static_cast<int>(cols));
                    v_max = std::max(v_max, terrain(ri, cj));
                }
            }
            out(i, j) = v_max;
        }
    }

    return out;
}

Eigen::MatrixXf min_filter_5x5_reflect(const Eigen::MatrixXf& terrain) {
    const Eigen::Index rows = terrain.rows();
    const Eigen::Index cols = terrain.cols();
    Eigen::MatrixXf out(rows, cols);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            float v_min = std::numeric_limits<float>::infinity();
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    const int ri = reflect_index(static_cast<int>(i) + di, static_cast<int>(rows));
                    const int cj = reflect_index(static_cast<int>(j) + dj, static_cast<int>(cols));
                    v_min = std::min(v_min, terrain(ri, cj));
                }
            }
            out(i, j) = v_min;
        }
    }

    return out;
}

Eigen::MatrixXf _estimate_danger_value(
    const Eigen::MatrixXf& terrain,
    float polar_grid_size_r,
    float polar_grid_size_theta,
    float scrit_deg,
    float rcrit_m,
    float hcrit_m) {
    const float inf = std::numeric_limits<float>::infinity();
    const float scrit = scrit_deg * kPi / 180.0f;

    auto grads = gradient(terrain, polar_grid_size_r, polar_grid_size_theta);
    Eigen::MatrixXf slope(terrain.rows(), terrain.cols());
    for (Eigen::Index i = 0; i < terrain.rows(); ++i) {
        for (Eigen::Index j = 0; j < terrain.cols(); ++j) {
            const float dzdx = grads.first(i, j);
            const float dzdy = grads.second(i, j);
            float s = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy));
            if (s > scrit) {
                s = inf;
            }
            slope(i, j) = s;
        }
    }

    Eigen::MatrixXf roughness = std_filter_3x3_reflect(terrain);
    for (Eigen::Index i = 0; i < roughness.rows(); ++i) {
        for (Eigen::Index j = 0; j < roughness.cols(); ++j) {
            if (roughness(i, j) > rcrit_m) {
                roughness(i, j) = inf;
            }
        }
    }

    const Eigen::MatrixXf max_z = max_filter_5x5_reflect(terrain);
    const Eigen::MatrixXf min_z = min_filter_5x5_reflect(terrain);
    Eigen::MatrixXf step_height(terrain.rows(), terrain.cols());
    for (Eigen::Index i = 0; i < terrain.rows(); ++i) {
        for (Eigen::Index j = 0; j < terrain.cols(); ++j) {
            const float dmax = std::abs(terrain(i, j) - max_z(i, j));
            const float dmin = std::abs(terrain(i, j) - min_z(i, j));
            step_height(i, j) = std::max(dmax, dmin);
        }
    }

    const int n_crit = 24;
    Eigen::MatrixXi st_mask = Eigen::MatrixXi::Zero(terrain.rows(), terrain.cols());
    for (Eigen::Index i = 0; i < terrain.rows(); ++i) {
        for (Eigen::Index j = 0; j < terrain.cols(); ++j) {
            const float center = terrain(i, j);
            int count = 0;
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    if (di == 0 && dj == 0) {
                        continue;
                    }
                    const int ri = reflect_index(static_cast<int>(i) + di, static_cast<int>(terrain.rows()));
                    const int cj = reflect_index(static_cast<int>(j) + dj, static_cast<int>(terrain.cols()));
                    if (std::abs(center - terrain(ri, cj)) > hcrit_m) {
                        ++count;
                    }
                }
            }
            st_mask(i, j) = count;
        }
    }

    for (Eigen::Index i = 0; i < step_height.rows(); ++i) {
        for (Eigen::Index j = 0; j < step_height.cols(); ++j) {
            const float scaled = step_height(i, j) * static_cast<float>(st_mask(i, j)) /
                                 static_cast<float>(n_crit);
            step_height(i, j) = std::min(step_height(i, j), scaled);
            if (step_height(i, j) > hcrit_m) {
                step_height(i, j) = inf;
            }
        }
    }

    Eigen::MatrixXf danger_value(terrain.rows(), terrain.cols());
    for (Eigen::Index i = 0; i < terrain.rows(); ++i) {
        for (Eigen::Index j = 0; j < terrain.cols(); ++j) {
            danger_value(i, j) =
                0.3f * slope(i, j) / scrit +
                0.3f * roughness(i, j) / rcrit_m +
                0.4f * step_height(i, j) / hcrit_m;
        }
    }

    return danger_value;
}

} // namespace

Traversability::Traversability(const TraversabilityConfig& config) : config_(config) {}

TraversabilityResult Traversability::process(const Eigen::MatrixXf& points) const {
    if (points.cols() < 3) {
        throw std::invalid_argument(
            "`points` must be 2D with at least 3 columns, got cols=" +
            std::to_string(points.cols()));
    }

    const Eigen::MatrixXf polar_points = points.leftCols(3);
    if (polar_points.size() == 0) {
        return TraversabilityResult{
            Eigen::MatrixXf(0, 0),
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>(0, 0),
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>(0, 0),
            Eigen::VectorXf(0),
            Eigen::VectorXf(0)};
    }

    const float danger_threshold = config_.danger_threshold;
    const float scrit_deg = config_.scrit_deg;
    const float rcrit_m = config_.rcrit_m;
    const float hcrit_m = config_.hcrit_m;
    const float polar_grid_size_r = config_.polar_grid_size_r_m;
    const float polar_grid_size_theta_deg = config_.polar_grid_size_theta_deg;
    const float polar_grid_size_theta = polar_grid_size_theta_deg * kPi / 180.0f;

    if (polar_grid_size_r <= 0.0f || polar_grid_size_theta <= 0.0f) {
        throw std::invalid_argument("Polar grid sizes must be positive.");
    }
    if (rcrit_m <= 0.0f || hcrit_m <= 0.0f) {
        throw std::invalid_argument("`rcrit_m` and `hcrit_m` must be positive.");
    }

    const float r_min = polar_points.col(0).minCoeff();
    const float r_max = polar_points.col(0).maxCoeff();
    const float theta_min = polar_points.col(1).minCoeff();
    const float theta_max = polar_points.col(1).maxCoeff();

    Eigen::VectorXf r_edges = arange_with_step(r_min, r_max + polar_grid_size_r, polar_grid_size_r);
    Eigen::VectorXf theta_edges = arange_with_step(
        theta_min, theta_max + polar_grid_size_theta, polar_grid_size_theta);

    if (r_edges.size() < 2 || theta_edges.size() < 2) {
        return TraversabilityResult{
            Eigen::MatrixXf(0, 0),
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>(0, 0),
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>(0, 0),
            r_edges,
            theta_edges};
    }

    const Eigen::Index r_bins = r_edges.size() - 1;
    const Eigen::Index theta_bins = theta_edges.size() - 1;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Eigen::MatrixXf height_map = Eigen::MatrixXf::Constant(r_bins, theta_bins, nan);

    for (Eigen::Index i = 0; i < polar_points.rows(); ++i) {
        const float r = polar_points(i, 0);
        const float theta = polar_points(i, 1);
        const float z = polar_points(i, 2);

        const int r_bin = find_bin(r, r_edges);
        const int theta_bin = find_bin(theta, theta_edges);
        if (r_bin < 0 || theta_bin < 0) {
            continue;
        }

        float& cell = height_map(r_bin, theta_bin);
        if (std::isnan(cell) || z > cell) {
            cell = z;
        }
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask(r_bins, theta_bins);
    Eigen::MatrixXf terrain = height_map;
    for (Eigen::Index i = 0; i < r_bins; ++i) {
        for (Eigen::Index j = 0; j < theta_bins; ++j) {
            const bool valid = !std::isnan(height_map(i, j));
            valid_mask(i, j) = valid;
            if (!valid) {
                terrain(i, j) = 0.0f;
            }
        }
    }

    Eigen::MatrixXf danger_grid = _estimate_danger_value(
        terrain,
        polar_grid_size_r,
        polar_grid_size_theta,
        scrit_deg,
        rcrit_m,
        hcrit_m);

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> nontraversable(r_bins, theta_bins);
    for (Eigen::Index i = 0; i < r_bins; ++i) {
        for (Eigen::Index j = 0; j < theta_bins; ++j) {
            if (!valid_mask(i, j)) {
                danger_grid(i, j) = nan;
            }
            nontraversable(i, j) = valid_mask(i, j) && (danger_grid(i, j) > danger_threshold);
        }
    }

    return TraversabilityResult{
        danger_grid,
        valid_mask,
        nontraversable,
        r_edges,
        theta_edges};
}

} // namespace traversability
