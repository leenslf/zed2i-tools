#include "traversability/traversability.hpp"

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

Eigen::MatrixXf _estimate_danger_value(
    const Eigen::MatrixXf& terrain,
    float r_min,
    float theta_min,
    float polar_grid_size_r,
    float polar_grid_size_theta,
    float scrit_deg,
    float rcrit_m,
    float hcrit_m) {
    const float inf = std::numeric_limits<float>::infinity();
    const float scrit = scrit_deg * kPi / 180.0f;
    const Eigen::Index nr = terrain.rows();
    const Eigen::Index nc = terrain.cols();

    // --- Slope with arc-length correction ---
    // r_centres are used to convert dzdy from z/rad to z/m: dzdy_metric = dzdy / r.
    Eigen::VectorXf r_centres(nr);
    for (Eigen::Index i = 0; i < nr; ++i) {
        r_centres(i) = r_min + (static_cast<float>(i) + 0.5f) * polar_grid_size_r;
    }

    auto grads = gradient(terrain, polar_grid_size_r, polar_grid_size_theta);
    Eigen::MatrixXf slope(nr, nc);
    for (Eigen::Index i = 0; i < nr; ++i) {
        for (Eigen::Index j = 0; j < nc; ++j) {
            const float dzdx = grads.first(i, j);
            // Arc-length correction: arc length = r*dtheta, so metric slope = dzdy / r.
            const float dzdy_metric = grads.second(i, j) / r_centres(i);
            float s = std::atan(std::sqrt(dzdx * dzdx + dzdy_metric * dzdy_metric));
            if (s > scrit) {
                s = inf;
            }
            slope(i, j) = s;
        }
    }

    // --- Roughness: std deviation over 3x3 neighborhood ---
    Eigen::MatrixXf roughness = std_filter_3x3_reflect(terrain);
    for (Eigen::Index i = 0; i < roughness.rows(); ++i) {
        for (Eigen::Index j = 0; j < roughness.cols(); ++j) {
            if (roughness(i, j) > rcrit_m) {
                roughness(i, j) = inf;
            }
        }
    }

    // --- Step height: Cartesian distances + pairwise slope check (5x5 window) ---
    // Build Cartesian cell centres so distances are in true metric space, treating
    // cells at different ranges fairly.
    Eigen::VectorXf t_centres(nc);
    for (Eigen::Index j = 0; j < nc; ++j) {
        t_centres(j) = theta_min + (static_cast<float>(j) + 0.5f) * polar_grid_size_theta;
    }
    Eigen::MatrixXf Xgrid(nr, nc);
    Eigen::MatrixXf Ygrid(nr, nc);
    for (Eigen::Index i = 0; i < nr; ++i) {
        for (Eigen::Index j = 0; j < nc; ++j) {
            Xgrid(i, j) = r_centres(i) * std::cos(t_centres(j));
            Ygrid(i, j) = r_centres(i) * std::sin(t_centres(j));
        }
    }

    constexpr int kHalf = 2;                                     // 5x5 window
    constexpr int kNCrit = (2 * kHalf + 1) * (2 * kHalf + 1) - 1;  // 24 neighbors

    Eigen::MatrixXi st_mask = Eigen::MatrixXi::Zero(nr, nc);
    Eigen::MatrixXf h_max = Eigen::MatrixXf::Zero(nr, nc);

    for (int di = -kHalf; di <= kHalf; ++di) {
        for (int dj = -kHalf; dj <= kHalf; ++dj) {
            if (di == 0 && dj == 0) {
                continue;
            }
            for (Eigen::Index i = 0; i < nr; ++i) {
                for (Eigen::Index j = 0; j < nc; ++j) {
                    const int ri = reflect_index(static_cast<int>(i) + di, static_cast<int>(nr));
                    const int cj = reflect_index(static_cast<int>(j) + dj, static_cast<int>(nc));

                    const float dz = std::abs(terrain(i, j) - terrain(ri, cj));
                    const float dx = Xgrid(i, j) - Xgrid(ri, cj);
                    const float dy = Ygrid(i, j) - Ygrid(ri, cj);
                    const float dxy = std::sqrt(dx * dx + dy * dy);

                    // Mirror Python: arctan2(dz, nan) when dxy==0 → nan → does not qualify.
                    if (dxy == 0.0f) {
                        continue;
                    }
                    const float pair_slope = std::atan2(dz, dxy);

                    if (dz > hcrit_m && pair_slope > scrit) {
                        st_mask(i, j) += 1;
                        if (dz > h_max(i, j)) {
                            h_max(i, j) = dz;
                        }
                    }
                }
            }
        }
    }

    Eigen::MatrixXf step_height(nr, nc);
    for (Eigen::Index i = 0; i < nr; ++i) {
        for (Eigen::Index j = 0; j < nc; ++j) {
            const float scaled = h_max(i, j) * static_cast<float>(st_mask(i, j)) /
                                 static_cast<float>(kNCrit);
            step_height(i, j) = std::min(h_max(i, j), scaled);
            if (step_height(i, j) > hcrit_m) {
                step_height(i, j) = inf;
            }
        }
    }

    Eigen::MatrixXf danger_value(nr, nc);
    for (Eigen::Index i = 0; i < nr; ++i) {
        for (Eigen::Index j = 0; j < nc; ++j) {
            danger_value(i, j) =
                0.3f * slope(i, j) / scrit +
                0.3f * roughness(i, j) / rcrit_m +
                0.4f * step_height(i, j) / hcrit_m;
        }
    }

    return danger_value;
}

// For each angular column, finds the closest non-traversable cell radially.
// All cells between the sensor origin and that obstacle are marked free space.
Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> _compute_ray_cast_mask(
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& nontraversable) {
    const Eigen::Index n_r = nontraversable.rows();
    const Eigen::Index n_c = nontraversable.cols();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> mask(n_r, n_c);
    mask.setConstant(false);

    for (Eigen::Index j = 0; j < n_c; ++j) {
        int closest = -1;
        for (Eigen::Index i = 0; i < n_r; ++i) {
            if (nontraversable(i, j)) {
                closest = static_cast<int>(i);
                break;
            }
        }
        for (int i = 0; i < closest; ++i) {
            mask(i, j) = true;
        }
    }

    return mask;
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
            Eigen::MatrixXf(0, 0),
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

    // Grid extent: use config bounds if provided, else derive from data.
    const float r_min = std::isnan(config_.r_min_m)
        ? polar_points.col(0).minCoeff()
        : config_.r_min_m;
    const float r_max = std::isnan(config_.r_max_m)
        ? polar_points.col(0).maxCoeff()
        : config_.r_max_m;
    const float theta_min = std::isnan(config_.theta_min_deg)
        ? polar_points.col(1).minCoeff()
        : config_.theta_min_deg * kPi / 180.0f;
    const float theta_max = std::isnan(config_.theta_max_deg)
        ? polar_points.col(1).maxCoeff()
        : config_.theta_max_deg * kPi / 180.0f;

    Eigen::VectorXf r_edges = arange_with_step(r_min, r_max + polar_grid_size_r, polar_grid_size_r);
    Eigen::VectorXf theta_edges = arange_with_step(
        theta_min, theta_max + polar_grid_size_theta, polar_grid_size_theta);

    if (r_edges.size() < 2 || theta_edges.size() < 2) {
        return TraversabilityResult{
            Eigen::MatrixXf(0, 0),
            Eigen::MatrixXf(0, 0),
            r_edges,
            theta_edges};
    }

    const Eigen::Index r_bins = r_edges.size() - 1;
    const Eigen::Index theta_bins = theta_edges.size() - 1;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // Build height map: max z per bin, NaN for unobserved bins.
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

    // valid_mask: true where height_map has data.
    // terrain: height_map with missing bins filled with -0.3 (slightly below camera plane).
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask(r_bins, theta_bins);
    Eigen::MatrixXf terrain = height_map;
    for (Eigen::Index i = 0; i < r_bins; ++i) {
        for (Eigen::Index j = 0; j < theta_bins; ++j) {
            const bool valid = !std::isnan(height_map(i, j));
            valid_mask(i, j) = valid;
            if (!valid) {
                terrain(i, j) = -0.3f;
            }
        }
    }

    Eigen::MatrixXf danger_grid = _estimate_danger_value(
        terrain,
        r_min,
        theta_min,
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

    const auto observed_mask = _compute_ray_cast_mask(nontraversable);

    // trav_grid: NaN = unknown, 0 = observed & traversable, 1 = non-traversable.
    Eigen::MatrixXf trav_grid = Eigen::MatrixXf::Constant(r_bins, theta_bins, nan);
    for (Eigen::Index i = 0; i < r_bins; ++i) {
        for (Eigen::Index j = 0; j < theta_bins; ++j) {
            if (observed_mask(i, j) && !nontraversable(i, j)) {
                trav_grid(i, j) = 0.0f;
            }
            if (valid_mask(i, j) && danger_grid(i, j) > danger_threshold) {
                trav_grid(i, j) = 1.0f;
            }
        }
    }

    return TraversabilityResult{trav_grid, terrain, r_edges, theta_edges};
}

} // namespace traversability
