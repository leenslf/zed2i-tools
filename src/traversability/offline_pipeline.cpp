#include "traversability/offline_pipeline.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>

namespace traversability {
namespace {

std::filesystem::path frame_path(const std::string& output_dir, int frame_index) {
    std::ostringstream name;
    name << "frame_" << std::setfill('0') << std::setw(5) << frame_index << ".bin";
    return std::filesystem::path(output_dir) / name.str();
}

void write_matrix_bin(const std::filesystem::path& path, const Eigen::MatrixXf& matrix) {
    const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
    const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major = matrix;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(
        reinterpret_cast<const char*>(row_major.data()),
        static_cast<std::streamsize>(sizeof(float) * row_major.size()));
}

void write_mask_bin(std::ofstream& out,
                    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& mask) {
    const std::int32_t rows = static_cast<std::int32_t>(mask.rows());
    const std::int32_t cols = static_cast<std::int32_t>(mask.cols());
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    for (Eigen::Index i = 0; i < mask.rows(); ++i) {
        for (Eigen::Index j = 0; j < mask.cols(); ++j) {
            const std::uint8_t v = mask(i, j) ? 1u : 0u;
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }
}

void write_vector_bin(std::ofstream& out, const Eigen::VectorXf& vec) {
    const std::int32_t size = static_cast<std::int32_t>(vec.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(
        reinterpret_cast<const char*>(vec.data()),
        static_cast<std::streamsize>(sizeof(float) * vec.size()));
}

void write_traversability_bin(const std::filesystem::path& path,
                              const TraversabilityResult& result) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    const std::int32_t danger_rows = static_cast<std::int32_t>(result.danger_grid.rows());
    const std::int32_t danger_cols = static_cast<std::int32_t>(result.danger_grid.cols());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> danger_row_major =
        result.danger_grid;

    out.write(reinterpret_cast<const char*>(&danger_rows), sizeof(danger_rows));
    out.write(reinterpret_cast<const char*>(&danger_cols), sizeof(danger_cols));
    out.write(
        reinterpret_cast<const char*>(danger_row_major.data()),
        static_cast<std::streamsize>(sizeof(float) * danger_row_major.size()));

    write_mask_bin(out, result.valid_mask);
    write_mask_bin(out, result.nontraversable);
    write_vector_bin(out, result.r_edges);
    write_vector_bin(out, result.theta_edges);
}

} // namespace

OfflinePipeline::OfflinePipeline(const PipelineConfig& config)
    : config_(config),
      tilt_compensate_(config.tilt_compensate),
      voxel_filter_(config.voxel_filter),
      polarize_(config.polarize),
      traversability_(config.traversability) {}

void OfflinePipeline::process_frame(const Eigen::MatrixXf& points,
                                    const Eigen::Vector4f& quaternion,
                                    int frame_index) {
    const Eigen::MatrixXf detilted = tilt_compensate_.process(points, quaternion);
    if (config_.tilt_compensate.write_output) {
        std::filesystem::create_directories(config_.tilt_compensate.output_dir);
        write_matrix_bin(frame_path(config_.tilt_compensate.output_dir, frame_index), detilted);
    }

    const Eigen::MatrixXf filtered = voxel_filter_.process(detilted);
    if (config_.voxel_filter.write_output) {
        std::filesystem::create_directories(config_.voxel_filter.output_dir);
        write_matrix_bin(frame_path(config_.voxel_filter.output_dir, frame_index), filtered);
    }

    const Eigen::MatrixXf polarized = polarize_.process(filtered);
    if (config_.polarize.write_output) {
        std::filesystem::create_directories(config_.polarize.output_dir);
        write_matrix_bin(frame_path(config_.polarize.output_dir, frame_index), polarized);
    }

    const TraversabilityResult traversability_result = traversability_.process(polarized);
    if (config_.traversability.write_output) {
        std::filesystem::create_directories(config_.traversability.output_dir);
        write_traversability_bin(
            frame_path(config_.traversability.output_dir, frame_index),
            traversability_result);
    }
}

} // namespace traversability
