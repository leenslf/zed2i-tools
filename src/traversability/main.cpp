#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sl/Camera.hpp>

#include "traversability/config.hpp"
#include "traversability/offline_pipeline.hpp"

namespace {

sl::UNIT parse_coordinate_units(const std::string& value) {
    if (value == "METER") {
        return sl::UNIT::METER;
    }
    if (value == "CENTIMETER") {
        return sl::UNIT::CENTIMETER;
    }
    if (value == "MILLIMETER") {
        return sl::UNIT::MILLIMETER;
    }
    if (value == "INCH") {
        return sl::UNIT::INCH;
    }
    if (value == "FOOT") {
        return sl::UNIT::FOOT;
    }
    throw std::runtime_error("Unknown coordinate_units: " + value);
}

sl::COORDINATE_SYSTEM parse_coordinate_system(const std::string& value) {
    if (value == "RIGHT_HANDED_Z_UP") {
        return sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP;
    }
    if (value == "RIGHT_HANDED_Z_UP_X_FWD") {
        return sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    }
    if (value == "RIGHT_HANDED_Y_UP") {
        return sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    }
    if (value == "LEFT_HANDED_Y_UP") {
        return sl::COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
    }
    if (value == "LEFT_HANDED_Z_UP") {
        return sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP;
    }
    throw std::runtime_error("Unknown coordinate_system: " + value);
}

sl::DEPTH_MODE parse_depth_mode(const std::string& value) {
    if (value == "PERFORMANCE") {
        return sl::DEPTH_MODE::PERFORMANCE;
    }
    if (value == "QUALITY") {
        return sl::DEPTH_MODE::QUALITY;
    }
    if (value == "ULTRA") {
        return sl::DEPTH_MODE::ULTRA;
    }
    if (value == "NEURAL") {
        return sl::DEPTH_MODE::NEURAL;
    }
    throw std::runtime_error("Unknown depth_mode: " + value);
}

Eigen::MatrixXf extract_finite_xyz(sl::Mat& point_cloud) {
    const int width = point_cloud.getWidth();
    const int height = point_cloud.getHeight();
    if (width <= 0 || height <= 0) {
        return Eigen::MatrixXf(0, 3);
    }

    std::vector<Eigen::Vector3f> finite_points;
    finite_points.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    sl::float4 point;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            point_cloud.getValue(x, y, &point);
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                finite_points.emplace_back(point.x, point.y, point.z);
            }
        }
    }

    Eigen::MatrixXf out(static_cast<Eigen::Index>(finite_points.size()), 3);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(finite_points.size()); ++i) {
        out.row(i) = finite_points[static_cast<std::size_t>(i)];
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <svo_path>\n";
        return 1;
    }

    const std::string svo_path = argv[1];
    const traversability::PipelineConfig config =
        traversability::load_config("config/pipeline_config.yaml");
    if (config.svo.frame_skip <= 0) {
        throw std::runtime_error("svo.frame_skip must be > 0");
    }

    traversability::OfflinePipeline pipeline(config);

    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.input.setFromSVOFile(svo_path.c_str());
    init_params.coordinate_units = parse_coordinate_units(config.svo.coordinate_units);
    init_params.coordinate_system = parse_coordinate_system(config.svo.coordinate_system);
    init_params.depth_mode = parse_depth_mode(config.svo.depth_mode);

    const sl::ERROR_CODE open_status = zed.open(init_params);
    if (open_status != sl::ERROR_CODE::SUCCESS) {
        throw std::runtime_error("Failed to open SVO file: " + svo_path);
    }

    const sl::ERROR_CODE tracking_status =
        zed.enablePositionalTracking(sl::PositionalTrackingParameters());
    if (tracking_status != sl::ERROR_CODE::SUCCESS) {
        zed.close();
        throw std::runtime_error("Failed to enable positional tracking.");
    }

    sl::Mat point_cloud;
    sl::Pose pose;
    int frame_index = 0;

    try {
        while (true) {
            const sl::ERROR_CODE grab_status = zed.grab();
            if (grab_status == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
                break;
            }
            if (grab_status != sl::ERROR_CODE::SUCCESS) {
                break;
            }

            const sl::ERROR_CODE measure_status =
                zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZ, sl::MEM::CPU); // why CPU?
            if (measure_status != sl::ERROR_CODE::SUCCESS) {
                break;
            }

            const sl::POSITIONAL_TRACKING_STATE pose_status =
                zed.getPosition(pose, sl::REFERENCE_FRAME::WORLD);
            if (pose_status == sl::POSITIONAL_TRACKING_STATE::OFF) {
                break;
            }

            const Eigen::MatrixXf points = extract_finite_xyz(point_cloud);
            if (frame_index % config.svo.frame_skip == 0) {
                const sl::Orientation orientation = pose.getOrientation();
                const Eigen::Vector4f quaternion(
                    orientation.ox, orientation.oy, orientation.oz, orientation.ow);
                pipeline.process_frame(points, quaternion, frame_index);
            }

            ++frame_index;
        }
    } catch (...) {
        zed.close();
        throw;
    }

    zed.close();
    return 0;
}
