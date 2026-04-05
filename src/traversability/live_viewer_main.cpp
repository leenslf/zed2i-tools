/// Live traversability viewer — C++ implementation.
///
/// Opens a ZED camera (live or SVO) and runs the full traversability pipeline
/// in a background thread.  The main thread reads the latest result and
/// renders it as a Cartesian bird's-eye BGR image via OpenCV, updated in real
/// time.
///
/// The display is oriented with the robot at the bottom-centre, forward (X)
/// pointing up, and lateral (Y) pointing right.  Each coloured cell represents
/// a single Cartesian grid cell:
///   - Green  → traversable (danger below threshold, ray-cast observed free)
///   - Red    → non-traversable (danger above threshold)
///   - Gray   → unknown (not yet observed or beyond sensor range)
///
/// The optional --nt-height-overlay flag blends a magma colormap over
/// non-traversable cells, keyed to the height of the highest point in each
/// polar bin.  This helps distinguish obstacles by height (kerbs, walls, etc.).
///
/// Usage (SVO):  ./live_trav_viewer --svo <path>
/// Usage (live): ./live_trav_viewer
/// Press 'q' or ESC to quit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sl/Camera.hpp>

#include "traversability/config.hpp"
#include "traversability/pipeline_runner.hpp"

namespace {

// ---------------------------------------------------------------------------
// Display constants — match the Python live_trav_viewer for visual parity.
// ---------------------------------------------------------------------------

/// Cartesian display region: X (forward) in metres.
constexpr float kXMin = 0.0f;
constexpr float kXMax = 2.0f;
/// Cartesian display region: Y (lateral) in metres.
constexpr float kYMin = -0.75f;
constexpr float kYMax =  0.75f;

/// Scale factor applied to the raw grid image before display.
/// Each grid cell is rendered as kDisplayScale × kDisplayScale pixels.
constexpr int kDisplayScale = 10;

/// Alpha used when blending the magma height overlay onto non-traversable cells.
constexpr float kAlpha = 0.65f;

// BGR colours for each traversability state (matches Python viewer exactly).
const cv::Vec3b kBgrTraversable   {80,  175, 76};   // #4CAF50 green
const cv::Vec3b kBgrNonTraversable{54,  67,  244};  // #F44336 red
const cv::Vec3b kBgrUnknown       {136, 136, 136};  // #888888 gray

const char* const kWindowName = "Traversability";

// ---------------------------------------------------------------------------
// Command-line arguments
// ---------------------------------------------------------------------------

struct Args {
    std::string svo_path;    ///< Path to SVO/SVO2 file, or empty for live camera.
    std::string config_path; ///< YAML config path; empty → "config/pipeline_config.yaml".
    int   frame_skip    = -1;    ///< -1 means use the value from the config file.
    int   max_processed_frames = -1; ///< Stop after N processed frames; -1 means no limit.
    bool  nt_height_overlay = false;
    bool  no_render     = false; ///< Disable OpenCV rendering and window/event handling.
    float grid_res_m    = 0.05f; ///< Cartesian cell side length in metres.
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --svo <path>         SVO/SVO2 file (omit to use a live camera)\n"
        << "  --config <path>      Pipeline config YAML (default: config/pipeline_config.yaml)\n"
        << "  --frame-skip <N>     Process every Nth frame (overrides config value)\n"
        << "  --max-processed-frames <N>  Stop after N processed frames\n"
        << "  --no-render          Disable window creation and image rendering\n"
        << "  --nt-height-overlay  Colour non-traversable cells by height (magma)\n"
        << "  --grid-res-m <f>     Cartesian raster resolution in metres (default: 0.05)\n"
        << "  --help, -h           Show this message\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--help" || a == "-h")) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--svo" && i + 1 < argc) {
            args.svo_path = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (a == "--frame-skip" && i + 1 < argc) {
            args.frame_skip = std::stoi(argv[++i]);
        } else if (a == "--max-processed-frames" && i + 1 < argc) {
            args.max_processed_frames = std::stoi(argv[++i]);
        } else if (a == "--no-render") {
            args.no_render = true;
        } else if (a == "--nt-height-overlay") {
            args.nt_height_overlay = true;
        } else if (a == "--grid-res-m" && i + 1 < argc) {
            args.grid_res_m = std::stof(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return args;
}

std::string format_seconds(double seconds) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << seconds << " s";
    return oss.str();
}

// ---------------------------------------------------------------------------
// ZED SDK enum parsers (mirror those in src/traversability/main.cpp)
// ---------------------------------------------------------------------------

sl::UNIT parse_coordinate_units(const std::string& v) {
    if (v == "METER")      return sl::UNIT::METER;
    if (v == "CENTIMETER") return sl::UNIT::CENTIMETER;
    if (v == "MILLIMETER") return sl::UNIT::MILLIMETER;
    if (v == "INCH")       return sl::UNIT::INCH;
    if (v == "FOOT")       return sl::UNIT::FOOT;
    throw std::runtime_error("Unknown coordinate_units: " + v);
}

sl::COORDINATE_SYSTEM parse_coordinate_system(const std::string& v) {
    if (v == "RIGHT_HANDED_Z_UP")       return sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP;
    if (v == "RIGHT_HANDED_Z_UP_X_FWD") return sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    if (v == "RIGHT_HANDED_Y_UP")       return sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    if (v == "LEFT_HANDED_Y_UP")        return sl::COORDINATE_SYSTEM::LEFT_HANDED_Y_UP;
    if (v == "LEFT_HANDED_Z_UP")        return sl::COORDINATE_SYSTEM::LEFT_HANDED_Z_UP;
    throw std::runtime_error("Unknown coordinate_system: " + v);
}

sl::DEPTH_MODE parse_depth_mode(const std::string& v) {
    if (v == "PERFORMANCE") return sl::DEPTH_MODE::PERFORMANCE;
    if (v == "QUALITY")     return sl::DEPTH_MODE::QUALITY;
    if (v == "ULTRA")       return sl::DEPTH_MODE::ULTRA;
    if (v == "NEURAL")      return sl::DEPTH_MODE::NEURAL;
    throw std::runtime_error("Unknown depth_mode: " + v);
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Find the bin index for `val` in a sorted edge array.
/// Equivalent to np.searchsorted(edges, val, side="right") - 1.
/// Returns -1 when val falls outside [edges[0], edges[last]).
int bin_index(float val, const Eigen::VectorXf& edges) {
    const int n_bins = static_cast<int>(edges.size()) - 1;
    if (n_bins <= 0) return -1;
    const float* b = edges.data();
    const float* e = b + edges.size();
    const int idx = static_cast<int>(std::upper_bound(b, e, val) - b) - 1;
    return (idx >= 0 && idx < n_bins) ? idx : -1;
}

/// Return the p-th percentile (p in [0, 100]) of an already-sorted vector.
/// Uses linear interpolation between the two nearest elements.
float sorted_percentile(const std::vector<float>& sorted, float p) {
    if (sorted.size() == 1) return sorted[0];
    const float pos  = p / 100.0f * static_cast<float>(sorted.size() - 1);
    const int   lo   = static_cast<int>(pos);
    const int   hi   = std::min(lo + 1, static_cast<int>(sorted.size()) - 1);
    const float frac = pos - static_cast<float>(lo);
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

/// Draw a coloured-swatch legend in the top-left corner of `img`.
void draw_legend(cv::Mat& img, bool show_nt_height_overlay) {
    struct Item { const char* label; cv::Vec3b color; };
    std::vector<Item> items = {
        {"traversable",     kBgrTraversable},
        {"non-traversable", kBgrNonTraversable},
        {"unknown",         kBgrUnknown},
    };
    if (show_nt_height_overlay) {
        items.push_back({"nt height (magma)", {180, 50, 180}});
    }

    int x0 = 8, y0 = 8;
    constexpr int kSwatch = 12;
    constexpr int kPad    = 4;

    for (const auto& item : items) {
        const cv::Scalar color(item.color[0], item.color[1], item.color[2]);
        cv::rectangle(img,
            cv::Point(x0, y0), cv::Point(x0 + kSwatch, y0 + kSwatch),
            color, cv::FILLED);
        cv::putText(img, item.label,
            cv::Point(x0 + kSwatch + kPad, y0 + kSwatch - 1),
            cv::FONT_HERSHEY_SIMPLEX, 0.38,
            cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        y0 += kSwatch + kPad;
    }
}

/// Convert a PipelineFrame into a display-ready BGR image.
///
/// Algorithm overview:
///   1. Iterate over Cartesian cells [x_min,x_max] × [y_min,y_max].
///   2. For each cell centre (x, y) compute polar coordinates (r, θ).
///   3. Look up the traversability value in the polar bin via binary search.
///   4. Colour the cell green / red / gray.
///   5. Optionally blend a magma height overlay on non-traversable cells.
///   6. Transpose and vertically flip to orient the image with forward-up.
///   7. Scale up by kDisplayScale and draw legend + frame counter.
///
/// @param frame               Pipeline result and frame index.
/// @param grid_res_m          Cartesian cell resolution in metres.
/// @param show_nt_height_overlay  Whether to add the magma height overlay.
/// @returns BGR image suitable for cv::imshow, or empty Mat if result is empty.
cv::Mat render_frame(
    const traversability::PipelineFrame& frame,
    float grid_res_m,
    bool  show_nt_height_overlay)
{
    const auto& result = frame.result;

    // Guard against empty results (e.g. too few points for the polar grid).
    if (result.trav_grid.size() == 0 ||
        result.r_edges.size() < 2 ||
        result.theta_edges.size() < 2) {
        return {};
    }

    const int nr = static_cast<int>(result.r_edges.size()) - 1;
    const int nt = static_cast<int>(result.theta_edges.size()) - 1;

    // Cartesian grid dimensions.
    // nx bins span the forward range [x_min, x_max].
    // ny bins span the lateral range [y_min, y_max].
    const int nx = static_cast<int>(std::ceil((kXMax - kXMin) / grid_res_m));
    const int ny = static_cast<int>(std::ceil((kYMax - kYMin) / grid_res_m));

    // Primary output image: (ny rows × nx cols), filled with the "unknown" gray.
    // Row axis = Y (lateral), column axis = X (forward).
    cv::Mat cart_bgr(ny, nx, CV_8UC3, cv::Scalar(kBgrUnknown[0], kBgrUnknown[1], kBgrUnknown[2]));

    // Track which Cartesian pixels are non-traversable (needed for the overlay).
    cv::Mat nt_cart(ny, nx, CV_8U, cv::Scalar(0));

    // ----- Pass 1: colorise the traversability grid -----
    for (int iy = 0; iy < ny; ++iy) {
        const float y = kYMin + (iy + 0.5f) * grid_res_m;
        for (int ix = 0; ix < nx; ++ix) {
            const float x = kXMin + (ix + 0.5f) * grid_res_m;

            // Convert Cartesian cell centre to polar coordinates.
            const float r     = std::sqrt(x * x + y * y);
            const float theta = std::atan2(y, x);

            const int ir = bin_index(r,     result.r_edges);
            const int it = bin_index(theta, result.theta_edges);
            if (ir < 0 || it < 0) continue;  // outside polar domain

            const float tval = result.trav_grid(ir, it);
            if (!std::isfinite(tval)) continue;  // NaN → unknown, leave gray

            if (std::abs(tval) < 1e-5f) {
                // tval ≈ 0.0 → traversable
                cart_bgr.at<cv::Vec3b>(iy, ix) = kBgrTraversable;
            } else {
                // tval ≈ 1.0 → non-traversable
                cart_bgr.at<cv::Vec3b>(iy, ix) = kBgrNonTraversable;
                nt_cart.at<uint8_t>(iy, ix)    = 255;
            }
        }
    }

    // ----- Pass 2 (optional): magma height overlay on non-traversable cells -----
    if (show_nt_height_overlay && result.height_map.size() > 0) {
        // Build the polar mask: only bins classified non-traversable.
        // We restrict the height rasterisation to these bins so that only
        // obstacle heights feed into the colormap normalisation.
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> nt_polar(nr, nt);
        for (int i = 0; i < nr; ++i) {
            for (int j = 0; j < nt; ++j) {
                const float tv = result.trav_grid(i, j);
                nt_polar(i, j) = std::isfinite(tv) && std::abs(tv - 1.0f) < 1e-5f;
            }
        }

        // Rasterise the height map into Cartesian space, restricted to
        // non-traversable polar bins.  Cells that map to traversable or unknown
        // bins are left as NaN.
        cv::Mat height_cart(ny, nx, CV_32F, std::numeric_limits<float>::quiet_NaN());
        for (int iy = 0; iy < ny; ++iy) {
            const float y = kYMin + (iy + 0.5f) * grid_res_m;
            for (int ix = 0; ix < nx; ++ix) {
                const float x = kXMin + (ix + 0.5f) * grid_res_m;
                const float r     = std::sqrt(x * x + y * y);
                const float theta = std::atan2(y, x);
                const int ir = bin_index(r,     result.r_edges);
                const int it = bin_index(theta, result.theta_edges);
                if (ir < 0 || it < 0 || !nt_polar(ir, it)) continue;
                height_cart.at<float>(iy, ix) = result.height_map(ir, it);
            }
        }

        // Collect valid heights (non-traversable Cartesian pixels with a finite
        // height value) so we can compute robust percentile bounds.
        std::vector<float> valid_heights;
        valid_heights.reserve(static_cast<std::size_t>(nx * ny) / 4);
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const float h = height_cart.at<float>(iy, ix);
                if (nt_cart.at<uint8_t>(iy, ix) && std::isfinite(h)) {
                    valid_heights.push_back(h);
                }
            }
        }

        if (!valid_heights.empty()) {
            std::sort(valid_heights.begin(), valid_heights.end());

            // Use 5th–95th percentile range to suppress extreme outliers.
            float h_lo = sorted_percentile(valid_heights, 5.0f);
            float h_hi = sorted_percentile(valid_heights, 95.0f);

            // Fallback to data min/max if the percentile range is degenerate.
            if (!std::isfinite(h_lo) || !std::isfinite(h_hi) || h_hi <= h_lo) {
                h_lo = valid_heights.front();
                h_hi = std::max(valid_heights.back(), h_lo + 1e-3f);
            }

            // Build a uint8 single-channel image for cv::applyColorMap.
            // Pixels that are not valid height cells stay at 0 (will be masked out).
            cv::Mat norm_gray(ny, nx, CV_8U, cv::Scalar(0));
            cv::Mat valid_h_mask(ny, nx, CV_8U, cv::Scalar(0));
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    const float h = height_cart.at<float>(iy, ix);
                    if (!nt_cart.at<uint8_t>(iy, ix) || !std::isfinite(h)) continue;
                    const float norm = std::clamp((h - h_lo) / (h_hi - h_lo), 0.0f, 1.0f);
                    norm_gray.at<uint8_t>(iy, ix) = static_cast<uint8_t>(norm * 255.0f);
                    valid_h_mask.at<uint8_t>(iy, ix) = 255;
                }
            }

            // cv::COLORMAP_MAGMA is available in OpenCV ≥ 3.4.1.
            cv::Mat magma_bgr;
            cv::applyColorMap(norm_gray, magma_bgr, cv::COLORMAP_MAGMA);

            // Alpha-blend the magma result onto non-traversable Cartesian pixels.
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    if (!valid_h_mask.at<uint8_t>(iy, ix)) continue;
                    cv::Vec3b& base       = cart_bgr.at<cv::Vec3b>(iy, ix);
                    const cv::Vec3b& over = magma_bgr.at<cv::Vec3b>(iy, ix);
                    for (int c = 0; c < 3; ++c) {
                        base[c] = cv::saturate_cast<uint8_t>(
                            (1.0f - kAlpha) * base[c] + kAlpha * over[c]);
                    }
                }
            }
        }
    }

    // ----- Orient the image for display -----
    // cart_bgr: row = Y (lateral, index 0 = y_min / left side)
    //           col = X (forward,  index 0 = x_min / near)
    //
    // After cv::transpose: row = X (forward, index 0 = x_min / near)
    //                      col = Y (lateral, index 0 = y_min / left)
    //
    // After cv::flip(flipCode=0, vertical): row 0 = x_max (far / top of image)
    //                                       last row = x_min (near / bottom)
    //
    // This matches the Python viewer:
    //   np.flipud(img.transpose(1, 0, 2))
    cv::Mat display;
    cv::transpose(cart_bgr, display);
    cv::flip(display, display, 0);

    // Scale up each grid cell to kDisplayScale × kDisplayScale pixels.
    cv::resize(display, display,
               cv::Size(display.cols * kDisplayScale, display.rows * kDisplayScale),
               0.0, 0.0, cv::INTER_NEAREST);

    draw_legend(display, show_nt_height_overlay);

    // Frame index in the bottom-left corner.
    cv::putText(
        display,
        "frame " + std::to_string(frame.frame_index),
        cv::Point(8, display.rows - 8),
        cv::FONT_HERSHEY_SIMPLEX, 0.45,
        cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    return display;
}

// ---------------------------------------------------------------------------
// Camera initialisation
// ---------------------------------------------------------------------------

/// Open an sl::Camera for SVO playback or live capture and enable tracking.
///
/// @param zed       Camera object to open in-place.
/// @param svo_path  Path to SVO/SVO2 file; empty string for live camera.
/// @param svo_cfg   Camera parameter settings from the pipeline config.
/// @throws std::runtime_error if the camera cannot be opened or tracking fails.
void open_camera(sl::Camera& zed,
                 const std::string& svo_path,
                 const traversability::SvoConfig& svo_cfg)
{
    sl::InitParameters init;
    init.coordinate_units  = parse_coordinate_units(svo_cfg.coordinate_units);
    init.coordinate_system = parse_coordinate_system(svo_cfg.coordinate_system);
    init.depth_mode        = parse_depth_mode(svo_cfg.depth_mode);

    if (!svo_path.empty()) {
        init.input.setFromSVOFile(svo_path.c_str());
    }
    // If svo_path is empty, the SDK opens the first available physical camera.

    const sl::ERROR_CODE open_err = zed.open(init);
    if (open_err != sl::ERROR_CODE::SUCCESS) {
        const std::string source = svo_path.empty() ? "live camera" : "SVO '" + svo_path + "'";
        throw std::runtime_error(
            "Failed to open " + source + ": " +
            std::string(sl::toString(open_err).c_str()));
    }

    const sl::ERROR_CODE track_err =
        zed.enablePositionalTracking(sl::PositionalTrackingParameters());
    if (track_err != sl::ERROR_CODE::SUCCESS) {
        zed.close();
        throw std::runtime_error(
            "Failed to enable positional tracking: " +
            std::string(sl::toString(track_err).c_str()));
    }
}

// ---------------------------------------------------------------------------
// Main render loop
// ---------------------------------------------------------------------------

void run(const Args& args) {
    const std::string config_path =
        args.config_path.empty() ? "config/pipeline_config.yaml" : args.config_path;

    traversability::PipelineConfig config =
        traversability::load_config(config_path);

    const int frame_skip = (args.frame_skip > 0) ? args.frame_skip : config.svo.frame_skip;
    if (frame_skip <= 0) {
        throw std::runtime_error("frame_skip must be >= 1");
    }
    if (args.max_processed_frames == 0 || args.max_processed_frames < -1) {
        throw std::runtime_error("max_processed_frames must be >= 1 or -1");
    }

    // Disable disk writes — we only care about in-memory results.
    config.tilt_compensate.write_output = false;
    config.voxel_filter.write_output    = false;
    config.polarize.write_output        = false;
    config.traversability.write_output  = false;

    sl::Camera zed;
    open_camera(zed, args.svo_path, config.svo);

    const std::string source_label =
        args.svo_path.empty() ? "live camera" : "SVO: " + args.svo_path;
    std::cout << "[live_trav_viewer] source=" << source_label
              << "  frame_skip=" << frame_skip
              << "  max_processed_frames=" << args.max_processed_frames
              << "  no_render=" << std::boolalpha << args.no_render
              << "  grid_res=" << args.grid_res_m << " m"
              << "  nt_height_overlay=" << args.nt_height_overlay
              << "\n";

    traversability::PipelineRunner runner(config);
    const auto wall_start = std::chrono::steady_clock::now();
    runner.start(zed, frame_skip, args.max_processed_frames);

    if (!args.no_render) {
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    }

    // ---- Render loop ----
    // The main thread only does: wait for key / window close, drain the drop
    // slot, render, and imshow.  All heavy computation is in the worker thread.
    while (!runner.done()) {
        if (!args.no_render) {
            // Check for quit events first so the UI stays responsive even when
            // the pipeline worker is slow.
            if (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) < 1) {
                std::cout << "[live_trav_viewer] Window closed, stopping.\n";
                break;
            }
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27 /* ESC */) {
                break;
            }
        }

        // Block up to 50 ms for the next result.  This timeout keeps the
        // event loop running (for key / window checks) even if the pipeline
        // worker is slower than 20 fps.
        traversability::PipelineFrame frame;
        if (!runner.next(frame, std::chrono::milliseconds(50))) {
            continue;
        }

        if (!args.no_render) {
            const cv::Mat display = render_frame(frame, args.grid_res_m, args.nt_height_overlay);
            if (!display.empty()) {
                cv::imshow(kWindowName, display);
            }
        }
    }

    runner.stop();
    const auto wall_end = std::chrono::steady_clock::now();
    const traversability::PipelineStats stats = runner.stats();
    zed.close();
    if (!args.no_render) {
        cv::destroyAllWindows();
    }

    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();

    std::cout << "[live_trav_viewer] stats"
              << "  wall=" << format_seconds(wall_seconds)
              << "  successful_grabs=" << stats.successful_grabs
              << "  processed=" << stats.processed_frames
              << "  grab_failures=" << stats.grab_failures
              << "  last_frame_index=" << stats.last_frame_index
              << "\n";

    if (!args.svo_path.empty() &&
        stats.first_timestamp_ns > 0 &&
        stats.last_timestamp_ns >= stats.first_timestamp_ns) {
        const double svo_seconds =
            static_cast<double>(stats.last_timestamp_ns - stats.first_timestamp_ns) / 1e9;
        std::cout << "[live_trav_viewer] svo_stats"
                  << "  svo_span=" << format_seconds(svo_seconds)
                  << "  status=" << (stats.reached_end_of_input ? "complete" : "stopped_early");
        if (svo_seconds > 0.0) {
            const double wall_over_svo = wall_seconds / svo_seconds;
            const double realtime_factor = svo_seconds / wall_seconds;
            std::cout << "  wall_over_svo=" << std::fixed << std::setprecision(3)
                      << wall_over_svo
                      << "  realtime_factor=" << realtime_factor << "x";
        }
        std::cout << "\n";
    }

    if (stats.processed_frames > 0) {
        const double processed = static_cast<double>(stats.processed_frames);
        std::cout << "[live_trav_viewer] stage_last_ms"
                  << "  retrieve=" << std::fixed << std::setprecision(3) << stats.last_retrieve_ms
                  << "  extract=" << stats.last_extract_ms
                  << "  tilt=" << stats.last_tilt_ms
                  << "  voxel=" << stats.last_voxel_ms
                  << "  polarize=" << stats.last_polarize_ms
                  << "  traversability=" << stats.last_traversability_ms
                  << "\n";
        std::cout << "[live_trav_viewer] stage_avg_ms"
                  << "  retrieve=" << std::fixed << std::setprecision(3)
                  << (stats.total_retrieve_ms / processed)
                  << "  extract=" << (stats.total_extract_ms / processed)
                  << "  tilt=" << (stats.total_tilt_ms / processed)
                  << "  voxel=" << (stats.total_voxel_ms / processed)
                  << "  polarize=" << (stats.total_polarize_ms / processed)
                  << "  traversability=" << (stats.total_traversability_ms / processed)
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        run(args);
    } catch (const std::exception& ex) {
        std::cerr << "[live_trav_viewer] Fatal error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
