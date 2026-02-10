# ZED 2i C++ Application

Minimal, modular ZED 2i application using the official ZED SDK C++ API. The code is organized for easy extension and testing.

## Structure
- `include/` public headers
- `src/` implementation
- `tests/` unit + integration tests
- `config/` example configuration

## Prerequisites
- ZED 2i camera connected via USB 3.x
- NVIDIA GPU supported by the ZED SDK
- ZED SDK installed (headers + libraries)
- CUDA toolkit compatible with your ZED SDK version
- C++17 compiler (GCC/Clang)
- CMake 3.16+

## Easy installation (Ubuntu 22)
```bash
# Download installer
wget https://download.stereolabs.com/zedsdk/4.2/cu130/ubuntu22

# Make executable
chmod +x ubuntu22

# Install
./ubuntu22

# Follow prompts (accept license, choose components)
```

Verify installation:
```bash
# Check version
/usr/local/zed/tools/ZED_Explorer

# Add to bashrc if needed
echo 'export LD_LIBRARY_PATH=/usr/local/zed/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Run
```bash
./build/zed_app --config config/example.conf --iterations=200
```

### Recording output
When recording is enabled, the app writes to `recordings/YYYY-MM-DD_HH-MM-SS/` with:
- `images/` left/right frame images
- `imu.csv`
- `odometry.csv`
- `pointclouds/` (`.ply` per frame or a single `pointcloud.svo` when SVO recording is enabled)

### Command-line flags
- `--config <path>` load a config file
- `--enable-frames` / `--disable-frames`
- `--enable-imu` / `--disable-imu`
- `--enable-odometry` / `--disable-odometry`
- `--enable-point-cloud` / `--disable-point-cloud`
- `--record` / `--no-record` start or disable recording
- `--record-toggle` enable keyboard toggling (`r` to toggle, `q` to quit)
- `--record-duration=<seconds>` stop after N seconds
- `--record-frames=<N>` stop after N recorded frames
- `--record-stride=<N>` record every Nth frame
- `--record-image-format=png|jpg`
- `--record-pointcloud-format=ply|svo`
- `--record-root=<path>` output root folder (default `recordings`)
- `--depth-mode=PERFORMANCE|QUALITY|ULTRA|NEURAL`
- `--serial=<serial>`
- `--iterations=<N>` run N loops, 0 for infinite (default)
- `--sleep-ms=<N>` sleep between loops (default 5 ms)

## Tests
```bash
ctest --test-dir build
```

### Integration tests
Set `ZED_TEST_LIVE=1` to enable the live camera integration test:
```bash
ZED_TEST_LIVE=1 ctest --test-dir build -R Integration
```

## Analysis (Python)
The recording analysis script lives in `analysis/scripts/analyze_recording.py` and writes results to `analysis/results/<recording_name>/`.

### Setup (venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib
```

### Run analysis
```bash
python analysis/scripts/analyze_recording.py recordings/<recording_name>/
```

## Selecting a Coordinate System
The ZED uses a 3D Cartesian coordinate system (X, Y, Z) to express positions and orientations, and it can be configured as right-handed or left-handed. 

By default, the ZED uses the Image Coordinate System: right-handed with +Y down, +X right, and +Z pointing away from the camera. 

![ZED right-handed image coordinates](resources/zed_right_handed.jpg)

You can select a different coordinate system via `sl::InitParameters`:
- `IMAGE` - Right handed, y-down (default)
- `LEFT_HANDED_Y_UP` - Left handed, y-up (Unity 3D)
- `RIGHT_HANDED_Y_UP` - Right handed, y-up (OpenGL)
- `LEFT_HANDED_Z_UP` - Left handed, z-up (Unreal Engine)
- `RIGHT_HANDED_Z_UP` - Right handed, z-up (CADs, e.g., 3DS Max)
- `RIGHT_HANDED_Z_UP_X_FORWARD` - Right handed, z-up, x-forward (ROS - REP 103) 

## Notes
- `--serial` is parsed but not applied yet. Add the SDK call to select a specific camera serial after verifying the current API.
- If `find_package(ZED)` fails, set `ZED_SDK_ROOT_DIR` or adjust the CMake logic to match your SDK installation.
