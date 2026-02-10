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

### Command-line flags
- `--config <path>` load a config file
- `--enable-frames` / `--disable-frames`
- `--enable-imu` / `--disable-imu`
- `--enable-odometry` / `--disable-odometry`
- `--enable-point-cloud` / `--disable-point-cloud`
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

## Notes
- `--serial` is parsed but not applied yet. Add the SDK call to select a specific camera serial after verifying the current API.
- If `find_package(ZED)` fails, set `ZED_SDK_ROOT_DIR` or adjust the CMake logic to match your SDK installation.
