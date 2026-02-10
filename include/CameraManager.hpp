#pragma once
// Camera initialization and configuration.

#include <memory>

#include "Config.hpp"
#include "IZedCamera.hpp"

namespace zedapp {

class CameraManager {
public:
    std::unique_ptr<IZedCamera> openCamera(const Config& config);
};

} // namespace zedapp
