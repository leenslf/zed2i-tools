cmake_minimum_required(VERSION 3.16)

project(zed_2i_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ZED REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(_zed_expected_cuda "")
if (DEFINED ZED_CUDA_VERSION)
    set(_zed_expected_cuda "${ZED_CUDA_VERSION}")
endif()

if (_zed_expected_cuda)
    find_package(CUDA ${_zed_expected_cuda} EXACT REQUIRED)
    if (DEFINED CUDA_VERSION AND NOT CUDA_VERSION VERSION_EQUAL _zed_expected_cuda)
        message(FATAL_ERROR "ZED SDK expects CUDA ${_zed_expected_cuda} but found CUDA ${CUDA_VERSION}.")
    endif()
else()
    find_package(CUDA REQUIRED)
    if (NOT DEFINED CUDA_VERSION)
        message(FATAL_ERROR "CUDA was found but its version could not be determined.")
    endif()
endif()

add_library(zed_app_core
    src/camera/CameraManager.cpp
    src/camera/Config.cpp
    src/camera/DataRecorder.cpp
    src/camera/DataRetriever.cpp
    src/camera/Logger.cpp
    src/camera/RealZedCamera.cpp
    src/camera/Validator.cpp
)

target_include_directories(zed_app_core
    PUBLIC
        include
        ${ZED_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(zed_app_core
    PUBLIC
        ${ZED_LIBRARIES}
        ${CUDA_LIBRARIES}
)

add_executable(zed_app src/camera/main.cpp)

target_link_libraries(zed_app PRIVATE zed_app_core)

add_library(zed_traversability_core
    src/traversability/config.cpp
    src/traversability/offline_pipeline.cpp
    src/traversability/polarize.cpp
    src/traversability/tilt_compensate.cpp
    src/traversability/traversability.cpp
    src/traversability/voxel_filter.cpp
)

target_include_directories(zed_traversability_core
    PUBLIC
        include
)

target_link_libraries(zed_traversability_core
    PUBLIC
        Eigen3::Eigen
        yaml-cpp
)

add_executable(zed_traversability_pipeline src/traversability/main.cpp)

target_link_libraries(zed_traversability_pipeline
    PRIVATE
        zed_traversability_core
        ${ZED_LIBRARIES}
        ${CUDA_LIBRARIES}
)

target_include_directories(zed_traversability_pipeline
    PRIVATE
        ${ZED_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

enable_testing()
include(FetchContent)

option(USE_SYSTEM_GTEST "Use system-installed GoogleTest" OFF)

if (USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
else()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(zed_app_tests
    tests/test_config.cpp
    tests/test_data_recorder.cpp
    tests/test_validator.cpp
    tests/test_data_retriever.cpp
    tests/test_integration_pipeline.cpp
)

target_link_libraries(zed_app_tests
    PRIVATE
        zed_app_core
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(zed_app_tests)
