#include "camera/Logger.hpp"

#include <iostream>
#include <mutex>

namespace zedapp {

LogLevel Logger::min_level_ = LogLevel::Info;

namespace {

const char* levelLabel(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
    }
    return "INFO";
}

} // namespace

void Logger::setMinLevel(LogLevel level) {
    min_level_ = level;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (static_cast<int>(level) < static_cast<int>(min_level_)) {
        return;
    }

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << "[" << levelLabel(level) << "] " << message << std::endl;
}

} // namespace zedapp
