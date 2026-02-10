#pragma once
// Minimal logging utility.

#include <string>

namespace zedapp {

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error
};

class Logger {
public:
    static void log(LogLevel level, const std::string& message);
    static void setMinLevel(LogLevel level);

private:
    static LogLevel min_level_;
};

} // namespace zedapp
