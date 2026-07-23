#pragma once
// Optional structured logging. Default is silent (no callback).

#include <functional>
#include <string>

namespace arbo::doclayout {

enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
};

using LogCallback = std::function<void(LogLevel level, const std::string& message)>;

void setLogCallback(LogCallback callback);
void setMinLogLevel(LogLevel level);
LogLevel minLogLevel();
void log(LogLevel level, const std::string& message);
LogCallback makeStderrLogger();

} // namespace arbo::doclayout
