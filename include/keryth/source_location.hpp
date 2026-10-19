#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace keryth {

struct SourceLocation {
    std::string_view filename{};
    uint32_t line{1};
    uint32_t column{1};
    uint32_t offset{0};

    constexpr bool operator==(const SourceLocation& other) const noexcept {
        return line == other.line && column == other.column && offset == other.offset && filename == other.filename;
    }
};

struct SourceSpan {
    SourceLocation start{};
    SourceLocation end{};

    [[nodiscard]] constexpr bool is_empty() const noexcept {
        return start.offset == end.offset;
    }

    [[nodiscard]] std::string to_string() const {
        std::string result;
        if (!start.filename.empty()) {
            result += std::string(start.filename) + ":";
        }
        result += std::to_string(start.line) + ":" + std::to_string(start.column);
        if (start.line != end.line || start.column != end.column) {
            result += "-" + std::to_string(end.line) + ":" + std::to_string(end.column);
        }
        return result;
    }
};

} // namespace keryth
