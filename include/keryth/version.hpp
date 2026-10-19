#pragma once

#include <string_view>

namespace keryth {

inline constexpr int VERSION_MAJOR = 0;
inline constexpr int VERSION_MINOR = 1;
inline constexpr int VERSION_PATCH = 0;

inline constexpr std::string_view VERSION_STRING = "0.1.0";
inline constexpr std::string_view TAGLINE = "Write once. Optimize for the machine.";

[[nodiscard]] constexpr std::string_view get_version() noexcept {
    return VERSION_STRING;
}

[[nodiscard]] constexpr std::string_view get_tagline() noexcept {
    return TAGLINE;
}

} // namespace keryth
