#pragma once

#include "keryth/ir.hpp"
#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace keryth {

struct ValueLifetime {
    ValueId id{INVALID_VALUE_ID};
    size_t first_def{0};
    size_t last_use{0};
    size_t size_bytes{0};
    size_t alignment{64}; // 64-byte alignment for cache line & SIMD
    std::string name_hint{};

    [[nodiscard]] bool overlaps_with(const ValueLifetime& other) const noexcept {
        return !(last_use < other.first_def || other.last_use < first_def);
    }
};

struct MemoryPlan {
    size_t total_arena_bytes{0};
    size_t unoptimized_total_bytes{0};
    std::unordered_map<ValueId, size_t> offsets;
    std::unordered_map<ValueId, ValueLifetime> lifetimes;

    [[nodiscard]] double memory_saving_percent() const noexcept {
        if (unoptimized_total_bytes == 0) return 0.0;
        return (1.0 - (static_cast<double>(total_arena_bytes) / static_cast<double>(unoptimized_total_bytes))) * 100.0;
    }

    [[nodiscard]] std::string dump() const;
};

class MemoryPlanner {
public:
    explicit MemoryPlanner(size_t alignment = 64) noexcept
        : default_alignment_(alignment) {}

    [[nodiscard]] std::unordered_map<ValueId, ValueLifetime> analyze_liveness(const IRFunction& fn) const;
    [[nodiscard]] MemoryPlan plan(const IRFunction& fn) const;

private:
    [[nodiscard]] size_t get_size_bytes(const Type* type) const noexcept;
    static size_t align_up(size_t offset, size_t alignment) noexcept {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    size_t default_alignment_{64};
};

} // namespace keryth
