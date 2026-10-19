#pragma once

#include "keryth/ir.hpp"
#include "keryth/memory_planner.hpp"
#include "keryth/runtime.hpp"
#include <vector>

namespace keryth {

class ArenaExecutor {
public:
    ArenaExecutor() = default;

    RuntimeValue execute(const IRFunction& fn, const MemoryPlan& plan, const std::vector<RuntimeValue>& args);
};

} // namespace keryth
