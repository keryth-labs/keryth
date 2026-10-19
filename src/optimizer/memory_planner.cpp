#include "keryth/memory_planner.hpp"
#include <iomanip>
#include <sstream>

namespace keryth {

size_t MemoryPlanner::get_size_bytes(const Type* type) const noexcept {
    if (!type) return 0;
    if (type->is_tensor()) {
        return static_cast<size_t>(type->num_elements()) * sizeof(float);
    }
    if (type->element_kind() == ScalarKind::I32) {
        return sizeof(int32_t);
    }
    if (type->element_kind() == ScalarKind::F32) {
        return sizeof(float);
    }
    return sizeof(bool);
}

std::unordered_map<ValueId, ValueLifetime> MemoryPlanner::analyze_liveness(const IRFunction& fn) const {
    std::unordered_map<ValueId, ValueLifetime> lifetimes;

    // Initialize lifetimes for all defined values
    for (const auto& [vid, val] : fn.values) {
        lifetimes[vid] = ValueLifetime{
            .id = vid,
            .first_def = 0,
            .last_use = 0,
            .size_bytes = get_size_bytes(val.type.get()),
            .alignment = default_alignment_,
            .name_hint = val.name_hint
        };
    }

    // Traverse instructions to find exact definition points and last usages
    for (size_t i = 0; i < fn.instructions.size(); ++i) {
        const auto& inst = fn.instructions[i];

        if (inst.has_result() && inst.result_id != INVALID_VALUE_ID) {
            auto it = lifetimes.find(inst.result_id);
            if (it != lifetimes.end()) {
                it->second.first_def = i;
                it->second.last_use = i;
            }
        }

        for (ValueId op : inst.operands) {
            auto it = lifetimes.find(op);
            if (it != lifetimes.end()) {
                it->second.last_use = std::max(it->second.last_use, i);
            }
        }
    }

    return lifetimes;
}

MemoryPlan MemoryPlanner::plan(const IRFunction& fn) const {
    MemoryPlan plan;
    plan.lifetimes = analyze_liveness(fn);

    // Compute unoptimized footprint (sum of all buffers if never reused)
    size_t unopt_bytes = 0;
    std::vector<ValueLifetime> sorted_lifetimes;
    sorted_lifetimes.reserve(plan.lifetimes.size());

    for (const auto& [vid, lt] : plan.lifetimes) {
        unopt_bytes += align_up(lt.size_bytes, default_alignment_);
        sorted_lifetimes.push_back(lt);
    }
    plan.unoptimized_total_bytes = unopt_bytes;

    // Sort by first_def ascending, then by size_bytes descending
    std::sort(sorted_lifetimes.begin(), sorted_lifetimes.end(), [](const ValueLifetime& a, const ValueLifetime& b) {
        if (a.first_def != b.first_def) return a.first_def < b.first_def;
        return a.size_bytes > b.size_bytes;
    });

    struct PlacedBlock {
        ValueId id;
        size_t offset;
        size_t size;
        size_t first_def;
        size_t last_use;
    };
    std::vector<PlacedBlock> placed;

    size_t max_arena = 0;

    for (const auto& lt : sorted_lifetimes) {
        if (lt.size_bytes == 0) {
            plan.offsets[lt.id] = 0;
            continue;
        }

        // First-fit search for the lowest aligned offset
        size_t candidate_offset = 0;
        while (true) {
            candidate_offset = align_up(candidate_offset, lt.alignment);
            size_t cand_end = candidate_offset + lt.size_bytes;
            bool conflict = false;

            for (const auto& p : placed) {
                // Check if lifetimes overlap in time
                bool time_overlap = !(lt.last_use < p.first_def || p.last_use < lt.first_def);
                if (time_overlap) {
                    // Check if byte ranges overlap in memory
                    size_t p_end = p.offset + p.size;
                    bool space_overlap = !(cand_end <= p.offset || p_end <= candidate_offset);
                    if (space_overlap) {
                        conflict = true;
                        candidate_offset = p_end; // Skip past conflicting block
                        break;
                    }
                }
            }

            if (!conflict) {
                // Found valid placement
                break;
            }
        }

        placed.push_back(PlacedBlock{
            .id = lt.id,
            .offset = candidate_offset,
            .size = lt.size_bytes,
            .first_def = lt.first_def,
            .last_use = lt.last_use
        });

        plan.offsets[lt.id] = candidate_offset;
        max_arena = std::max(max_arena, candidate_offset + lt.size_bytes);
    }

    plan.total_arena_bytes = align_up(max_arena, default_alignment_);
    return plan;
}

std::string MemoryPlan::dump() const {
    std::ostringstream ss;
    ss << "======================================================================\n"
       << "                   KERYTH STATIC MEMORY ARENA PLAN                    \n"
       << "======================================================================\n";
    ss << std::left << std::setw(8) << "Value"
       << std::setw(16) << "Name"
       << std::setw(12) << "Size"
       << std::setw(14) << "Lifetime"
       << std::setw(14) << "Arena Offset"
       << "\n";
    ss << "----------------------------------------------------------------------\n";

    for (const auto& [vid, lt] : lifetimes) {
        auto it = offsets.find(vid);
        size_t off = (it != offsets.end()) ? it->second : 0;
        std::string lifetime_str = "[" + std::to_string(lt.first_def) + ".." + std::to_string(lt.last_use) + "]";

        ss << std::left << std::setw(8) << ("%" + std::to_string(vid))
           << std::setw(16) << (lt.name_hint.empty() ? "-" : lt.name_hint)
           << std::setw(12) << (std::to_string(lt.size_bytes) + " B")
           << std::setw(14) << lifetime_str
           << std::setw(14) << ("0x" + ([off]() {
               std::ostringstream h; h << std::hex << off; return h.str();
           })() + " (" + std::to_string(off) + ")")
           << "\n";
    }

    ss << "----------------------------------------------------------------------\n";
    ss << "Unoptimized (without reuse): " << unoptimized_total_bytes / 1024 << " KB (" << unoptimized_total_bytes << " B)\n";
    ss << "Planned Arena (with reuse):   " << total_arena_bytes / 1024 << " KB (" << total_arena_bytes << " B)\n";
    ss << std::fixed << std::setprecision(1);
    ss << "Memory Footprint Reduction:  " << memory_saving_percent() << "%\n";
    ss << "======================================================================\n";
    return ss.str();
}

} // namespace keryth
