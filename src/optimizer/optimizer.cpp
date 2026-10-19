#include "keryth/optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace keryth {

// -----------------------------------------------------------------------------
// Constant Folding
// -----------------------------------------------------------------------------

bool ConstantFoldingPass::run_on_function(IRFunction& fn) {
    bool changed = false;
    std::unordered_map<ValueId, LiteralValue> const_map;

    for (auto& inst : fn.instructions) {
        if (inst.opcode == Opcode::ConstantInt ||
            inst.opcode == Opcode::ConstantFloat ||
            inst.opcode == Opcode::ConstantBool) {
            const_map[inst.result_id] = inst.literal;
            continue;
        }

        if (inst.opcode == Opcode::Relu && inst.operands.size() == 1) {
            auto it = const_map.find(inst.operands[0]);
            if (it != const_map.end() && std::holds_alternative<double>(it->second)) {
                double val = std::max(0.0, std::get<double>(it->second));
                inst.opcode = Opcode::ConstantFloat;
                inst.literal = val;
                inst.operands.clear();
                const_map[inst.result_id] = val;
                changed = true;
                continue;
            }
        }

        if (inst.operands.size() == 2) {
            auto it1 = const_map.find(inst.operands[0]);
            auto it2 = const_map.find(inst.operands[1]);
            if (it1 != const_map.end() && it2 != const_map.end()) {
                const auto& v1 = it1->second;
                const auto& v2 = it2->second;

                if (std::holds_alternative<int64_t>(v1) && std::holds_alternative<int64_t>(v2)) {
                    int64_t i1 = std::get<int64_t>(v1);
                    int64_t i2 = std::get<int64_t>(v2);
                    int64_t res = 0;
                    bool fold = true;
                    switch (inst.opcode) {
                        case Opcode::Add: res = i1 + i2; break;
                        case Opcode::Sub: res = i1 - i2; break;
                        case Opcode::Mul: res = i1 * i2; break;
                        case Opcode::Div:
                            if (i2 != 0) res = i1 / i2;
                            else fold = false;
                            break;
                        default: fold = false; break;
                    }
                    if (fold) {
                        inst.opcode = Opcode::ConstantInt;
                        inst.literal = res;
                        inst.operands.clear();
                        const_map[inst.result_id] = res;
                        changed = true;
                    }
                } else if (std::holds_alternative<double>(v1) && std::holds_alternative<double>(v2)) {
                    double d1 = std::get<double>(v1);
                    double d2 = std::get<double>(v2);
                    double res = 0.0;
                    bool fold = true;
                    switch (inst.opcode) {
                        case Opcode::Add: res = d1 + d2; break;
                        case Opcode::Sub: res = d1 - d2; break;
                        case Opcode::Mul: res = d1 * d2; break;
                        case Opcode::Div:
                            if (d2 != 0.0) res = d1 / d2;
                            else fold = false;
                            break;
                        default: fold = false; break;
                    }
                    if (fold) {
                        inst.opcode = Opcode::ConstantFloat;
                        inst.literal = res;
                        inst.operands.clear();
                        const_map[inst.result_id] = res;
                        changed = true;
                    }
                }
            }
        }
    }

    return changed;
}

// -----------------------------------------------------------------------------
// Common Subexpression Elimination (CSE)
// -----------------------------------------------------------------------------

namespace {
struct ExprKey {
    Opcode opcode;
    std::vector<ValueId> operands;
    LiteralValue literal;

    bool operator==(const ExprKey& other) const {
        return opcode == other.opcode && operands == other.operands && literal == other.literal;
    }
};

struct ExprKeyHash {
    size_t operator()(const ExprKey& k) const {
        size_t h = static_cast<size_t>(k.opcode);
        for (ValueId op : k.operands) {
            h ^= std::hash<uint32_t>{}(op) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
} // namespace

bool CommonSubexpressionEliminationPass::run_on_function(IRFunction& fn) {
    bool changed = false;
    std::unordered_map<ExprKey, ValueId, ExprKeyHash> expr_map;
    std::unordered_map<ValueId, ValueId> replace_map;

    for (auto& inst : fn.instructions) {
        // First substitute any already replaced operands
        for (auto& op : inst.operands) {
            auto it = replace_map.find(op);
            if (it != replace_map.end()) {
                op = it->second;
                changed = true;
            }
        }

        if (!inst.has_result() || inst.opcode == Opcode::Parameter) {
            continue;
        }

        ExprKey key{inst.opcode, inst.operands, inst.literal};
        auto it = expr_map.find(key);
        if (it != expr_map.end()) {
            replace_map[inst.result_id] = it->second;
            changed = true;
        } else {
            expr_map[key] = inst.result_id;
        }
    }

    // Rewrite any remaining uses
    if (!replace_map.empty()) {
        for (auto& inst : fn.instructions) {
            for (auto& op : inst.operands) {
                auto it = replace_map.find(op);
                if (it != replace_map.end()) {
                    op = it->second;
                }
            }
        }
    }

    return changed;
}

// -----------------------------------------------------------------------------
// Dead Code Elimination (DCE)
// -----------------------------------------------------------------------------

bool DeadCodeEliminationPass::run_on_function(IRFunction& fn) {
    std::unordered_set<ValueId> live;
    std::unordered_map<ValueId, const Instruction*> def_map;

    for (const auto& inst : fn.instructions) {
        if (inst.has_result()) {
            def_map[inst.result_id] = &inst;
        }
        if (inst.opcode == Opcode::Return) {
            for (ValueId op : inst.operands) {
                live.insert(op);
            }
        }
    }

    // Propagate liveness
    std::vector<ValueId> worklist(live.begin(), live.end());
    while (!worklist.empty()) {
        ValueId curr = worklist.back();
        worklist.pop_back();

        auto it = def_map.find(curr);
        if (it != def_map.end()) {
            for (ValueId op : it->second->operands) {
                if (live.insert(op).second) {
                    worklist.push_back(op);
                }
            }
        }
    }

    // Filter dead instructions
    size_t orig_size = fn.instructions.size();
    std::vector<Instruction> new_insts;
    new_insts.reserve(orig_size);

    for (auto& inst : fn.instructions) {
        if (inst.opcode == Opcode::Return || inst.opcode == Opcode::Parameter) {
            new_insts.push_back(std::move(inst));
        } else if (live.find(inst.result_id) != live.end()) {
            new_insts.push_back(std::move(inst));
        }
    }

    bool changed = (new_insts.size() != orig_size);
    fn.instructions = std::move(new_insts);
    return changed;
}

// -----------------------------------------------------------------------------
// Operator Epilogue Fusion Pass
// -----------------------------------------------------------------------------

bool OperatorFusionPass::run_on_function(IRFunction& fn) {
    bool changed = false;

    // Count uses of each value
    std::unordered_map<ValueId, int> use_counts;
    for (const auto& inst : fn.instructions) {
        for (ValueId op : inst.operands) {
            use_counts[op]++;
        }
    }

    // Map result_id -> pointer to instruction
    std::unordered_map<ValueId, Instruction*> def_map;
    for (auto& inst : fn.instructions) {
        if (inst.has_result()) {
            def_map[inst.result_id] = &inst;
        }
    }

    // Look for:
    // %z = matmul %x, %w
    // %y = add %z, %b (or %b, %z)
    // %out = relu %y
    for (auto& inst : fn.instructions) {
        if (inst.opcode == Opcode::Relu && inst.operands.size() == 1) {
            ValueId y_id = inst.operands[0];
            auto it_y = def_map.find(y_id);
            if (it_y == def_map.end()) continue;

            Instruction* y_inst = it_y->second;
            if (y_inst->opcode != Opcode::Add || y_inst->operands.size() != 2) continue;
            if (use_counts[y_id] != 1) continue; // Must be uniquely consumed by ReLU

            ValueId z_id = INVALID_VALUE_ID;
            ValueId b_id = INVALID_VALUE_ID;

            auto it_op0 = def_map.find(y_inst->operands[0]);
            auto it_op1 = def_map.find(y_inst->operands[1]);

            if (it_op0 != def_map.end() && it_op0->second->opcode == Opcode::Matmul) {
                z_id = y_inst->operands[0];
                b_id = y_inst->operands[1];
            } else if (it_op1 != def_map.end() && it_op1->second->opcode == Opcode::Matmul) {
                z_id = y_inst->operands[1];
                b_id = y_inst->operands[0];
            } else {
                continue;
            }

            Instruction* z_inst = def_map[z_id];
            if (use_counts[z_id] != 1) continue; // Must be uniquely consumed by Add

            ValueId x_id = z_inst->operands[0];
            ValueId w_id = z_inst->operands[1];

            // Fuse into: %out = fused_matmul_bias_relu %x, %w, %b
            inst.opcode = Opcode::FusedMatmulBiasRelu;
            inst.operands = {x_id, w_id, b_id};
            changed = true;
        }
    }

    return changed;
}

} // namespace keryth
