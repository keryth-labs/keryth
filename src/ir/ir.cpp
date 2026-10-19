#include "keryth/ir.hpp"
#include <sstream>

namespace keryth {

std::string dump_function(const IRFunction& fn) {
    std::ostringstream ss;
    ss << "fn @" << fn.name << "(";
    for (size_t i = 0; i < fn.parameters.size(); ++i) {
        ValueId pid = fn.parameters[i];
        const auto* val = fn.get_value(pid);
        ss << "%" << pid << ": " << (val && val->type ? val->type->to_string() : "unknown");
        if (i + 1 < fn.parameters.size()) {
            ss << ", ";
        }
    }
    ss << ") -> " << (fn.return_type ? fn.return_type->to_string() : "void") << " {\n";

    for (const auto& inst : fn.instructions) {
        ss << "  ";
        if (inst.has_result()) {
            ss << "%" << inst.result_id << ": "
               << (inst.type ? inst.type->to_string() : "void") << " = ";
        }

        ss << to_string(inst.opcode);

        if (!inst.operands.empty()) {
            ss << " ";
            for (size_t i = 0; i < inst.operands.size(); ++i) {
                ss << "%" << inst.operands[i];
                if (i + 1 < inst.operands.size()) {
                    ss << ", ";
                }
            }
        }

        if (std::holds_alternative<int64_t>(inst.literal)) {
            ss << " " << std::get<int64_t>(inst.literal);
        } else if (std::holds_alternative<double>(inst.literal)) {
            ss << " " << std::get<double>(inst.literal);
        } else if (std::holds_alternative<bool>(inst.literal)) {
            ss << " " << (std::get<bool>(inst.literal) ? "true" : "false");
        }

        ss << "\n";
    }

    ss << "}\n";
    return ss.str();
}

std::string dump_ir(const IRModule& module) {
    std::ostringstream ss;
    for (const auto& fn : module.functions) {
        ss << dump_function(fn) << "\n";
    }
    return ss.str();
}

} // namespace keryth
