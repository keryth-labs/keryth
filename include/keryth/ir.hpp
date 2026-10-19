#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>
#include "keryth/source_location.hpp"
#include "keryth/types.hpp"

namespace keryth {

using ValueId = uint32_t;
inline constexpr ValueId INVALID_VALUE_ID = 0xFFFFFFFF;

enum class Opcode {
    Parameter,
    ConstantInt,
    ConstantFloat,
    ConstantBool,
    ConstantTensor,
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Relu,
    Sum,
    Transpose,
    Reshape,
    LoadTensor,
    // Fused operations
    FusedMatmulBiasRelu,
    Return
};

[[nodiscard]] constexpr std::string_view to_string(Opcode op) noexcept {
    switch (op) {
        case Opcode::Parameter:           return "param";
        case Opcode::ConstantInt:         return "const.i32";
        case Opcode::ConstantFloat:       return "const.f32";
        case Opcode::ConstantBool:        return "const.bool";
        case Opcode::ConstantTensor:      return "const.tensor";
        case Opcode::Add:                 return "add";
        case Opcode::Sub:                 return "sub";
        case Opcode::Mul:                 return "mul";
        case Opcode::Div:                 return "div";
        case Opcode::Matmul:              return "matmul";
        case Opcode::Relu:                return "relu";
        case Opcode::Sum:                 return "sum";
        case Opcode::Transpose:           return "transpose";
        case Opcode::Reshape:             return "reshape";
        case Opcode::LoadTensor:          return "load_tensor";
        case Opcode::FusedMatmulBiasRelu: return "fused_matmul_bias_relu";
        case Opcode::Return:              return "return";
    }
    return "<unknown_op>";
}

using LiteralValue = std::variant<std::monostate, int64_t, double, bool, std::string, std::vector<float>>;

struct Value {
    ValueId id{INVALID_VALUE_ID};
    TypePtr type{nullptr};
    std::string name_hint{};
};

struct Instruction {
    ValueId result_id{INVALID_VALUE_ID};
    Opcode opcode{Opcode::Return};
    std::vector<ValueId> operands{};
    LiteralValue literal{};
    TypePtr type{nullptr};
    SourceSpan span{};

    [[nodiscard]] bool has_result() const noexcept {
        return opcode != Opcode::Return;
    }
};

struct IRFunction {
    std::string name;
    std::vector<ValueId> parameters;
    TypePtr return_type;
    std::vector<Instruction> instructions;
    std::unordered_map<ValueId, Value> values;

    [[nodiscard]] const Value* get_value(ValueId id) const {
        auto it = values.find(id);
        if (it != values.end()) {
            return &it->second;
        }
        return nullptr;
    }
};

struct IRModule {
    std::vector<IRFunction> functions;

    [[nodiscard]] const IRFunction* get_function(std::string_view name) const {
        for (const auto& fn : functions) {
            if (fn.name == name) {
                return &fn;
            }
        }
        return nullptr;
    }
};

// Textual IR formatting
std::string dump_function(const IRFunction& fn);
std::string dump_ir(const IRModule& module);

} // namespace keryth
