#pragma once

#include "keryth/ir.hpp"
#include <string>
#include <string_view>

namespace keryth {

class LLVMCodeGen {
public:
    explicit LLVMCodeGen(std::string target_triple = "") noexcept;

    [[nodiscard]] std::string emit_module(const IRModule& module) const;
    [[nodiscard]] std::string emit_function(const IRFunction& fn) const;

    [[nodiscard]] bool compile_to_native_binary(
        const IRModule& module,
        const std::string& output_binary_path,
        int opt_level = 3) const;

    [[nodiscard]] bool compile_to_object_file(
        const IRModule& module,
        const std::string& output_obj_path,
        int opt_level = 3) const;

private:
    [[nodiscard]] std::string llvm_type(const Type* type) const;
    void emit_instruction(const Instruction& inst, const IRFunction& fn, std::string& out) const;

    std::string target_triple_;
};

} // namespace keryth
