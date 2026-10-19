#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "keryth/ast.hpp"
#include "keryth/ir.hpp"
#include "keryth/type_checker.hpp"

namespace keryth {

class ASTLowerer {
public:
    explicit ASTLowerer(const TypeChecker& type_checker) noexcept
        : type_checker_(type_checker) {}

    [[nodiscard]] IRModule lower_program(const ProgramAST& program);

private:
    IRFunction lower_function(const FunctionDeclAST& fn);
    void lower_statement(const StmtNode& stmt, IRFunction& out_fn);
    ValueId lower_expression(const ExprNode& expr, IRFunction& out_fn);

    ValueId allocate_value(TypePtr type, std::string name_hint, IRFunction& out_fn);

    const TypeChecker& type_checker_;
    std::unordered_map<std::string, ValueId> var_to_val_;
    ValueId next_value_id_{0};
};

} // namespace keryth
