#pragma once

#include <memory>
#include <string>
#include <vector>
#include "keryth/ast.hpp"
#include "keryth/parser.hpp"
#include "keryth/symbol_table.hpp"
#include "keryth/types.hpp"

namespace keryth {

class TypeChecker {
public:
    TypeChecker() = default;

    bool check_program(const ProgramAST& program);

    [[nodiscard]] bool has_errors() const noexcept { return !diagnostics_.empty(); }
    [[nodiscard]] const std::vector<Diagnostic>& diagnostics() const noexcept { return diagnostics_; }

    [[nodiscard]] TypePtr resolve_type(const TypeNode& node);
    [[nodiscard]] TypePtr check_expression(const ExprNode& expr);

    // Query inferred types for expressions or declarations
    [[nodiscard]] TypePtr get_node_type(const ASTNode* node) const;

private:
    void record_error(SourceSpan span, std::string message);

    void check_function(const FunctionDeclAST& fn);
    void check_statement(const StmtNode& stmt);
    void check_let_statement(const LetStmtAST& stmt);
    void check_return_statement(const ReturnStmtAST& stmt);

    TypePtr check_identifier(const IdentifierExprAST& expr);
    TypePtr check_binary_expr(const BinaryExprAST& expr);
    TypePtr check_call_expr(const CallExprAST& expr);

    SymbolTable symbol_table_;
    std::vector<Diagnostic> diagnostics_;
    TypePtr current_function_return_type_{nullptr};
    std::string current_function_name_{};

    std::unordered_map<const ASTNode*, TypePtr> node_types_;
};

} // namespace keryth
