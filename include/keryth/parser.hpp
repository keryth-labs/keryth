#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "keryth/ast.hpp"
#include "keryth/token.hpp"

namespace keryth {

struct Diagnostic {
    std::string message;
    SourceSpan span;
};

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) noexcept;

    [[nodiscard]] std::unique_ptr<ProgramAST> parse_program();

    [[nodiscard]] bool has_errors() const noexcept { return !diagnostics_.empty(); }
    [[nodiscard]] const std::vector<Diagnostic>& diagnostics() const noexcept { return diagnostics_; }

private:
    // Token navigation
    [[nodiscard]] const Token& peek() const noexcept;
    [[nodiscard]] const Token& peek_next() const noexcept;
    [[nodiscard]] const Token& previous() const noexcept;
    [[nodiscard]] bool is_at_end() const noexcept;
    const Token& advance() noexcept;
    bool check(TokenKind kind) const noexcept;
    bool match(TokenKind kind) noexcept;
    const Token* consume(TokenKind kind, std::string_view error_msg);

    void record_error(SourceSpan span, std::string message);
    void synchronize();

    // Grammar rules
    std::unique_ptr<FunctionDeclAST> parse_function_decl();
    std::vector<ParamDecl> parse_parameter_list();
    std::unique_ptr<TypeNode> parse_type();
    std::unique_ptr<StmtNode> parse_statement();
    std::unique_ptr<LetStmtAST> parse_let_statement();
    std::unique_ptr<ReturnStmtAST> parse_return_statement();

    std::unique_ptr<ExprNode> parse_expression();
    std::unique_ptr<ExprNode> parse_additive();
    std::unique_ptr<ExprNode> parse_multiplicative();
    std::unique_ptr<ExprNode> parse_primary();

    std::vector<Token> tokens_;
    size_t current_{0};
    std::vector<Diagnostic> diagnostics_;
};

} // namespace keryth
