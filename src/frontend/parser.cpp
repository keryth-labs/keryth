#include "keryth/parser.hpp"
#include <charconv>
#include <string>

namespace keryth {

Parser::Parser(std::vector<Token> tokens) noexcept
    : tokens_(std::move(tokens)) {}

const Token& Parser::peek() const noexcept {
    if (current_ < tokens_.size()) {
        return tokens_[current_];
    }
    static const Token eof_token{TokenKind::EndOfFile, "", SourceSpan{}};
    return eof_token;
}

const Token& Parser::peek_next() const noexcept {
    if (current_ + 1 < tokens_.size()) {
        return tokens_[current_ + 1];
    }
    return peek();
}

const Token& Parser::previous() const noexcept {
    if (current_ > 0 && current_ - 1 < tokens_.size()) {
        return tokens_[current_ - 1];
    }
    return peek();
}

bool Parser::is_at_end() const noexcept {
    return peek().is(TokenKind::EndOfFile);
}

const Token& Parser::advance() noexcept {
    if (!is_at_end()) {
        current_++;
    }
    return previous();
}

bool Parser::check(TokenKind kind) const noexcept {
    if (is_at_end()) {
        return kind == TokenKind::EndOfFile;
    }
    return peek().is(kind);
}

bool Parser::match(TokenKind kind) noexcept {
    if (check(kind)) {
        advance();
        return true;
    }
    return false;
}

const Token* Parser::consume(TokenKind kind, std::string_view error_msg) {
    if (check(kind)) {
        return &advance();
    }
    record_error(peek().span, std::string(error_msg));
    return nullptr;
}

void Parser::record_error(SourceSpan span, std::string message) {
    diagnostics_.push_back(Diagnostic{
        .message = std::move(message),
        .span = span
    });
}

void Parser::synchronize() {
    advance();
    while (!is_at_end()) {
        if (previous().is(TokenKind::Semicolon) || previous().is(TokenKind::RBrace)) {
            return;
        }
        switch (peek().kind) {
            case TokenKind::KwFn:
            case TokenKind::KwLet:
            case TokenKind::KwReturn:
                return;
            default:
                break;
        }
        advance();
    }
}

std::unique_ptr<ProgramAST> Parser::parse_program() {
    std::vector<std::unique_ptr<FunctionDeclAST>> functions;
    SourceLocation start_loc = peek().span.start;

    while (!is_at_end()) {
        if (check(TokenKind::KwFn)) {
            auto fn = parse_function_decl();
            if (fn) {
                functions.push_back(std::move(fn));
            } else {
                synchronize();
            }
        } else {
            record_error(peek().span, "Expected top-level function declaration ('fn')");
            synchronize();
        }
    }

    SourceSpan span{start_loc, previous().span.end};
    return std::make_unique<ProgramAST>(std::move(functions), span);
}

std::unique_ptr<FunctionDeclAST> Parser::parse_function_decl() {
    const Token* fn_tok = consume(TokenKind::KwFn, "Expected 'fn'");
    if (!fn_tok) return nullptr;
    SourceLocation start_loc = fn_tok->span.start;

    const Token* name_tok = consume(TokenKind::Identifier, "Expected function name");
    if (!name_tok) return nullptr;
    std::string fn_name(name_tok->lexeme);

    if (!consume(TokenKind::LParen, "Expected '(' after function name")) {
        return nullptr;
    }

    std::vector<ParamDecl> params = parse_parameter_list();

    if (!consume(TokenKind::RParen, "Expected ')' after parameter list")) {
        return nullptr;
    }

    if (!consume(TokenKind::Arrow, "Expected '->' before return type")) {
        return nullptr;
    }

    auto ret_type = parse_type();
    if (!ret_type) return nullptr;

    if (!consume(TokenKind::LBrace, "Expected '{' to start function body")) {
        return nullptr;
    }

    std::vector<std::unique_ptr<StmtNode>> body;
    while (!check(TokenKind::RBrace) && !is_at_end()) {
        auto stmt = parse_statement();
        if (stmt) {
            body.push_back(std::move(stmt));
        } else {
            synchronize();
        }
    }

    const Token* rbrace = consume(TokenKind::RBrace, "Expected '}' after function body");
    if (!rbrace) return nullptr;

    SourceSpan span{start_loc, rbrace->span.end};
    return std::make_unique<FunctionDeclAST>(
        std::move(fn_name),
        std::move(params),
        std::move(ret_type),
        std::move(body),
        span
    );
}

std::vector<ParamDecl> Parser::parse_parameter_list() {
    std::vector<ParamDecl> params;
    if (check(TokenKind::RParen)) {
        return params;
    }

    while (true) {
        const Token* p_name = consume(TokenKind::Identifier, "Expected parameter name");
        if (!p_name) break;

        SourceLocation start_loc = p_name->span.start;
        if (!consume(TokenKind::Colon, "Expected ':' after parameter name")) {
            break;
        }

        auto p_type = parse_type();
        if (!p_type) break;

        SourceSpan span{start_loc, p_type->span.end};
        params.push_back(ParamDecl{
            .name = std::string(p_name->lexeme),
            .type = std::move(p_type),
            .span = span
        });

        if (!match(TokenKind::Comma)) {
            break;
        }
    }

    return params;
}

std::unique_ptr<TypeNode> Parser::parse_type() {
    SourceLocation start_loc = peek().span.start;

    if (match(TokenKind::KwF32)) {
        return std::make_unique<ScalarTypeNode>(ScalarKind::F32, SourceSpan{start_loc, previous().span.end});
    }
    if (match(TokenKind::KwI32)) {
        return std::make_unique<ScalarTypeNode>(ScalarKind::I32, SourceSpan{start_loc, previous().span.end});
    }
    if (match(TokenKind::KwBool)) {
        return std::make_unique<ScalarTypeNode>(ScalarKind::Bool, SourceSpan{start_loc, previous().span.end});
    }

    if (match(TokenKind::KwTensor)) {
        if (!consume(TokenKind::LAngle, "Expected '<' after 'Tensor'")) {
            return nullptr;
        }

        ScalarKind elem_kind = ScalarKind::F32;
        if (match(TokenKind::KwF32)) {
            elem_kind = ScalarKind::F32;
        } else if (match(TokenKind::KwI32)) {
            elem_kind = ScalarKind::I32;
        } else if (match(TokenKind::KwBool)) {
            elem_kind = ScalarKind::Bool;
        } else {
            record_error(peek().span, "Expected scalar element type (f32, i32, or bool) in Tensor type");
            return nullptr;
        }

        if (!consume(TokenKind::Comma, "Expected ',' after element type in Tensor")) {
            return nullptr;
        }

        if (!consume(TokenKind::LBracket, "Expected '[' to begin dimension list in Tensor type")) {
            return nullptr;
        }

        std::vector<int64_t> dims;
        if (!check(TokenKind::RBracket)) {
            while (true) {
                const Token* dim_tok = consume(TokenKind::IntegerLiteral, "Expected integer dimension size");
                if (!dim_tok) return nullptr;

                int64_t dim_val = 0;
                auto [ptr, ec] = std::from_chars(dim_tok->lexeme.data(), dim_tok->lexeme.data() + dim_tok->lexeme.size(), dim_val);
                if (ec != std::errc()) {
                    record_error(dim_tok->span, "Invalid integer literal for dimension");
                    return nullptr;
                }
                if (dim_val <= 0) {
                    record_error(dim_tok->span, "Tensor dimensions must be strictly positive");
                    return nullptr;
                }
                dims.push_back(dim_val);

                if (!match(TokenKind::Comma)) {
                    break;
                }
            }
        }

        if (!consume(TokenKind::RBracket, "Expected ']' after dimension list")) {
            return nullptr;
        }

        const Token* rangle = consume(TokenKind::RAngle, "Expected '>' to close Tensor type");
        if (!rangle) return nullptr;

        return std::make_unique<TensorTypeNode>(elem_kind, std::move(dims), SourceSpan{start_loc, rangle->span.end});
    }

    record_error(peek().span, "Expected type (f32, i32, bool, or Tensor<...>)");
    return nullptr;
}

std::unique_ptr<StmtNode> Parser::parse_statement() {
    if (check(TokenKind::KwLet)) {
        return parse_let_statement();
    }
    if (check(TokenKind::KwReturn)) {
        return parse_return_statement();
    }
    record_error(peek().span, "Expected statement ('let' or 'return')");
    return nullptr;
}

std::unique_ptr<LetStmtAST> Parser::parse_let_statement() {
    const Token* let_tok = consume(TokenKind::KwLet, "Expected 'let'");
    if (!let_tok) return nullptr;
    SourceLocation start_loc = let_tok->span.start;

    const Token* name_tok = consume(TokenKind::Identifier, "Expected variable name after 'let'");
    if (!name_tok) return nullptr;
    std::string var_name(name_tok->lexeme);

    if (!consume(TokenKind::Colon, "Expected ':' after variable name")) {
        return nullptr;
    }

    auto type = parse_type();
    if (!type) return nullptr;

    if (!consume(TokenKind::Equal, "Expected '=' in variable binding")) {
        return nullptr;
    }

    auto init = parse_expression();
    if (!init) return nullptr;

    const Token* semi = consume(TokenKind::Semicolon, "Expected ';' after let statement");
    if (!semi) return nullptr;

    SourceSpan span{start_loc, semi->span.end};
    return std::make_unique<LetStmtAST>(std::move(var_name), std::move(type), std::move(init), span);
}

std::unique_ptr<ReturnStmtAST> Parser::parse_return_statement() {
    const Token* ret_tok = consume(TokenKind::KwReturn, "Expected 'return'");
    if (!ret_tok) return nullptr;
    SourceLocation start_loc = ret_tok->span.start;

    auto expr = parse_expression();
    if (!expr) return nullptr;

    const Token* semi = consume(TokenKind::Semicolon, "Expected ';' after return statement");
    if (!semi) return nullptr;

    SourceSpan span{start_loc, semi->span.end};
    return std::make_unique<ReturnStmtAST>(std::move(expr), span);
}

std::unique_ptr<ExprNode> Parser::parse_expression() {
    return parse_additive();
}

std::unique_ptr<ExprNode> Parser::parse_additive() {
    auto lhs = parse_multiplicative();
    if (!lhs) return nullptr;

    while (check(TokenKind::Plus) || check(TokenKind::Minus)) {
        const Token& op_tok = advance();
        BinaryOpKind op = (op_tok.is(TokenKind::Plus)) ? BinaryOpKind::Add : BinaryOpKind::Sub;

        auto rhs = parse_multiplicative();
        if (!rhs) return nullptr;

        SourceSpan span{lhs->span.start, rhs->span.end};
        lhs = std::make_unique<BinaryExprAST>(op, std::move(lhs), std::move(rhs), span);
    }

    return lhs;
}

std::unique_ptr<ExprNode> Parser::parse_multiplicative() {
    auto lhs = parse_primary();
    if (!lhs) return nullptr;

    while (check(TokenKind::Star) || check(TokenKind::Slash)) {
        const Token& op_tok = advance();
        BinaryOpKind op = (op_tok.is(TokenKind::Star)) ? BinaryOpKind::Mul : BinaryOpKind::Div;

        auto rhs = parse_primary();
        if (!rhs) return nullptr;

        SourceSpan span{lhs->span.start, rhs->span.end};
        lhs = std::make_unique<BinaryExprAST>(op, std::move(lhs), std::move(rhs), span);
    }

    return lhs;
}

std::unique_ptr<ExprNode> Parser::parse_primary() {
    SourceLocation start_loc = peek().span.start;

    if (match(TokenKind::IntegerLiteral)) {
        int64_t val = 0;
        const auto& lex = previous().lexeme;
        auto [ptr, ec] = std::from_chars(lex.data(), lex.data() + lex.size(), val);
        if (ec != std::errc()) {
            record_error(previous().span, "Invalid integer literal");
            return nullptr;
        }
        return std::make_unique<IntegerLiteralExprAST>(val, previous().span);
    }

    if (match(TokenKind::FloatLiteral)) {
        std::string lex(previous().lexeme);
        if (!lex.empty() && (lex.back() == 'f' || lex.back() == 'F')) {
            lex.pop_back();
        }
        double val = std::stod(lex);
        return std::make_unique<FloatLiteralExprAST>(val, previous().span);
    }

    if (match(TokenKind::KwTrue)) {
        return std::make_unique<BoolLiteralExprAST>(true, previous().span);
    }
    if (match(TokenKind::KwFalse)) {
        return std::make_unique<BoolLiteralExprAST>(false, previous().span);
    }

    if (check(TokenKind::Identifier)) {
        const Token& id_tok = advance();
        std::string name(id_tok.lexeme);

        // Check for function call
        if (match(TokenKind::LParen)) {
            std::vector<std::unique_ptr<ExprNode>> args;
            if (!check(TokenKind::RParen)) {
                while (true) {
                    auto arg = parse_expression();
                    if (!arg) return nullptr;
                    args.push_back(std::move(arg));

                    if (!match(TokenKind::Comma)) {
                        break;
                    }
                }
            }
            const Token* rparen = consume(TokenKind::RParen, "Expected ')' after call arguments");
            if (!rparen) return nullptr;

            SourceSpan span{start_loc, rparen->span.end};
            return std::make_unique<CallExprAST>(std::move(name), std::move(args), span);
        }

        return std::make_unique<IdentifierExprAST>(std::move(name), id_tok.span);
    }

    if (match(TokenKind::LParen)) {
        auto inner = parse_expression();
        if (!inner) return nullptr;
        const Token* rparen = consume(TokenKind::RParen, "Expected ')' to match '('");
        if (!rparen) return nullptr;
        return inner;
    }

    record_error(peek().span, "Expected expression");
    return nullptr;
}

} // namespace keryth
