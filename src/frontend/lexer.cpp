#include "keryth/lexer.hpp"
#include <cctype>

namespace keryth {

Lexer::Lexer(std::string_view source, std::string_view filename) noexcept
    : source_(source), filename_(filename) {}

bool Lexer::is_at_end() const noexcept {
    return offset_ >= source_.size();
}

char Lexer::peek() const noexcept {
    if (is_at_end()) {
        return '\0';
    }
    return source_[offset_];
}

char Lexer::peek_next() const noexcept {
    if (offset_ + 1 >= source_.size()) {
        return '\0';
    }
    return source_[offset_ + 1];
}

char Lexer::advance() noexcept {
    if (is_at_end()) {
        return '\0';
    }
    char c = source_[offset_++];
    if (c == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    return c;
}

bool Lexer::match(char expected) noexcept {
    if (is_at_end() || source_[offset_] != expected) {
        return false;
    }
    advance();
    return true;
}

SourceLocation Lexer::current_location() const noexcept {
    return SourceLocation{
        .filename = filename_,
        .line = line_,
        .column = column_,
        .offset = offset_
    };
}

void Lexer::skip_whitespace_and_comments() {
    while (!is_at_end()) {
        char c = peek();
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
        } else if (c == '/' && peek_next() == '/') {
            // Line comment: skip until newline or EOF
            advance(); // consume first /
            advance(); // consume second /
            while (!is_at_end() && peek() != '\n') {
                advance();
            }
        } else if (c == '/' && peek_next() == '*') {
            // Block comment: skip until */ or EOF
            advance(); // consume /
            advance(); // consume *
            while (!is_at_end()) {
                if (peek() == '*' && peek_next() == '/') {
                    advance(); // consume *
                    advance(); // consume /
                    break;
                }
                advance();
            }
        } else {
            break;
        }
    }
}

Token Lexer::lex_identifier_or_keyword(SourceLocation start_loc) {
    uint32_t start_offset = start_loc.offset;

    while (!is_at_end() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')) {
        advance();
    }

    std::string_view lexeme = source_.substr(start_offset, offset_ - start_offset);
    SourceSpan span{start_loc, current_location()};

    TokenKind kind = TokenKind::Identifier;
    if (lexeme == "fn") {
        kind = TokenKind::KwFn;
    } else if (lexeme == "let") {
        kind = TokenKind::KwLet;
    } else if (lexeme == "return") {
        kind = TokenKind::KwReturn;
    } else if (lexeme == "Tensor") {
        kind = TokenKind::KwTensor;
    } else if (lexeme == "f32") {
        kind = TokenKind::KwF32;
    } else if (lexeme == "i32") {
        kind = TokenKind::KwI32;
    } else if (lexeme == "bool") {
        kind = TokenKind::KwBool;
    } else if (lexeme == "true") {
        kind = TokenKind::KwTrue;
    } else if (lexeme == "false") {
        kind = TokenKind::KwFalse;
    }

    return Token{kind, lexeme, span};
}

Token Lexer::lex_number(SourceLocation start_loc) {
    uint32_t start_offset = start_loc.offset;
    bool is_float = false;

    while (!is_at_end() && std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    // Fraction part
    if (peek() == '.' && std::isdigit(static_cast<unsigned char>(peek_next()))) {
        is_float = true;
        advance(); // consume '.'
        while (!is_at_end() && std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
    }

    // Exponent part
    if (peek() == 'e' || peek() == 'E') {
        char next = peek_next();
        if (std::isdigit(static_cast<unsigned char>(next)) ||
            ((next == '+' || next == '-') && offset_ + 2 < source_.size() &&
             std::isdigit(static_cast<unsigned char>(source_[offset_ + 2])))) {
            is_float = true;
            advance(); // consume 'e' or 'E'
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            while (!is_at_end() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        }
    }

    // Optional 'f' or 'F' suffix for float
    if (peek() == 'f' || peek() == 'F') {
        is_float = true;
        advance();
    }

    std::string_view lexeme = source_.substr(start_offset, offset_ - start_offset);
    SourceSpan span{start_loc, current_location()};

    return Token{is_float ? TokenKind::FloatLiteral : TokenKind::IntegerLiteral, lexeme, span};
}

Token Lexer::next_token() {
    skip_whitespace_and_comments();

    SourceLocation start_loc = current_location();

    if (is_at_end()) {
        return Token{TokenKind::EndOfFile, "", SourceSpan{start_loc, start_loc}};
    }

    char c = advance();

    // String literals
    if (c == '"') {
        while (!is_at_end() && peek() != '"') {
            if (peek() == '\n') break;
            advance();
        }
        if (is_at_end() || peek() != '"') {
            return Token{TokenKind::Invalid, source_.substr(start_loc.offset, offset_ - start_loc.offset), SourceSpan{start_loc, current_location()}};
        }
        advance(); // consume closing "
        std::string_view lexeme = source_.substr(start_loc.offset, offset_ - start_loc.offset);
        return Token{TokenKind::StringLiteral, lexeme, SourceSpan{start_loc, current_location()}};
    }

    // Identifiers and keywords
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        offset_--;
        column_--;
        return lex_identifier_or_keyword(start_loc);
    }

    // Numbers
    if (std::isdigit(static_cast<unsigned char>(c))) {
        offset_--;
        column_--;
        return lex_number(start_loc);
    }

    // Two-character symbols
    if (c == '-' && peek() == '>') {
        advance();
        return Token{TokenKind::Arrow, source_.substr(start_loc.offset, 2), SourceSpan{start_loc, current_location()}};
    }

    // Single-character tokens
    TokenKind kind = TokenKind::Invalid;
    switch (c) {
        case '(': kind = TokenKind::LParen; break;
        case ')': kind = TokenKind::RParen; break;
        case '[': kind = TokenKind::LBracket; break;
        case ']': kind = TokenKind::RBracket; break;
        case '{': kind = TokenKind::LBrace; break;
        case '}': kind = TokenKind::RBrace; break;
        case '<': kind = TokenKind::LAngle; break;
        case '>': kind = TokenKind::RAngle; break;
        case ':': kind = TokenKind::Colon; break;
        case ';': kind = TokenKind::Semicolon; break;
        case ',': kind = TokenKind::Comma; break;
        case '=': kind = TokenKind::Equal; break;
        case '+': kind = TokenKind::Plus; break;
        case '-': kind = TokenKind::Minus; break;
        case '*': kind = TokenKind::Star; break;
        case '/': kind = TokenKind::Slash; break;
        default:  kind = TokenKind::Invalid; break;
    }

    std::string_view lexeme = source_.substr(start_loc.offset, 1);
    return Token{kind, lexeme, SourceSpan{start_loc, current_location()}};
}

std::vector<Token> Lexer::tokenize_all() {
    std::vector<Token> tokens;
    while (true) {
        Token tok = next_token();
        tokens.push_back(tok);
        if (tok.kind == TokenKind::EndOfFile) {
            break;
        }
    }
    return tokens;
}

} // namespace keryth
