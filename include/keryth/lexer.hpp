#pragma once

#include <string_view>
#include <vector>
#include "keryth/token.hpp"

namespace keryth {

class Lexer {
public:
    explicit Lexer(std::string_view source, std::string_view filename = "") noexcept;

    [[nodiscard]] Token next_token();
    [[nodiscard]] std::vector<Token> tokenize_all();

private:
    [[nodiscard]] bool is_at_end() const noexcept;
    [[nodiscard]] char peek() const noexcept;
    [[nodiscard]] char peek_next() const noexcept;
    char advance() noexcept;
    bool match(char expected) noexcept;

    void skip_whitespace_and_comments();
    Token lex_identifier_or_keyword(SourceLocation start_loc);
    Token lex_number(SourceLocation start_loc);

    [[nodiscard]] SourceLocation current_location() const noexcept;

    std::string_view source_;
    std::string_view filename_;
    uint32_t offset_{0};
    uint32_t line_{1};
    uint32_t column_{1};
};

} // namespace keryth
