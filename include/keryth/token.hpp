#pragma once

#include <string_view>
#include "keryth/source_location.hpp"

namespace keryth {

enum class TokenKind {
    // Special
    EndOfFile,
    Invalid,

    // Literals & Identifiers
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,

    // Keywords
    KwFn,       // fn
    KwLet,      // let
    KwReturn,   // return
    KwTensor,   // Tensor
    KwF32,      // f32
    KwI32,      // i32
    KwBool,     // bool
    KwTrue,     // true
    KwFalse,    // false

    // Punctuation & Operators
    LParen,     // (
    RParen,     // )
    LBracket,   // [
    RBracket,   // ]
    LBrace,     // {
    RBrace,     // }
    LAngle,     // <
    RAngle,     // >
    Colon,      // :
    Semicolon,  // ;
    Comma,      // ,
    Arrow,      // ->
    Equal,      // =
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash       // /
};

struct Token {
    TokenKind kind{TokenKind::Invalid};
    std::string_view lexeme{};
    SourceSpan span{};

    [[nodiscard]] constexpr bool is(TokenKind expected) const noexcept {
        return kind == expected;
    }

    [[nodiscard]] constexpr bool is_not(TokenKind expected) const noexcept {
        return kind != expected;
    }
};

[[nodiscard]] constexpr std::string_view to_string(TokenKind kind) noexcept {
    switch (kind) {
        case TokenKind::EndOfFile:      return "EOF";
        case TokenKind::Invalid:        return "Invalid";
        case TokenKind::Identifier:     return "Identifier";
        case TokenKind::IntegerLiteral: return "IntegerLiteral";
        case TokenKind::FloatLiteral:   return "FloatLiteral";
        case TokenKind::StringLiteral:  return "StringLiteral";
        case TokenKind::KwFn:           return "fn";
        case TokenKind::KwLet:          return "let";
        case TokenKind::KwReturn:       return "return";
        case TokenKind::KwTensor:       return "Tensor";
        case TokenKind::KwF32:          return "f32";
        case TokenKind::KwI32:          return "i32";
        case TokenKind::KwBool:         return "bool";
        case TokenKind::KwTrue:         return "true";
        case TokenKind::KwFalse:        return "false";
        case TokenKind::LParen:         return "(";
        case TokenKind::RParen:         return ")";
        case TokenKind::LBracket:       return "[";
        case TokenKind::RBracket:       return "]";
        case TokenKind::LBrace:         return "{";
        case TokenKind::RBrace:         return "}";
        case TokenKind::LAngle:         return "<";
        case TokenKind::RAngle:         return ">";
        case TokenKind::Colon:          return ":";
        case TokenKind::Semicolon:      return ";";
        case TokenKind::Comma:          return ",";
        case TokenKind::Arrow:          return "->";
        case TokenKind::Equal:          return "=";
        case TokenKind::Plus:           return "+";
        case TokenKind::Minus:          return "-";
        case TokenKind::Star:           return "*";
        case TokenKind::Slash:          return "/";
    }
    return "<unknown>";
}

} // namespace keryth
