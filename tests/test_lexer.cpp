#include <cassert>
#include <iostream>
#include <string_view>
#include <vector>
#include "keryth/lexer.hpp"
#include "keryth/token.hpp"

using namespace keryth;

void test_empty_and_whitespace() {
    Lexer lexer1("", "empty.kr");
    Token t1 = lexer1.next_token();
    assert(t1.is(TokenKind::EndOfFile));

    Lexer lexer2("   \t  \n  \r\n  ", "whitespace.kr");
    Token t2 = lexer2.next_token();
    assert(t2.is(TokenKind::EndOfFile));
    assert(t2.span.start.line == 3);

    // Comments
    Lexer lexer3("// Single line comment\nfn /* block comment */ main", "comments.kr");
    std::vector<Token> tokens = lexer3.tokenize_all();
    assert(tokens.size() == 3); // fn, main, EOF
    assert(tokens[0].is(TokenKind::KwFn));
    assert(tokens[0].lexeme == "fn");
    assert(tokens[0].span.start.line == 2);
    assert(tokens[1].is(TokenKind::Identifier));
    assert(tokens[1].lexeme == "main");
    assert(tokens[2].is(TokenKind::EndOfFile));

    std::cout << "[PASS] test_empty_and_whitespace\n";
}

void test_keywords_and_types() {
    std::string_view src = "fn let return Tensor f32 i32 bool true false";
    Lexer lexer(src);
    std::vector<Token> tokens = lexer.tokenize_all();

    assert(tokens.size() == 10); // 9 keywords + EOF
    assert(tokens[0].is(TokenKind::KwFn));
    assert(tokens[1].is(TokenKind::KwLet));
    assert(tokens[2].is(TokenKind::KwReturn));
    assert(tokens[3].is(TokenKind::KwTensor));
    assert(tokens[4].is(TokenKind::KwF32));
    assert(tokens[5].is(TokenKind::KwI32));
    assert(tokens[6].is(TokenKind::KwBool));
    assert(tokens[7].is(TokenKind::KwTrue));
    assert(tokens[8].is(TokenKind::KwFalse));
    assert(tokens[9].is(TokenKind::EndOfFile));

    std::cout << "[PASS] test_keywords_and_types\n";
}

void test_identifiers() {
    std::string_view src = "foo bar_123 _temp matmul relu f32_tensor";
    Lexer lexer(src);
    std::vector<Token> tokens = lexer.tokenize_all();

    assert(tokens.size() == 7);
    assert(tokens[0].lexeme == "foo" && tokens[0].is(TokenKind::Identifier));
    assert(tokens[1].lexeme == "bar_123" && tokens[1].is(TokenKind::Identifier));
    assert(tokens[2].lexeme == "_temp" && tokens[2].is(TokenKind::Identifier));
    assert(tokens[3].lexeme == "matmul" && tokens[3].is(TokenKind::Identifier));
    assert(tokens[4].lexeme == "relu" && tokens[4].is(TokenKind::Identifier));
    assert(tokens[5].lexeme == "f32_tensor" && tokens[5].is(TokenKind::Identifier));

    std::cout << "[PASS] test_identifiers\n";
}

void test_numeric_literals() {
    std::string_view src = "0 42 1024 3.14 0.001 1e-4 2.5f 10e3";
    Lexer lexer(src);
    std::vector<Token> tokens = lexer.tokenize_all();

    assert(tokens.size() == 9); // 8 numbers + EOF
    assert(tokens[0].is(TokenKind::IntegerLiteral) && tokens[0].lexeme == "0");
    assert(tokens[1].is(TokenKind::IntegerLiteral) && tokens[1].lexeme == "42");
    assert(tokens[2].is(TokenKind::IntegerLiteral) && tokens[2].lexeme == "1024");
    assert(tokens[3].is(TokenKind::FloatLiteral) && tokens[3].lexeme == "3.14");
    assert(tokens[4].is(TokenKind::FloatLiteral) && tokens[4].lexeme == "0.001");
    assert(tokens[5].is(TokenKind::FloatLiteral) && tokens[5].lexeme == "1e-4");
    assert(tokens[6].is(TokenKind::FloatLiteral) && tokens[6].lexeme == "2.5f");
    assert(tokens[7].is(TokenKind::FloatLiteral) && tokens[7].lexeme == "10e3");

    std::cout << "[PASS] test_numeric_literals\n";
}

void test_operators_and_delimiters() {
    std::string_view src = "( ) [ ] { } < > : ; , -> = + - * /";
    Lexer lexer(src);
    std::vector<Token> tokens = lexer.tokenize_all();

    assert(tokens.size() == 18);
    assert(tokens[0].is(TokenKind::LParen));
    assert(tokens[1].is(TokenKind::RParen));
    assert(tokens[2].is(TokenKind::LBracket));
    assert(tokens[3].is(TokenKind::RBracket));
    assert(tokens[4].is(TokenKind::LBrace));
    assert(tokens[5].is(TokenKind::RBrace));
    assert(tokens[6].is(TokenKind::LAngle));
    assert(tokens[7].is(TokenKind::RAngle));
    assert(tokens[8].is(TokenKind::Colon));
    assert(tokens[9].is(TokenKind::Semicolon));
    assert(tokens[10].is(TokenKind::Comma));
    assert(tokens[11].is(TokenKind::Arrow) && tokens[11].lexeme == "->");
    assert(tokens[12].is(TokenKind::Equal));
    assert(tokens[13].is(TokenKind::Plus));
    assert(tokens[14].is(TokenKind::Minus));
    assert(tokens[15].is(TokenKind::Star));
    assert(tokens[16].is(TokenKind::Slash));
    assert(tokens[17].is(TokenKind::EndOfFile));

    std::cout << "[PASS] test_operators_and_delimiters\n";
}

void test_source_locations() {
    std::string_view src = "fn foo() {\n    let x: i32 = 10;\n}";
    Lexer lexer(src, "test.kr");
    std::vector<Token> tokens = lexer.tokenize_all();

    // Line 1: fn (col 1), foo (col 4), ( (col 7), ) (col 8), { (col 10)
    assert(tokens[0].lexeme == "fn");
    assert(tokens[0].span.start.line == 1 && tokens[0].span.start.column == 1);
    assert(tokens[1].lexeme == "foo");
    assert(tokens[1].span.start.line == 1 && tokens[1].span.start.column == 4);
    assert(tokens[4].lexeme == "{");
    assert(tokens[4].span.start.line == 1 && tokens[4].span.start.column == 10);

    // Line 2: let (col 5)
    assert(tokens[5].lexeme == "let");
    assert(tokens[5].span.start.line == 2 && tokens[5].span.start.column == 5);

    // Line 3: } (col 1)
    assert(tokens[12].lexeme == "}");
    assert(tokens[12].span.start.line == 3 && tokens[12].span.start.column == 1);

    std::cout << "[PASS] test_source_locations\n";
}

void test_invalid_characters() {
    std::string_view src = "let x = @ + $;";
    Lexer lexer(src);
    std::vector<Token> tokens = lexer.tokenize_all();

    // let, x, =, @(invalid), +, $(invalid), ;, EOF
    assert(tokens[3].is(TokenKind::Invalid) && tokens[3].lexeme == "@");
    assert(tokens[5].is(TokenKind::Invalid) && tokens[5].lexeme == "$");

    std::cout << "[PASS] test_invalid_characters\n";
}

void test_full_linear_function() {
    std::string_view code = R"(
fn linear(x: Tensor<f32, [2, 3]>,
          w: Tensor<f32, [3, 2]>,
          b: Tensor<f32, [2]>) -> Tensor<f32, [2, 2]> {
    let z: Tensor<f32, [2, 2]> = matmul(x, w);
    let y: Tensor<f32, [2, 2]> = z + b;
    return relu(y);
}
)";

    Lexer lexer(code, "linear.kr");
    std::vector<Token> tokens = lexer.tokenize_all();

    // Ensure no invalid tokens
    for (const auto& tok : tokens) {
        assert(tok.kind != TokenKind::Invalid);
    }

    assert(tokens.front().is(TokenKind::KwFn));
    assert(tokens.back().is(TokenKind::EndOfFile));

    std::cout << "[PASS] test_full_linear_function (" << tokens.size() << " tokens)\n";
}

int main() {
    test_empty_and_whitespace();
    test_keywords_and_types();
    test_identifiers();
    test_numeric_literals();
    test_operators_and_delimiters();
    test_source_locations();
    test_invalid_characters();
    test_full_linear_function();

    std::cout << "\nAll Lexer tests passed successfully!\n";
    return 0;
}
