#include <cassert>
#include <iostream>
#include <string_view>
#include "keryth/lexer.hpp"
#include "keryth/parser.hpp"
#include "keryth/type_checker.hpp"

using namespace keryth;

void test_valid_canonical_linear() {
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
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    bool success = checker.check_program(*program);
    if (!success) {
        for (const auto& diag : checker.diagnostics()) {
            std::cerr << "Diagnostic: " << diag.message << " at " << diag.span.to_string() << "\n";
        }
    }
    assert(success);
    assert(!checker.has_errors());

    std::cout << "[PASS] test_valid_canonical_linear\n";
}

void test_scalar_arithmetic_valid() {
    std::string_view code = R"(
fn compute(a: f32, b: f32) -> f32 {
    let x: f32 = a + b * 2.0;
    return relu(x);
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    std::cout << "[PASS] test_scalar_arithmetic_valid\n";
}

void test_broadcasting_valid() {
    std::string_view code = R"(
fn test_broadcast(a: Tensor<f32, [4, 1]>,
                  b: Tensor<f32, [1, 5]>) -> Tensor<f32, [4, 5]> {
    let c: Tensor<f32, [4, 5]> = a + b;
    return c;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    std::cout << "[PASS] test_broadcasting_valid\n";
}

void test_broadcasting_incompatible_rejected() {
    std::string_view code = R"(
fn bad_broadcast(a: Tensor<f32, [2, 3]>,
                 b: Tensor<f32, [2, 4]>) -> Tensor<f32, [2, 3]> {
    let c: Tensor<f32, [2, 3]> = a + b;
    return c;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    bool ok = checker.check_program(*program);
    assert(!ok);
    assert(checker.has_errors());

    std::cout << "[PASS] test_broadcasting_incompatible_rejected\n";
}

void test_matmul_inner_mismatch_rejected() {
    std::string_view code = R"(
fn bad_matmul(x: Tensor<f32, [2, 3]>,
              w: Tensor<f32, [4, 2]>) -> Tensor<f32, [2, 2]> {
    let z: Tensor<f32, [2, 2]> = matmul(x, w);
    return z;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(!checker.check_program(*program));
    assert(checker.has_errors());

    std::cout << "[PASS] test_matmul_inner_mismatch_rejected\n";
}

void test_matmul_non_matrix_rejected() {
    std::string_view code = R"(
fn bad_matmul_rank(x: Tensor<f32, [2, 3]>,
                   w: Tensor<f32, [3]>) -> Tensor<f32, [2]> {
    let z: Tensor<f32, [2]> = matmul(x, w);
    return z;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(!checker.check_program(*program));
    assert(checker.has_errors());

    std::cout << "[PASS] test_matmul_non_matrix_rejected\n";
}

void test_let_binding_type_mismatch_rejected() {
    std::string_view code = R"(
fn bad_let() -> i32 {
    let x: f32 = 42;
    return 0;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(!checker.check_program(*program));
    assert(checker.has_errors());

    std::cout << "[PASS] test_let_binding_type_mismatch_rejected\n";
}

void test_return_type_mismatch_rejected() {
    std::string_view code = R"(
fn bad_return(x: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 4]> {
    return x;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(!checker.check_program(*program));
    assert(checker.has_errors());

    std::cout << "[PASS] test_return_type_mismatch_rejected\n";
}

void test_undeclared_identifier_rejected() {
    std::string_view code = R"(
fn bad_id() -> f32 {
    let y: f32 = x + 1.0;
    return y;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(!checker.check_program(*program));
    assert(checker.has_errors());

    std::cout << "[PASS] test_undeclared_identifier_rejected\n";
}

void test_duplicate_variable_rejected() {
    std::string_view code = R"(
fn dup_var() -> i32 {
    let x: i32 = 1;
    let x: i32 = 2;
    return x;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(!checker.check_program(*program));
    assert(checker.has_errors());

    std::cout << "[PASS] test_duplicate_variable_rejected\n";
}

int main() {
    test_valid_canonical_linear();
    test_scalar_arithmetic_valid();
    test_broadcasting_valid();
    test_broadcasting_incompatible_rejected();
    test_matmul_inner_mismatch_rejected();
    test_matmul_non_matrix_rejected();
    test_let_binding_type_mismatch_rejected();
    test_return_type_mismatch_rejected();
    test_undeclared_identifier_rejected();
    test_duplicate_variable_rejected();

    std::cout << "\nAll Type & Shape Checker tests passed successfully!\n";
    return 0;
}
