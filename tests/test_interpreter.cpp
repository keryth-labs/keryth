#include <cassert>
#include <cmath>
#include <iostream>
#include <string_view>
#include "keryth/ir_builder.hpp"
#include "keryth/lexer.hpp"
#include "keryth/parser.hpp"
#include "keryth/runtime.hpp"
#include "keryth/type_checker.hpp"

using namespace keryth;

void test_scalar_execution() {
    std::string_view code = R"(
fn calc(a: i32, b: i32) -> i32 {
    let sum: i32 = a + b * 2;
    return sum;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);
    assert(module.functions.size() == 1);

    Interpreter interp;
    std::vector<RuntimeValue> args = {int32_t{5}, int32_t{10}};
    RuntimeValue res = interp.execute_function(module.functions[0], args);

    assert(std::holds_alternative<int32_t>(res));
    assert(std::get<int32_t>(res) == 25);

    std::cout << "[PASS] test_scalar_execution\n";
}

void test_canonical_linear_execution() {
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
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);
    assert(module.functions.size() == 1);

    // X: [[1.0, 2.0, 3.0], [-1.0, -2.0, 0.0]] (shape: [2, 3])
    Tensor x({2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.0f});

    // W: [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]] (shape: [3, 2])
    Tensor w({3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f});

    // B: [0.5, -2.0] (shape: [2])
    Tensor b({2}, {0.5f, -2.0f});

    Interpreter interp;
    std::vector<RuntimeValue> args = {x, w, b};
    RuntimeValue res = interp.execute_function(module.functions[0], args);

    assert(std::holds_alternative<Tensor>(res));
    const auto& out_tensor = std::get<Tensor>(res);

    assert((out_tensor.shape() == std::vector<int64_t>{2, 2}));
    assert(out_tensor.numel() == 4);

    // Expected:
    // Z = [[4.0, -1.0], [-1.0, -2.0]]
    // Y = [[4.5, -3.0], [-0.5, -4.0]]
    // Relu(Y) = [[4.5, 0.0], [0.0, 0.0]]
    const auto& data = out_tensor.data();
    assert(std::fabs(data[0] - 4.5f) < 1e-5f);
    assert(std::fabs(data[1] - 0.0f) < 1e-5f);
    assert(std::fabs(data[2] - 0.0f) < 1e-5f);
    assert(std::fabs(data[3] - 0.0f) < 1e-5f);

    std::cout << "[PASS] test_canonical_linear_execution\n";
}

int main() {
    test_scalar_execution();
    test_canonical_linear_execution();

    std::cout << "\nAll Interpreter tests passed successfully!\n";
    return 0;
}
