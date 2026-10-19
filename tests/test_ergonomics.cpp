#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string_view>
#include "keryth/arena_executor.hpp"
#include "keryth/ir_builder.hpp"
#include "keryth/lexer.hpp"
#include "keryth/llvm_codegen.hpp"
#include "keryth/memory_planner.hpp"
#include "keryth/optimizer.hpp"
#include "keryth/parser.hpp"
#include "keryth/runtime.hpp"
#include "keryth/type_checker.hpp"

using namespace keryth;

void test_tensor_literals() {
    std::string_view code = R"(
fn make_weights() -> Tensor<f32, [2, 2]> {
    let w: Tensor<f32, [2, 2]> = [[1.5, -2.0],
                                  [0.5,  3.0]];
    return w;
}
)";

    Lexer lexer(code, "lit.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);

    Interpreter interp;
    auto res = interp.execute_function(module.functions[0], {});
    assert(std::holds_alternative<Tensor>(res));
    const auto& t = std::get<Tensor>(res);

    assert((t.shape() == std::vector<int64_t>{2, 2}));
    assert(std::fabs(t.at_flat(0) - 1.5f) < 1e-6f);
    assert(std::fabs(t.at_flat(1) - (-2.0f)) < 1e-6f);
    assert(std::fabs(t.at_flat(2) - 0.5f) < 1e-6f);
    assert(std::fabs(t.at_flat(3) - 3.0f) < 1e-6f);

    std::cout << "[PASS] test_tensor_literals (Nested 2D tensor literal parsed and evaluated)\n";
}

void test_transpose_and_reshape() {
    std::string_view code = R"(
fn transform(x: Tensor<f32, [2, 3]>) -> Tensor<f32, [6]> {
    let xt: Tensor<f32, [3, 2]> = transpose(x);
    let y: Tensor<f32, [6]> = reshape(xt);
    return y;
}
)";

    Lexer lexer(code, "transform.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);

    // X: [[1, 2, 3], [4, 5, 6]]
    Tensor x({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Interpreter interp;
    auto res = interp.execute_function(module.functions[0], {x});
    assert(std::holds_alternative<Tensor>(res));
    const auto& t = std::get<Tensor>(res);

    // X^T = [[1, 4], [2, 5], [3, 6]]
    // Reshaped to 1D = [1, 4, 2, 5, 3, 6]
    assert((t.shape() == std::vector<int64_t>{6}));
    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (int64_t i = 0; i < 6; ++i) {
        assert(std::fabs(t.at_flat(i) - expected[i]) < 1e-6f);
    }

    std::cout << "[PASS] test_transpose_and_reshape (Matrix transpose and zero-overhead view reshape)\n";
}

void test_binary_weights_save_load() {
    std::string bin_path = "/tmp/keryth_weights_test.bin";

    // Create and save reference weights
    Tensor w_saved({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    w_saved.save_to_binary(bin_path);

    std::string code = R"(
fn load_model(x: Tensor<f32, [1, 2]>) -> Tensor<f32, [1, 2]> {
    let w: Tensor<f32, [2, 2]> = load_tensor("/tmp/keryth_weights_test.bin");
    let y: Tensor<f32, [1, 2]> = matmul(x, w);
    return y;
}
)";

    Lexer lexer(code, "load.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);

    Tensor x({1, 2}, {1.0f, 2.0f});

    // Execute via Interpreter
    Interpreter interp;
    auto res = interp.execute_function(module.functions[0], {x});
    assert(std::holds_alternative<Tensor>(res));
    const auto& t = std::get<Tensor>(res);

    // [1, 2] * [[10, 20], [30, 40]] = [1*10 + 2*30, 1*20 + 2*40] = [70, 100]
    assert((t.shape() == std::vector<int64_t>{1, 2}));
    assert(std::fabs(t.at_flat(0) - 70.0f) < 1e-5f);
    assert(std::fabs(t.at_flat(1) - 100.0f) < 1e-5f);

    // Also verify via ArenaExecutor
    MemoryPlanner planner(64);
    auto plan = planner.plan(module.functions[0]);
    ArenaExecutor arena_exec;
    auto arena_res = arena_exec.execute(module.functions[0], plan, {x});
    const auto& t_arena = std::get<Tensor>(arena_res);
    assert(std::fabs(t_arena.at_flat(0) - 70.0f) < 1e-5f);
    assert(std::fabs(t_arena.at_flat(1) - 100.0f) < 1e-5f);

    std::remove(bin_path.c_str());
    std::cout << "[PASS] test_binary_weights_save_load (Binary weights ingested and computed)\n";
}

int main() {
    test_tensor_literals();
    test_transpose_and_reshape();
    test_binary_weights_save_load();

    std::cout << "\nAll Language Ergonomics tests passed successfully!\n";
    return 0;
}
