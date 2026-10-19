#include <cassert>
#include <cmath>
#include <iostream>
#include <string_view>
#include "keryth/ir_builder.hpp"
#include "keryth/lexer.hpp"
#include "keryth/optimizer.hpp"
#include "keryth/parser.hpp"
#include "keryth/runtime.hpp"
#include "keryth/type_checker.hpp"

using namespace keryth;

void test_constant_folding() {
    std::string_view code = R"(
fn const_fold() -> i32 {
    let a: i32 = 10 + 20 * 2;
    return a;
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

    ConstantFoldingPass fold_pass;
    DeadCodeEliminationPass dce_pass;
    fold_pass.run_on_function(module.functions[0]);
    dce_pass.run_on_function(module.functions[0]);

    // Should fold into a single ConstantInt (50) and a Return
    const auto& insts = module.functions[0].instructions;
    assert(insts.size() == 2);
    assert(insts[0].opcode == Opcode::ConstantInt);
    assert(std::get<int64_t>(insts[0].literal) == 50);
    assert(insts[1].opcode == Opcode::Return);

    std::cout << "[PASS] test_constant_folding\n";
}

void test_dead_code_elimination() {
    std::string_view code = R"(
fn dce_test(x: i32) -> i32 {
    let dead1: i32 = x + 100;
    let dead2: i32 = dead1 * 2;
    let live: i32 = x + 1;
    return live;
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

    DeadCodeEliminationPass dce_pass;
    dce_pass.run_on_function(module.functions[0]);

    // dead1 and dead2 must be eliminated
    for (const auto& inst : module.functions[0].instructions) {
        if (inst.has_result()) {
            const auto* v = module.functions[0].get_value(inst.result_id);
            if (v) {
                assert(v->name_hint != "dead1" && v->name_hint != "dead2");
            }
        }
    }

    std::cout << "[PASS] test_dead_code_elimination\n";
}

void test_common_subexpression_elimination() {
    std::string_view code = R"(
fn cse_test(a: i32, b: i32) -> i32 {
    let x: i32 = a + b;
    let y: i32 = a + b;
    let z: i32 = x * y;
    return z;
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

    CommonSubexpressionEliminationPass cse_pass;
    DeadCodeEliminationPass dce_pass;
    cse_pass.run_on_function(module.functions[0]);
    dce_pass.run_on_function(module.functions[0]);

    int add_count = 0;
    for (const auto& inst : module.functions[0].instructions) {
        if (inst.opcode == Opcode::Add) {
            add_count++;
        }
    }
    // Only 1 Add instruction should remain after CSE + DCE
    assert(add_count == 1);

    std::cout << "[PASS] test_common_subexpression_elimination\n";
}

void test_operator_fusion_and_numerical_equivalence() {
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
    IRModule unopt_module = lowerer.lower_program(*program);
    IRModule opt_module = lowerer.lower_program(*program);

    // Run optimization pipeline on opt_module
    PassManager pm;
    pm.add_default_passes();
    pm.run(opt_module);

    // Verify fusion occurred
    bool has_fused_op = false;
    for (const auto& inst : opt_module.functions[0].instructions) {
        if (inst.opcode == Opcode::FusedMatmulBiasRelu) {
            has_fused_op = true;
            assert(inst.operands.size() == 3);
        }
    }
    assert(has_fused_op);

    // Verify unoptimized instruction count > optimized instruction count
    assert(opt_module.functions[0].instructions.size() < unopt_module.functions[0].instructions.size());

    // Prepare test inputs
    Tensor x({2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.0f});
    Tensor w({3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f});
    Tensor b({2}, {0.5f, -2.0f});

    Interpreter interp;
    std::vector<RuntimeValue> args = {x, w, b};

    RuntimeValue unopt_res = interp.execute_function(unopt_module.functions[0], args);
    RuntimeValue opt_res = interp.execute_function(opt_module.functions[0], args);

    const auto& unopt_tensor = std::get<Tensor>(unopt_res);
    const auto& opt_tensor = std::get<Tensor>(opt_res);

    assert(unopt_tensor.shape() == opt_tensor.shape());
    assert(unopt_tensor.numel() == opt_tensor.numel());

    for (int64_t i = 0; i < unopt_tensor.numel(); ++i) {
        assert(std::fabs(unopt_tensor.at_flat(i) - opt_tensor.at_flat(i)) < 1e-6f);
    }

    std::cout << "[PASS] test_operator_fusion_and_numerical_equivalence (Output matched perfectly: [4.5, 0, 0, 0])\n";
}

int main() {
    test_constant_folding();
    test_dead_code_elimination();
    test_common_subexpression_elimination();
    test_operator_fusion_and_numerical_equivalence();

    std::cout << "\nAll Optimizer tests passed successfully!\n";
    return 0;
}
