#include <cassert>
#include <cmath>
#include <iostream>
#include <string_view>
#include "keryth/arena_executor.hpp"
#include "keryth/ir_builder.hpp"
#include "keryth/lexer.hpp"
#include "keryth/memory_planner.hpp"
#include "keryth/optimizer.hpp"
#include "keryth/parser.hpp"
#include "keryth/runtime.hpp"
#include "keryth/type_checker.hpp"

using namespace keryth;

void test_liveness_intervals() {
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

    MemoryPlanner planner(64);
    auto lifetimes = planner.analyze_liveness(module.functions[0]);

    assert(lifetimes.size() >= 4);
    for (const auto& [vid, lt] : lifetimes) {
        assert(lt.first_def <= lt.last_use);
    }

    std::cout << "[PASS] test_liveness_intervals\n";
}

void test_memory_plan_soundness() {
    std::string_view code = R"(
fn mlp(x: Tensor<f32, [4, 2]>,
       w1: Tensor<f32, [2, 4]>,
       b1: Tensor<f32, [4]>,
       w2: Tensor<f32, [4, 1]>,
       b2: Tensor<f32, [1]>) -> Tensor<f32, [4, 1]> {
    let z1: Tensor<f32, [4, 4]> = matmul(x, w1);
    let a1: Tensor<f32, [4, 4]> = z1 + b1;
    let h: Tensor<f32, [4, 4]> = relu(a1);
    let z2: Tensor<f32, [4, 1]> = matmul(h, w2);
    let out: Tensor<f32, [4, 1]> = z2 + b2;
    return out;
}
)";

    Lexer lexer(code, "mlp.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);

    MemoryPlanner planner(64);
    MemoryPlan plan = planner.plan(module.functions[0]);

    assert(plan.total_arena_bytes > 0);
    assert(plan.total_arena_bytes % 64 == 0); // 64-byte aligned arena

    // Verify spatial soundness: any overlapping lifetimes must NOT overlap in memory offsets
    std::vector<ValueLifetime> lts;
    for (const auto& [vid, lt] : plan.lifetimes) {
        lts.push_back(lt);
    }

    for (size_t i = 0; i < lts.size(); ++i) {
        for (size_t j = i + 1; j < lts.size(); ++j) {
            const auto& a = lts[i];
            const auto& b = lts[j];
            if (a.size_bytes == 0 || b.size_bytes == 0) continue;

            bool time_overlap = !(a.last_use < b.first_def || b.last_use < a.first_def);
            if (time_overlap) {
                size_t off_a = plan.offsets.at(a.id);
                size_t off_b = plan.offsets.at(b.id);
                size_t end_a = off_a + a.size_bytes;
                size_t end_b = off_b + b.size_bytes;

                bool space_overlap = !(end_a <= off_b || end_b <= off_a);
                assert(!space_overlap && "Memory safety violation: overlapping lifetimes share conflicting arena space!");
            }
        }
    }

    // Verify significant memory footprint reduction on multi-layer pipeline
    assert(plan.total_arena_bytes <= plan.unoptimized_total_bytes);
    std::cout << "[PASS] test_memory_plan_soundness (Unoptimized: " << plan.unoptimized_total_bytes
              << " B -> Planned Arena: " << plan.total_arena_bytes
              << " B, Reduction: " << plan.memory_saving_percent() << "%)\n";
}

void test_arena_executor_numerical_equivalence() {
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

    // Inputs
    Tensor x({2, 3}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.0f});
    Tensor w({3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f});
    Tensor b({2}, {0.5f, -2.0f});
    std::vector<RuntimeValue> args = {x, w, b};

    // 1. Reference interpreter execution
    Interpreter interp;
    RuntimeValue interp_res = interp.execute_function(module.functions[0], args);
    const auto& interp_tensor = std::get<Tensor>(interp_res);

    // 2. Zero-allocation Arena execution
    MemoryPlanner planner(64);
    MemoryPlan plan = planner.plan(module.functions[0]);

    ArenaExecutor arena_exec;
    RuntimeValue arena_res = arena_exec.execute(module.functions[0], plan, args);
    const auto& arena_tensor = std::get<Tensor>(arena_res);

    // Verify numerical equivalence
    assert(interp_tensor.shape() == arena_tensor.shape());
    assert(interp_tensor.numel() == arena_tensor.numel());

    for (int64_t i = 0; i < interp_tensor.numel(); ++i) {
        assert(std::fabs(interp_tensor.at_flat(i) - arena_tensor.at_flat(i)) < 1e-6f);
    }

    std::cout << "[PASS] test_arena_executor_numerical_equivalence (Exact match with reference interpreter)\n";
}

int main() {
    test_liveness_intervals();
    test_memory_plan_soundness();
    test_arena_executor_numerical_equivalence();

    std::cout << "\nAll Static Memory Planner tests passed successfully!\n";
    return 0;
}
