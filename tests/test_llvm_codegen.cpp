#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include "keryth/ir_builder.hpp"
#include "keryth/lexer.hpp"
#include "keryth/llvm_codegen.hpp"
#include "keryth/parser.hpp"
#include "keryth/type_checker.hpp"

using namespace keryth;

void test_emit_llvm_ir_syntax() {
    std::string_view code = R"(
fn add_scalars(a: i32, b: i32) -> i32 {
    let sum: i32 = a + b;
    return sum;
}
)";

    Lexer lexer(code, "add.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);

    LLVMCodeGen codegen;
    std::string ll_ir = codegen.emit_module(module);

    assert(ll_ir.find("define i32 @add_scalars(i32 %v0, i32 %v1)") != std::string::npos);
    assert(ll_ir.find("ret i32 %v") != std::string::npos);

    std::cout << "[PASS] test_emit_llvm_ir_syntax\n";
}

void test_native_compile_and_run_scalar() {
    std::string_view code = R"(
fn compute(a: i32, b: i32) -> i32 {
    let x: i32 = a + b * 2;
    return x;
}
)";

    Lexer lexer(code, "compute.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();
    assert(!parser.has_errors());

    TypeChecker checker;
    assert(checker.check_program(*program));

    ASTLowerer lowerer(checker);
    IRModule module = lowerer.lower_program(*program);

    LLVMCodeGen codegen;
    std::string bin_path = "/tmp/keryth_test_scalar_bin";

    bool ok = codegen.compile_to_native_binary(module, bin_path, 3);
    assert(ok && "Native compilation of scalar function failed");

    // Execute the compiled native binary
    int ret = std::system((bin_path + " > /dev/null 2>&1").c_str());
    assert(ret == 0 && "Native execution failed");

    std::remove(bin_path.c_str());
    std::cout << "[PASS] test_native_compile_and_run_scalar (Native machine binary executed with 0 exit code)\n";
}

void test_native_compile_and_run_linear_layer() {
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

    LLVMCodeGen codegen;
    std::string bin_path = "/tmp/keryth_test_linear_bin";

    bool ok = codegen.compile_to_native_binary(module, bin_path, 3);
    assert(ok && "Native compilation of linear layer failed");

    // Execute the compiled native binary
    int ret = std::system((bin_path + " > /dev/null 2>&1").c_str());
    assert(ret == 0 && "Native tensor execution failed");

    std::remove(bin_path.c_str());
    std::cout << "[PASS] test_native_compile_and_run_linear_layer (Native tensor kernel compiled & executed)\n";
}

int main() {
    test_emit_llvm_ir_syntax();
    test_native_compile_and_run_scalar();
    test_native_compile_and_run_linear_layer();

    std::cout << "\nAll LLVM Native Codegen tests passed successfully!\n";
    return 0;
}
