#include <cassert>
#include <iostream>
#include <string_view>
#include "keryth/ast.hpp"
#include "keryth/lexer.hpp"
#include "keryth/parser.hpp"

using namespace keryth;

void test_parse_scalar_function() {
    std::string_view code = R"(
fn add_scalars(a: i32, b: i32) -> i32 {
    let sum: i32 = a + b;
    return sum;
}
)";

    Lexer lexer(code, "add_scalars.kr");
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();

    assert(!parser.has_errors());
    assert(program != nullptr);
    assert(program->functions.size() == 1);

    const auto& fn = program->functions[0];
    assert(fn->name == "add_scalars");
    assert(fn->params.size() == 2);
    assert(fn->params[0].name == "a");
    assert(!fn->params[0].type->is_tensor());
    assert(fn->params[0].type->to_string() == "i32");
    assert(fn->params[1].name == "b");

    assert(fn->return_type->to_string() == "i32");
    assert(fn->body.size() == 2);

    auto let_stmt = dynamic_cast<LetStmtAST*>(fn->body[0].get());
    assert(let_stmt != nullptr);
    assert(let_stmt->var_name == "sum");
    assert(let_stmt->type_annotation->to_string() == "i32");

    auto bin_expr = dynamic_cast<BinaryExprAST*>(let_stmt->initializer.get());
    assert(bin_expr != nullptr);
    assert(bin_expr->op == BinaryOpKind::Add);

    auto ret_stmt = dynamic_cast<ReturnStmtAST*>(fn->body[1].get());
    assert(ret_stmt != nullptr);
    auto ret_id = dynamic_cast<IdentifierExprAST*>(ret_stmt->expr.get());
    assert(ret_id != nullptr);
    assert(ret_id->name == "sum");

    std::cout << "[PASS] test_parse_scalar_function\n";
}

void test_parse_operator_precedence() {
    std::string_view code = R"(
fn math(a: f32, b: f32, c: f32) -> f32 {
    return a + b * c;
}
)";

    Lexer lexer(code);
    Parser parser(lexer.tokenize_all());
    auto program = parser.parse_program();

    assert(!parser.has_errors());
    assert(program->functions.size() == 1);

    const auto& fn = program->functions[0];
    auto ret_stmt = dynamic_cast<ReturnStmtAST*>(fn->body[0].get());
    assert(ret_stmt != nullptr);

    // Should parse as: Add(a, Mul(b, c))
    auto add_expr = dynamic_cast<BinaryExprAST*>(ret_stmt->expr.get());
    assert(add_expr != nullptr);
    assert(add_expr->op == BinaryOpKind::Add);

    auto lhs_id = dynamic_cast<IdentifierExprAST*>(add_expr->lhs.get());
    assert(lhs_id != nullptr);
    assert(lhs_id->name == "a");

    auto rhs_mul = dynamic_cast<BinaryExprAST*>(add_expr->rhs.get());
    assert(rhs_mul != nullptr);
    assert(rhs_mul->op == BinaryOpKind::Mul);

    // Parenthesized: (a + b) * c
    std::string_view paren_code = R"(
fn math_paren(a: f32, b: f32, c: f32) -> f32 {
    return (a + b) * c;
}
)";
    Lexer lexer2(paren_code);
    Parser parser2(lexer2.tokenize_all());
    auto program2 = parser2.parse_program();

    assert(!parser2.has_errors());
    const auto& fn2 = program2->functions[0];
    auto ret_stmt2 = dynamic_cast<ReturnStmtAST*>(fn2->body[0].get());
    auto mul_expr = dynamic_cast<BinaryExprAST*>(ret_stmt2->expr.get());
    assert(mul_expr != nullptr);
    assert(mul_expr->op == BinaryOpKind::Mul);

    auto lhs_add = dynamic_cast<BinaryExprAST*>(mul_expr->lhs.get());
    assert(lhs_add != nullptr);
    assert(lhs_add->op == BinaryOpKind::Add);

    std::cout << "[PASS] test_parse_operator_precedence\n";
}

void test_parse_full_linear_program() {
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
    assert(program != nullptr);
    assert(program->functions.size() == 1);

    const auto& fn = program->functions[0];
    assert(fn->name == "linear");
    assert(fn->params.size() == 3);

    // Param 0: x: Tensor<f32, [2, 3]>
    assert(fn->params[0].name == "x");
    assert(fn->params[0].type->is_tensor());
    auto t0 = dynamic_cast<TensorTypeNode*>(fn->params[0].type.get());
    assert(t0 != nullptr);
    assert(t0->element_kind == ScalarKind::F32);
    assert((t0->dimensions == std::vector<int64_t>{2, 3}));

    // Param 1: w: Tensor<f32, [3, 2]>
    assert(fn->params[1].name == "w");
    auto t1 = dynamic_cast<TensorTypeNode*>(fn->params[1].type.get());
    assert((t1->dimensions == std::vector<int64_t>{3, 2}));

    // Param 2: b: Tensor<f32, [2]>
    assert(fn->params[2].name == "b");
    auto t2 = dynamic_cast<TensorTypeNode*>(fn->params[2].type.get());
    assert((t2->dimensions == std::vector<int64_t>{2}));

    // Return type: Tensor<f32, [2, 2]>
    assert(fn->return_type->is_tensor());
    auto ret_t = dynamic_cast<TensorTypeNode*>(fn->return_type.get());
    assert((ret_t->dimensions == std::vector<int64_t>{2, 2}));

    // Body statements
    assert(fn->body.size() == 3);

    // Statement 0: let z: Tensor<f32, [2, 2]> = matmul(x, w);
    auto let_z = dynamic_cast<LetStmtAST*>(fn->body[0].get());
    assert(let_z != nullptr);
    assert(let_z->var_name == "z");
    auto call_matmul = dynamic_cast<CallExprAST*>(let_z->initializer.get());
    assert(call_matmul != nullptr);
    assert(call_matmul->callee == "matmul");
    assert(call_matmul->args.size() == 2);

    // Statement 1: let y: Tensor<f32, [2, 2]> = z + b;
    auto let_y = dynamic_cast<LetStmtAST*>(fn->body[1].get());
    assert(let_y != nullptr);
    assert(let_y->var_name == "y");
    auto add_zb = dynamic_cast<BinaryExprAST*>(let_y->initializer.get());
    assert(add_zb != nullptr);
    assert(add_zb->op == BinaryOpKind::Add);

    // Statement 2: return relu(y);
    auto ret_stmt = dynamic_cast<ReturnStmtAST*>(fn->body[2].get());
    assert(ret_stmt != nullptr);
    auto call_relu = dynamic_cast<CallExprAST*>(ret_stmt->expr.get());
    assert(call_relu != nullptr);
    assert(call_relu->callee == "relu");
    assert(call_relu->args.size() == 1);

    std::cout << "[PASS] test_parse_full_linear_program\n";
}

void test_syntax_errors() {
    // Missing semicolon
    {
        std::string_view code = "fn bad() -> i32 { let x: i32 = 1 return x; }";
        Lexer lexer(code);
        Parser parser(lexer.tokenize_all());
        auto prog = parser.parse_program();
        assert(parser.has_errors());
        assert(!parser.diagnostics().empty());
    }

    // Invalid zero dimension
    {
        std::string_view code = "fn bad(x: Tensor<f32, [0, 4]>) -> f32 { return 0.0; }";
        Lexer lexer(code);
        Parser parser(lexer.tokenize_all());
        auto prog = parser.parse_program();
        assert(parser.has_errors());
    }

    // Unbalanced parentheses in expression
    {
        std::string_view code = "fn bad() -> i32 { return (1 + 2; }";
        Lexer lexer(code);
        Parser parser(lexer.tokenize_all());
        auto prog = parser.parse_program();
        assert(parser.has_errors());
    }

    std::cout << "[PASS] test_syntax_errors\n";
}

int main() {
    test_parse_scalar_function();
    test_parse_operator_precedence();
    test_parse_full_linear_program();
    test_syntax_errors();

    std::cout << "\nAll Parser tests passed successfully!\n";
    return 0;
}
