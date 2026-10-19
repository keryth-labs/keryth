#include "keryth/ir_builder.hpp"

namespace keryth {

ValueId ASTLowerer::allocate_value(TypePtr type, std::string name_hint, IRFunction& out_fn) {
    ValueId vid = next_value_id_++;
    out_fn.values[vid] = Value{
        .id = vid,
        .type = std::move(type),
        .name_hint = std::move(name_hint)
    };
    return vid;
}

IRModule ASTLowerer::lower_program(const ProgramAST& program) {
    IRModule module;
    for (const auto& fn : program.functions) {
        module.functions.push_back(lower_function(*fn));
    }
    return module;
}

IRFunction ASTLowerer::lower_function(const FunctionDeclAST& fn) {
    IRFunction out_fn;
    out_fn.name = fn.name;
    out_fn.return_type = type_checker_.get_node_type(fn.return_type.get());
    if (!out_fn.return_type) {
        out_fn.return_type = const_cast<TypeChecker&>(type_checker_).resolve_type(*fn.return_type);
    }

    var_to_val_.clear();
    next_value_id_ = 0;

    // Parameters
    for (const auto& p : fn.params) {
        TypePtr p_type = const_cast<TypeChecker&>(type_checker_).resolve_type(*p.type);
        ValueId pid = allocate_value(p_type, p.name, out_fn);
        out_fn.parameters.push_back(pid);
        var_to_val_[p.name] = pid;

        out_fn.instructions.push_back(Instruction{
            .result_id = pid,
            .opcode = Opcode::Parameter,
            .operands = {},
            .literal = std::monostate{},
            .type = p_type,
            .span = p.span
        });
    }

    // Statements
    for (const auto& stmt : fn.body) {
        lower_statement(*stmt, out_fn);
    }

    return out_fn;
}

void ASTLowerer::lower_statement(const StmtNode& stmt, IRFunction& out_fn) {
    if (const auto* let_stmt = dynamic_cast<const LetStmtAST*>(&stmt)) {
        ValueId init_id = lower_expression(*let_stmt->initializer, out_fn);
        var_to_val_[let_stmt->var_name] = init_id;
    } else if (const auto* ret_stmt = dynamic_cast<const ReturnStmtAST*>(&stmt)) {
        ValueId ret_id = lower_expression(*ret_stmt->expr, out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = INVALID_VALUE_ID,
            .opcode = Opcode::Return,
            .operands = {ret_id},
            .literal = std::monostate{},
            .type = out_fn.get_value(ret_id)->type,
            .span = ret_stmt->span
        });
    }
}

ValueId ASTLowerer::lower_expression(const ExprNode& expr, IRFunction& out_fn) {
    TypePtr expr_type = type_checker_.get_node_type(&expr);

    if (const auto* id_expr = dynamic_cast<const IdentifierExprAST*>(&expr)) {
        auto it = var_to_val_.find(id_expr->name);
        if (it != var_to_val_.end()) {
            return it->second;
        }
    } else if (const auto* int_expr = dynamic_cast<const IntegerLiteralExprAST*>(&expr)) {
        ValueId vid = allocate_value(expr_type ? expr_type : Type::i32(), "c_i32", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = Opcode::ConstantInt,
            .operands = {},
            .literal = int_expr->value,
            .type = expr_type ? expr_type : Type::i32(),
            .span = expr.span
        });
        return vid;
    } else if (const auto* float_expr = dynamic_cast<const FloatLiteralExprAST*>(&expr)) {
        ValueId vid = allocate_value(expr_type ? expr_type : Type::f32(), "c_f32", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = Opcode::ConstantFloat,
            .operands = {},
            .literal = float_expr->value,
            .type = expr_type ? expr_type : Type::f32(),
            .span = expr.span
        });
        return vid;
    } else if (const auto* bool_expr = dynamic_cast<const BoolLiteralExprAST*>(&expr)) {
        ValueId vid = allocate_value(expr_type ? expr_type : Type::boolean(), "c_bool", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = Opcode::ConstantBool,
            .operands = {},
            .literal = bool_expr->value,
            .type = expr_type ? expr_type : Type::boolean(),
            .span = expr.span
        });
        return vid;
    } else if (const auto* str_expr = dynamic_cast<const StringLiteralExprAST*>(&expr)) {
        ValueId vid = allocate_value(Type::i32(), "c_str", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = Opcode::ConstantInt,
            .operands = {},
            .literal = str_expr->value,
            .type = Type::i32(),
            .span = expr.span
        });
        return vid;
    } else if (const auto* tensor_lit = dynamic_cast<const TensorLiteralExprAST*>(&expr)) {
        TypePtr t_type = expr_type ? expr_type : Type::make_tensor(ScalarKind::F32, tensor_lit->shape);
        ValueId vid = allocate_value(t_type, "c_tensor", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = Opcode::ConstantTensor,
            .operands = {},
            .literal = tensor_lit->values,
            .type = t_type,
            .span = expr.span
        });
        return vid;
    } else if (const auto* bin_expr = dynamic_cast<const BinaryExprAST*>(&expr)) {
        ValueId lhs_id = lower_expression(*bin_expr->lhs, out_fn);
        ValueId rhs_id = lower_expression(*bin_expr->rhs, out_fn);

        Opcode op = Opcode::Add;
        switch (bin_expr->op) {
            case BinaryOpKind::Add: op = Opcode::Add; break;
            case BinaryOpKind::Sub: op = Opcode::Sub; break;
            case BinaryOpKind::Mul: op = Opcode::Mul; break;
            case BinaryOpKind::Div: op = Opcode::Div; break;
        }

        ValueId vid = allocate_value(expr_type, "bin_res", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = op,
            .operands = {lhs_id, rhs_id},
            .literal = std::monostate{},
            .type = expr_type,
            .span = expr.span
        });
        return vid;
    } else if (const auto* call_expr = dynamic_cast<const CallExprAST*>(&expr)) {
        if (call_expr->callee == "load_tensor" && !call_expr->args.empty()) {
            std::string path;
            if (const auto* s = dynamic_cast<const StringLiteralExprAST*>(call_expr->args[0].get())) {
                path = s->value;
            }
            ValueId vid = allocate_value(expr_type, "loaded_tensor", out_fn);
            out_fn.instructions.push_back(Instruction{
                .result_id = vid,
                .opcode = Opcode::LoadTensor,
                .operands = {},
                .literal = path,
                .type = expr_type,
                .span = expr.span
            });
            return vid;
        }

        std::vector<ValueId> arg_ids;
        for (const auto& arg : call_expr->args) {
            arg_ids.push_back(lower_expression(*arg, out_fn));
        }

        Opcode op = Opcode::Matmul;
        if (call_expr->callee == "matmul") {
            op = Opcode::Matmul;
        } else if (call_expr->callee == "relu") {
            op = Opcode::Relu;
        } else if (call_expr->callee == "sum") {
            op = Opcode::Sum;
        } else if (call_expr->callee == "transpose") {
            op = Opcode::Transpose;
        } else if (call_expr->callee == "reshape") {
            op = Opcode::Reshape;
        }

        ValueId vid = allocate_value(expr_type, "call_res", out_fn);
        out_fn.instructions.push_back(Instruction{
            .result_id = vid,
            .opcode = op,
            .operands = std::move(arg_ids),
            .literal = std::monostate{},
            .type = expr_type,
            .span = expr.span
        });
        return vid;
    }

    return INVALID_VALUE_ID;
}

} // namespace keryth
