#include "keryth/type_checker.hpp"

namespace keryth {

void TypeChecker::record_error(SourceSpan span, std::string message) {
    diagnostics_.push_back(Diagnostic{
        .message = std::move(message),
        .span = span
    });
}

TypePtr TypeChecker::resolve_type(const TypeNode& node) {
    if (node.is_tensor()) {
        const auto& t_node = static_cast<const TensorTypeNode&>(node);
        return Type::make_tensor(t_node.element_kind, t_node.dimensions);
    }
    const auto& s_node = static_cast<const ScalarTypeNode&>(node);
    return Type::make_scalar(s_node.kind);
}

TypePtr TypeChecker::get_node_type(const ASTNode* node) const {
    auto it = node_types_.find(node);
    if (it != node_types_.end()) {
        return it->second;
    }
    return nullptr;
}

bool TypeChecker::check_program(const ProgramAST& program) {
    // Pass 1: Register all function signatures
    for (const auto& fn : program.functions) {
        std::vector<TypePtr> param_types;
        for (const auto& p : fn->params) {
            param_types.push_back(resolve_type(*p.type));
        }
        TypePtr ret_type = resolve_type(*fn->return_type);

        FunctionSignature sig{
            .name = fn->name,
            .param_types = std::move(param_types),
            .return_type = std::move(ret_type),
            .span = fn->span,
            .is_builtin = false
        };

        if (!symbol_table_.define_function(std::move(sig))) {
            record_error(fn->span, "Redefinition of function '" + fn->name + "'");
        }
    }

    // Pass 2: Type check each function body
    for (const auto& fn : program.functions) {
        check_function(*fn);
    }

    return !has_errors();
}

void TypeChecker::check_function(const FunctionDeclAST& fn) {
    symbol_table_.enter_scope();

    current_function_name_ = fn.name;
    current_function_return_type_ = resolve_type(*fn.return_type);

    // Define parameters in scope
    for (const auto& p : fn.params) {
        TypePtr p_type = resolve_type(*p.type);
        if (!symbol_table_.define_variable(p.name, p_type, p.span)) {
            record_error(p.span, "Duplicate parameter name '" + p.name + "' in function '" + fn.name + "'");
        }
    }

    bool has_return = false;
    for (const auto& stmt : fn.body) {
        check_statement(*stmt);
        if (dynamic_cast<const ReturnStmtAST*>(stmt.get())) {
            has_return = true;
        }
    }

    if (!has_return) {
        record_error(fn.span, "Function '" + fn.name + "' is missing a return statement");
    }

    symbol_table_.exit_scope();
}

void TypeChecker::check_statement(const StmtNode& stmt) {
    if (auto let_stmt = dynamic_cast<const LetStmtAST*>(&stmt)) {
        check_let_statement(*let_stmt);
    } else if (auto ret_stmt = dynamic_cast<const ReturnStmtAST*>(&stmt)) {
        check_return_statement(*ret_stmt);
    }
}

void TypeChecker::check_let_statement(const LetStmtAST& stmt) {
    TypePtr declared_type = resolve_type(*stmt.type_annotation);
    TypePtr init_type = check_expression(*stmt.initializer);

    if (init_type) {
        if (*declared_type != *init_type) {
            record_error(stmt.span, "Type mismatch in 'let " + stmt.var_name + "': expected " +
                         declared_type->to_string() + ", but initializer has type " + init_type->to_string());
        }
    }

    if (!symbol_table_.define_variable(stmt.var_name, declared_type, stmt.span)) {
        record_error(stmt.span, "Variable '" + stmt.var_name + "' is already defined in the current scope");
    }

    node_types_[&stmt] = declared_type;
}

void TypeChecker::check_return_statement(const ReturnStmtAST& stmt) {
    TypePtr expr_type = check_expression(*stmt.expr);
    if (!expr_type) return;

    if (*expr_type != *current_function_return_type_) {
        record_error(stmt.span, "Return type mismatch in function '" + current_function_name_ +
                     "': expected " + current_function_return_type_->to_string() +
                     ", but returned expression has type " + expr_type->to_string());
    }

    node_types_[&stmt] = expr_type;
}

TypePtr TypeChecker::check_expression(const ExprNode& expr) {
    TypePtr result = nullptr;

    if (auto id_expr = dynamic_cast<const IdentifierExprAST*>(&expr)) {
        result = check_identifier(*id_expr);
    } else if (dynamic_cast<const IntegerLiteralExprAST*>(&expr)) {
        result = Type::i32();
    } else if (dynamic_cast<const FloatLiteralExprAST*>(&expr)) {
        result = Type::f32();
    } else if (dynamic_cast<const BoolLiteralExprAST*>(&expr)) {
        result = Type::boolean();
    } else if (auto bin_expr = dynamic_cast<const BinaryExprAST*>(&expr)) {
        result = check_binary_expr(*bin_expr);
    } else if (auto call_expr = dynamic_cast<const CallExprAST*>(&expr)) {
        result = check_call_expr(*call_expr);
    }

    if (result) {
        node_types_[&expr] = result;
    }
    return result;
}

TypePtr TypeChecker::check_identifier(const IdentifierExprAST& expr) {
    TypePtr t = symbol_table_.lookup_variable(expr.name);
    if (!t) {
        record_error(expr.span, "Use of undeclared identifier '" + expr.name + "'");
        return nullptr;
    }
    return t;
}

TypePtr TypeChecker::check_binary_expr(const BinaryExprAST& expr) {
    TypePtr lhs = check_expression(*expr.lhs);
    TypePtr rhs = check_expression(*expr.rhs);
    if (!lhs || !rhs) return nullptr;

    // Both are scalars
    if (lhs->is_scalar() && rhs->is_scalar()) {
        if (lhs->element_kind() == ScalarKind::Bool || rhs->element_kind() == ScalarKind::Bool) {
            record_error(expr.span, "Binary operator '" + std::string(to_string(expr.op)) +
                         "' is not supported for boolean types");
            return nullptr;
        }
        if (lhs->element_kind() != rhs->element_kind()) {
            record_error(expr.span, "Scalar type mismatch in binary operation: " +
                         lhs->to_string() + " vs " + rhs->to_string());
            return nullptr;
        }
        return lhs;
    }

    // Both are tensors
    if (lhs->is_tensor() && rhs->is_tensor()) {
        if (lhs->element_kind() != rhs->element_kind()) {
            record_error(expr.span, "Tensor element type mismatch in binary operation: " +
                         lhs->to_string() + " vs " + rhs->to_string());
            return nullptr;
        }

        auto broadcasted = broadcast_shapes(lhs->dimensions(), rhs->dimensions());
        if (!broadcasted) {
            record_error(expr.span, "Incompatible tensor shapes for elementwise operation '" +
                         std::string(to_string(expr.op)) + "': " + lhs->to_string() +
                         " and " + rhs->to_string() + " cannot be broadcast");
            return nullptr;
        }

        return Type::make_tensor(lhs->element_kind(), *broadcasted);
    }

    // One scalar, one tensor (scalar broadcasting)
    TypePtr tensor_ty = lhs->is_tensor() ? lhs : rhs;
    TypePtr scalar_ty = lhs->is_scalar() ? lhs : rhs;

    if (tensor_ty->element_kind() != scalar_ty->element_kind()) {
        record_error(expr.span, "Type mismatch between tensor elements (" +
                     std::string(to_string(tensor_ty->element_kind())) + ") and scalar (" +
                     scalar_ty->to_string() + ")");
        return nullptr;
    }

    return tensor_ty;
}

TypePtr TypeChecker::check_call_expr(const CallExprAST& expr) {
    // Builtin: matmul(A, B)
    if (expr.callee == "matmul") {
        if (expr.args.size() != 2) {
            record_error(expr.span, "matmul expects exactly 2 arguments, got " + std::to_string(expr.args.size()));
            return nullptr;
        }
        TypePtr a = check_expression(*expr.args[0]);
        TypePtr b = check_expression(*expr.args[1]);
        if (!a || !b) return nullptr;

        if (!a->is_tensor() || !b->is_tensor()) {
            record_error(expr.span, "matmul requires tensor arguments");
            return nullptr;
        }
        if (a->rank() != 2 || b->rank() != 2) {
            record_error(expr.span, "matmul requires rank-2 (matrix) tensors, got rank " +
                         std::to_string(a->rank()) + " and " + std::to_string(b->rank()));
            return nullptr;
        }
        if (a->element_kind() != b->element_kind()) {
            record_error(expr.span, "matmul element types must match, got " +
                         a->to_string() + " and " + b->to_string());
            return nullptr;
        }

        int64_t m = a->dimensions()[0];
        int64_t k1 = a->dimensions()[1];
        int64_t k2 = b->dimensions()[0];
        int64_t n = b->dimensions()[1];

        if (k1 != k2) {
            record_error(expr.span, "matmul inner dimension mismatch: " + a->to_string() +
                         " and " + b->to_string() + " (inner dimensions " +
                         std::to_string(k1) + " != " + std::to_string(k2) + ")");
            return nullptr;
        }

        return Type::make_tensor(a->element_kind(), {m, n});
    }

    // Builtin: relu(x)
    if (expr.callee == "relu") {
        if (expr.args.size() != 1) {
            record_error(expr.span, "relu expects exactly 1 argument, got " + std::to_string(expr.args.size()));
            return nullptr;
        }
        TypePtr x = check_expression(*expr.args[0]);
        if (!x) return nullptr;

        if (x->element_kind() == ScalarKind::Bool) {
            record_error(expr.span, "relu does not support boolean tensors or scalars");
            return nullptr;
        }
        return x;
    }

    // Builtin: sum(x)
    if (expr.callee == "sum") {
        if (expr.args.size() != 1) {
            record_error(expr.span, "sum expects exactly 1 argument, got " + std::to_string(expr.args.size()));
            return nullptr;
        }
        TypePtr x = check_expression(*expr.args[0]);
        if (!x) return nullptr;

        if (!x->is_tensor()) {
            return x; // sum of scalar is scalar
        }
        return Type::make_scalar(x->element_kind());
    }

    // User-defined function call
    const auto* sig = symbol_table_.lookup_function(expr.callee);
    if (!sig) {
        record_error(expr.span, "Call to undefined function '" + expr.callee + "'");
        return nullptr;
    }

    if (expr.args.size() != sig->param_types.size()) {
        record_error(expr.span, "Function '" + expr.callee + "' expects " +
                     std::to_string(sig->param_types.size()) + " arguments, got " +
                     std::to_string(expr.args.size()));
        return nullptr;
    }

    for (size_t i = 0; i < expr.args.size(); ++i) {
        TypePtr arg_type = check_expression(*expr.args[i]);
        if (arg_type && *arg_type != *sig->param_types[i]) {
            record_error(expr.args[i]->span, "Argument " + std::to_string(i + 1) +
                         " to function '" + expr.callee + "' expects " +
                         sig->param_types[i]->to_string() + ", got " + arg_type->to_string());
        }
    }

    return sig->return_type;
}

} // namespace keryth
