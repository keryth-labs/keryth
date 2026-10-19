#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "keryth/source_location.hpp"

namespace keryth {

// Forward declarations
class ASTVisitor;

// Base AST Node
struct ASTNode {
    SourceSpan span;
    explicit ASTNode(SourceSpan s) noexcept : span(s) {}
    virtual ~ASTNode() = default;

    virtual void accept(ASTVisitor& visitor) const = 0;
};

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

enum class ScalarKind {
    F32,
    I32,
    Bool
};

[[nodiscard]] constexpr std::string_view to_string(ScalarKind kind) noexcept {
    switch (kind) {
        case ScalarKind::F32:  return "f32";
        case ScalarKind::I32:  return "i32";
        case ScalarKind::Bool: return "bool";
    }
    return "<unknown>";
}

struct TypeNode : public ASTNode {
    using ASTNode::ASTNode;
    [[nodiscard]] virtual bool is_tensor() const noexcept = 0;
    [[nodiscard]] virtual std::string to_string() const = 0;
};

struct ScalarTypeNode : public TypeNode {
    ScalarKind kind;

    ScalarTypeNode(ScalarKind k, SourceSpan s) noexcept
        : TypeNode(s), kind(k) {}

    [[nodiscard]] bool is_tensor() const noexcept override { return false; }
    [[nodiscard]] std::string to_string() const override {
        return std::string(keryth::to_string(kind));
    }

    void accept(ASTVisitor& visitor) const override;
};

struct TensorTypeNode : public TypeNode {
    ScalarKind element_kind;
    std::vector<int64_t> dimensions;

    TensorTypeNode(ScalarKind elem_k, std::vector<int64_t> dims, SourceSpan s) noexcept
        : TypeNode(s), element_kind(elem_k), dimensions(std::move(dims)) {}

    [[nodiscard]] bool is_tensor() const noexcept override { return true; }
    [[nodiscard]] std::string to_string() const override {
        std::string s = "Tensor<" + std::string(keryth::to_string(element_kind)) + ", [";
        for (size_t i = 0; i < dimensions.size(); ++i) {
            s += std::to_string(dimensions[i]);
            if (i + 1 < dimensions.size()) s += ", ";
        }
        s += "]>";
        return s;
    }

    void accept(ASTVisitor& visitor) const override;
};

// -----------------------------------------------------------------------------
// Expressions
// -----------------------------------------------------------------------------

struct ExprNode : public ASTNode {
    using ASTNode::ASTNode;
};

struct IdentifierExprAST : public ExprNode {
    std::string name;

    IdentifierExprAST(std::string n, SourceSpan s) noexcept
        : ExprNode(s), name(std::move(n)) {}

    void accept(ASTVisitor& visitor) const override;
};

struct IntegerLiteralExprAST : public ExprNode {
    int64_t value;

    IntegerLiteralExprAST(int64_t v, SourceSpan s) noexcept
        : ExprNode(s), value(v) {}

    void accept(ASTVisitor& visitor) const override;
};

struct FloatLiteralExprAST : public ExprNode {
    double value;

    FloatLiteralExprAST(double v, SourceSpan s) noexcept
        : ExprNode(s), value(v) {}

    void accept(ASTVisitor& visitor) const override;
};

struct BoolLiteralExprAST : public ExprNode {
    bool value;

    BoolLiteralExprAST(bool v, SourceSpan s) noexcept
        : ExprNode(s), value(v) {}

    void accept(ASTVisitor& visitor) const override;
};

struct StringLiteralExprAST : public ExprNode {
    std::string value;

    StringLiteralExprAST(std::string v, SourceSpan s) noexcept
        : ExprNode(s), value(std::move(v)) {}

    void accept(ASTVisitor& visitor) const override;
};

struct TensorLiteralExprAST : public ExprNode {
    std::vector<float> values;
    std::vector<int64_t> shape;

    TensorLiteralExprAST(std::vector<float> v, std::vector<int64_t> shp, SourceSpan s) noexcept
        : ExprNode(s), values(std::move(v)), shape(std::move(shp)) {}

    void accept(ASTVisitor& visitor) const override;
};

enum class BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div
};

[[nodiscard]] constexpr std::string_view to_string(BinaryOpKind op) noexcept {
    switch (op) {
        case BinaryOpKind::Add: return "+";
        case BinaryOpKind::Sub: return "-";
        case BinaryOpKind::Mul: return "*";
        case BinaryOpKind::Div: return "/";
    }
    return "?";
}

struct BinaryExprAST : public ExprNode {
    BinaryOpKind op;
    std::unique_ptr<ExprNode> lhs;
    std::unique_ptr<ExprNode> rhs;

    BinaryExprAST(BinaryOpKind o, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r, SourceSpan s) noexcept
        : ExprNode(s), op(o), lhs(std::move(l)), rhs(std::move(r)) {}

    void accept(ASTVisitor& visitor) const override;
};

struct CallExprAST : public ExprNode {
    std::string callee;
    std::vector<std::unique_ptr<ExprNode>> args;

    CallExprAST(std::string c, std::vector<std::unique_ptr<ExprNode>> a, SourceSpan s) noexcept
        : ExprNode(s), callee(std::move(c)), args(std::move(a)) {}

    void accept(ASTVisitor& visitor) const override;
};

// -----------------------------------------------------------------------------
// Statements
// -----------------------------------------------------------------------------

struct StmtNode : public ASTNode {
    using ASTNode::ASTNode;
};

struct LetStmtAST : public StmtNode {
    std::string var_name;
    std::unique_ptr<TypeNode> type_annotation;
    std::unique_ptr<ExprNode> initializer;

    LetStmtAST(std::string n, std::unique_ptr<TypeNode> ty, std::unique_ptr<ExprNode> init, SourceSpan s) noexcept
        : StmtNode(s), var_name(std::move(n)), type_annotation(std::move(ty)), initializer(std::move(init)) {}

    void accept(ASTVisitor& visitor) const override;
};

struct ReturnStmtAST : public StmtNode {
    std::unique_ptr<ExprNode> expr;

    ReturnStmtAST(std::unique_ptr<ExprNode> e, SourceSpan s) noexcept
        : StmtNode(s), expr(std::move(e)) {}

    void accept(ASTVisitor& visitor) const override;
};

// -----------------------------------------------------------------------------
// Functions and Program
// -----------------------------------------------------------------------------

struct ParamDecl {
    std::string name;
    std::unique_ptr<TypeNode> type;
    SourceSpan span;
};

struct FunctionDeclAST : public ASTNode {
    std::string name;
    std::vector<ParamDecl> params;
    std::unique_ptr<TypeNode> return_type;
    std::vector<std::unique_ptr<StmtNode>> body;

    FunctionDeclAST(std::string n,
                    std::vector<ParamDecl> p,
                    std::unique_ptr<TypeNode> ret,
                    std::vector<std::unique_ptr<StmtNode>> b,
                    SourceSpan s) noexcept
        : ASTNode(s), name(std::move(n)), params(std::move(p)), return_type(std::move(ret)), body(std::move(b)) {}

    void accept(ASTVisitor& visitor) const override;
};

struct ProgramAST : public ASTNode {
    std::vector<std::unique_ptr<FunctionDeclAST>> functions;

    ProgramAST(std::vector<std::unique_ptr<FunctionDeclAST>> funcs, SourceSpan s) noexcept
        : ASTNode(s), functions(std::move(funcs)) {}

    void accept(ASTVisitor& visitor) const override;
};

// -----------------------------------------------------------------------------
// Visitor Interface
// -----------------------------------------------------------------------------

class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;

    virtual void visit(const ScalarTypeNode& node) = 0;
    virtual void visit(const TensorTypeNode& node) = 0;

    virtual void visit(const IdentifierExprAST& node) = 0;
    virtual void visit(const IntegerLiteralExprAST& node) = 0;
    virtual void visit(const FloatLiteralExprAST& node) = 0;
    virtual void visit(const BoolLiteralExprAST& node) = 0;
    virtual void visit(const StringLiteralExprAST& node) = 0;
    virtual void visit(const TensorLiteralExprAST& node) = 0;
    virtual void visit(const BinaryExprAST& node) = 0;
    virtual void visit(const CallExprAST& node) = 0;

    virtual void visit(const LetStmtAST& node) = 0;
    virtual void visit(const ReturnStmtAST& node) = 0;

    virtual void visit(const FunctionDeclAST& node) = 0;
    virtual void visit(const ProgramAST& node) = 0;
};

inline void ScalarTypeNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void TensorTypeNode::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void IdentifierExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void IntegerLiteralExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void FloatLiteralExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void BoolLiteralExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void StringLiteralExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void TensorLiteralExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void BinaryExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void CallExprAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void LetStmtAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void ReturnStmtAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void FunctionDeclAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }
inline void ProgramAST::accept(ASTVisitor& visitor) const { visitor.visit(*this); }

} // namespace keryth
