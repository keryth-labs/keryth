#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "keryth/types.hpp"
#include "keryth/source_location.hpp"

namespace keryth {

struct FunctionSignature {
    std::string name;
    std::vector<TypePtr> param_types;
    TypePtr return_type;
    SourceSpan span;
    bool is_builtin{false};
};

class Scope {
public:
    explicit Scope(std::shared_ptr<Scope> parent = nullptr) noexcept
        : parent_(std::move(parent)) {}

    bool define(std::string name, TypePtr type, SourceSpan span) {
        if (symbols_.find(name) != symbols_.end()) {
            return false; // Already defined in this scope
        }
        symbols_.emplace(std::move(name), SymbolInfo{std::move(type), span});
        return true;
    }

    [[nodiscard]] TypePtr lookup(std::string_view name) const {
        auto it = symbols_.find(std::string(name));
        if (it != symbols_.end()) {
            return it->second.type;
        }
        if (parent_) {
            return parent_->lookup(name);
        }
        return nullptr;
    }

    [[nodiscard]] std::shared_ptr<Scope> parent() const noexcept { return parent_; }

private:
    struct SymbolInfo {
        TypePtr type;
        SourceSpan span;
    };
    std::shared_ptr<Scope> parent_;
    std::unordered_map<std::string, SymbolInfo> symbols_;
};

class SymbolTable {
public:
    SymbolTable() {
        current_scope_ = std::make_shared<Scope>();
        register_builtins();
    }

    void enter_scope() {
        current_scope_ = std::make_shared<Scope>(current_scope_);
    }

    void exit_scope() {
        if (current_scope_->parent()) {
            current_scope_ = current_scope_->parent();
        }
    }

    bool define_variable(std::string name, TypePtr type, SourceSpan span) {
        return current_scope_->define(std::move(name), std::move(type), span);
    }

    [[nodiscard]] TypePtr lookup_variable(std::string_view name) const {
        return current_scope_->lookup(name);
    }

    bool define_function(FunctionSignature sig) {
        if (functions_.find(sig.name) != functions_.end()) {
            return false;
        }
        functions_.emplace(sig.name, std::move(sig));
        return true;
    }

    [[nodiscard]] const FunctionSignature* lookup_function(std::string_view name) const {
        auto it = functions_.find(std::string(name));
        if (it != functions_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    [[nodiscard]] bool is_builtin(std::string_view name) const {
        auto* fn = lookup_function(name);
        return fn && fn->is_builtin;
    }

private:
    void register_builtins() {
        // Builtins like matmul, relu, sum have special polymophic shape validation handled in TypeChecker
        functions_.emplace("matmul", FunctionSignature{
            .name = "matmul",
            .param_types = {},
            .return_type = nullptr,
            .span = SourceSpan{},
            .is_builtin = true
        });

        functions_.emplace("relu", FunctionSignature{
            .name = "relu",
            .param_types = {},
            .return_type = nullptr,
            .span = SourceSpan{},
            .is_builtin = true
        });

        functions_.emplace("sum", FunctionSignature{
            .name = "sum",
            .param_types = {},
            .return_type = nullptr,
            .span = SourceSpan{},
            .is_builtin = true
        });
    }

    std::shared_ptr<Scope> current_scope_;
    std::unordered_map<std::string, FunctionSignature> functions_;
};

} // namespace keryth
