#pragma once

#include "keryth/ir.hpp"
#include <memory>
#include <vector>

namespace keryth {

class Pass {
public:
    virtual ~Pass() = default;
    virtual bool run_on_function(IRFunction& fn) = 0;
};

class ConstantFoldingPass : public Pass {
public:
    bool run_on_function(IRFunction& fn) override;
};

class CommonSubexpressionEliminationPass : public Pass {
public:
    bool run_on_function(IRFunction& fn) override;
};

class DeadCodeEliminationPass : public Pass {
public:
    bool run_on_function(IRFunction& fn) override;
};

class OperatorFusionPass : public Pass {
public:
    bool run_on_function(IRFunction& fn) override;
};

class PassManager {
public:
    PassManager() = default;

    void add_pass(std::unique_ptr<Pass> pass) {
        passes_.push_back(std::move(pass));
    }

    void add_default_passes() {
        passes_.push_back(std::make_unique<ConstantFoldingPass>());
        passes_.push_back(std::make_unique<CommonSubexpressionEliminationPass>());
        passes_.push_back(std::make_unique<DeadCodeEliminationPass>());
        passes_.push_back(std::make_unique<OperatorFusionPass>());
        passes_.push_back(std::make_unique<DeadCodeEliminationPass>());
    }

    bool run(IRModule& module) {
        bool changed = false;
        for (auto& fn : module.functions) {
            for (const auto& pass : passes_) {
                changed |= pass->run_on_function(fn);
            }
        }
        return changed;
    }

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

} // namespace keryth
