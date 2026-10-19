#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <variant>
#include <vector>
#include "keryth/ir.hpp"

namespace keryth {

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shape, float initial_val = 0.0f)
        : shape_(std::move(shape)) {
        compute_strides();
        int64_t total = numel();
        data_.assign(total, initial_val);
    }

    Tensor(std::vector<int64_t> shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        compute_strides();
        assert(static_cast<size_t>(numel()) == data_.size());
    }

    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<int64_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] const std::vector<float>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<float>& data() noexcept { return data_; }

    [[nodiscard]] int64_t rank() const noexcept { return static_cast<int64_t>(shape_.size()); }

    [[nodiscard]] int64_t numel() const noexcept {
        if (shape_.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : shape_) n *= d;
        return n;
    }

    [[nodiscard]] float at(const std::vector<int64_t>& coords) const {
        int64_t idx = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            idx += coords[i] * strides_[i];
        }
        return data_[idx];
    }

    void set(const std::vector<int64_t>& coords, float val) {
        int64_t idx = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            idx += coords[i] * strides_[i];
        }
        data_[idx] = val;
    }

    [[nodiscard]] float at_flat(int64_t idx) const { return data_[idx]; }
    void set_flat(int64_t idx, float val) { data_[idx] = val; }

    // Tensor operations
    [[nodiscard]] static Tensor add(const Tensor& a, const Tensor& b);
    [[nodiscard]] static Tensor sub(const Tensor& a, const Tensor& b);
    [[nodiscard]] static Tensor mul(const Tensor& a, const Tensor& b);
    [[nodiscard]] static Tensor div(const Tensor& a, const Tensor& b);
    [[nodiscard]] static Tensor matmul(const Tensor& a, const Tensor& b);
    [[nodiscard]] static Tensor relu(const Tensor& a);
    [[nodiscard]] float sum() const;

private:
    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        int64_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<float> data_;
};

using RuntimeValue = std::variant<std::monostate, int32_t, float, bool, Tensor>;

class Interpreter {
public:
    Interpreter() = default;

    RuntimeValue execute_function(const IRFunction& fn, const std::vector<RuntimeValue>& args);
};

} // namespace keryth
