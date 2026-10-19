#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "keryth/ast.hpp"

namespace keryth {

class Type;
using TypePtr = std::shared_ptr<const Type>;

class Type {
public:
    virtual ~Type() = default;

    [[nodiscard]] virtual bool is_tensor() const noexcept = 0;
    [[nodiscard]] virtual bool is_scalar() const noexcept { return !is_tensor(); }

    [[nodiscard]] virtual ScalarKind element_kind() const noexcept = 0;
    [[nodiscard]] virtual const std::vector<int64_t>& dimensions() const noexcept = 0;
    [[nodiscard]] virtual int64_t rank() const noexcept = 0;
    [[nodiscard]] virtual int64_t num_elements() const noexcept = 0;

    [[nodiscard]] virtual std::string to_string() const = 0;
    [[nodiscard]] virtual bool equals(const Type& other) const noexcept = 0;

    bool operator==(const Type& other) const noexcept { return equals(other); }
    bool operator!=(const Type& other) const noexcept { return !equals(other); }

    // Static factories
    static TypePtr make_scalar(ScalarKind kind);
    static TypePtr make_tensor(ScalarKind elem_kind, std::vector<int64_t> dims);

    static TypePtr f32() { return make_scalar(ScalarKind::F32); }
    static TypePtr i32() { return make_scalar(ScalarKind::I32); }
    static TypePtr boolean() { return make_scalar(ScalarKind::Bool); }
};

class ScalarType : public Type {
public:
    explicit ScalarType(ScalarKind kind) noexcept : kind_(kind) {}

    [[nodiscard]] bool is_tensor() const noexcept override { return false; }
    [[nodiscard]] ScalarKind element_kind() const noexcept override { return kind_; }
    [[nodiscard]] const std::vector<int64_t>& dimensions() const noexcept override {
        static const std::vector<int64_t> empty_dims{};
        return empty_dims;
    }
    [[nodiscard]] int64_t rank() const noexcept override { return 0; }
    [[nodiscard]] int64_t num_elements() const noexcept override { return 1; }

    [[nodiscard]] std::string to_string() const override {
        return std::string(keryth::to_string(kind_));
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (other.is_tensor()) return false;
        return kind_ == other.element_kind();
    }

private:
    ScalarKind kind_;
};

class TensorType : public Type {
public:
    TensorType(ScalarKind elem_kind, std::vector<int64_t> dims) noexcept
        : element_kind_(elem_kind), dimensions_(std::move(dims)) {}

    [[nodiscard]] bool is_tensor() const noexcept override { return true; }
    [[nodiscard]] ScalarKind element_kind() const noexcept override { return element_kind_; }
    [[nodiscard]] const std::vector<int64_t>& dimensions() const noexcept override { return dimensions_; }
    [[nodiscard]] int64_t rank() const noexcept override { return static_cast<int64_t>(dimensions_.size()); }

    [[nodiscard]] int64_t num_elements() const noexcept override {
        if (dimensions_.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : dimensions_) {
            n *= d;
        }
        return n;
    }

    [[nodiscard]] std::string to_string() const override {
        std::string s = "Tensor<" + std::string(keryth::to_string(element_kind_)) + ", [";
        for (size_t i = 0; i < dimensions_.size(); ++i) {
            s += std::to_string(dimensions_[i]);
            if (i + 1 < dimensions_.size()) s += ", ";
        }
        s += "]>";
        return s;
    }

    [[nodiscard]] bool equals(const Type& other) const noexcept override {
        if (!other.is_tensor()) return false;
        if (element_kind_ != other.element_kind()) return false;
        return dimensions_ == other.dimensions();
    }

private:
    ScalarKind element_kind_;
    std::vector<int64_t> dimensions_;
};

inline TypePtr Type::make_scalar(ScalarKind kind) {
    return std::make_shared<const ScalarType>(kind);
}

inline TypePtr Type::make_tensor(ScalarKind elem_kind, std::vector<int64_t> dims) {
    return std::make_shared<const TensorType>(elem_kind, std::move(dims));
}

// Broadcasting utility functions
[[nodiscard]] inline std::optional<std::vector<int64_t>> broadcast_shapes(
    const std::vector<int64_t>& s1,
    const std::vector<int64_t>& s2) {

    size_t r1 = s1.size();
    size_t r2 = s2.size();
    size_t max_rank = (r1 > r2) ? r1 : r2;

    std::vector<int64_t> result(max_rank, 1);

    for (size_t i = 0; i < max_rank; ++i) {
        int64_t d1 = (i < r1) ? s1[r1 - 1 - i] : 1;
        int64_t d2 = (i < r2) ? s2[r2 - 1 - i] : 1;

        if (d1 == d2) {
            result[max_rank - 1 - i] = d1;
        } else if (d1 == 1) {
            result[max_rank - 1 - i] = d2;
        } else if (d2 == 1) {
            result[max_rank - 1 - i] = d1;
        } else {
            return std::nullopt; // Incompatible shapes
        }
    }

    return result;
}

} // namespace keryth
