#include "keryth/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace keryth {

namespace {

std::vector<int64_t> unflatten_index(int64_t flat_idx, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    std::vector<int64_t> coords(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        coords[i] = flat_idx / strides[i];
        flat_idx %= strides[i];
    }
    return coords;
}

int64_t map_broadcast_index(const std::vector<int64_t>& out_coords,
                            const std::vector<int64_t>& in_shape,
                            const std::vector<int64_t>& in_strides) {
    if (in_shape.empty()) return 0;
    int64_t in_flat = 0;
    size_t out_rank = out_coords.size();
    size_t in_rank = in_shape.size();
    size_t rank_diff = out_rank - in_rank;

    for (size_t i = 0; i < in_rank; ++i) {
        int64_t coord = out_coords[i + rank_diff];
        if (in_shape[i] == 1) {
            coord = 0; // Broadcast dimension
        }
        in_flat += coord * in_strides[i];
    }
    return in_flat;
}

template <typename Op>
Tensor apply_binary_elementwise(const Tensor& a, const Tensor& b, Op op) {
    if (a.shape() == b.shape()) {
        std::vector<float> res_data(a.numel());
        for (int64_t i = 0; i < a.numel(); ++i) {
            res_data[i] = op(a.at_flat(i), b.at_flat(i));
        }
        return Tensor(a.shape(), std::move(res_data));
    }

    auto out_shape_opt = broadcast_shapes(a.shape(), b.shape());
    assert(out_shape_opt.has_value());
    const auto& out_shape = *out_shape_opt;

    Tensor res(out_shape);
    int64_t total = res.numel();

    for (int64_t i = 0; i < total; ++i) {
        auto coords = unflatten_index(i, res.shape(), res.strides());
        int64_t idx_a = map_broadcast_index(coords, a.shape(), a.strides());
        int64_t idx_b = map_broadcast_index(coords, b.shape(), b.strides());
        res.set_flat(i, op(a.at_flat(idx_a), b.at_flat(idx_b)));
    }

    return res;
}

} // namespace

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    return apply_binary_elementwise(a, b, [](float x, float y) { return x + y; });
}

Tensor Tensor::sub(const Tensor& a, const Tensor& b) {
    return apply_binary_elementwise(a, b, [](float x, float y) { return x - y; });
}

Tensor Tensor::mul(const Tensor& a, const Tensor& b) {
    return apply_binary_elementwise(a, b, [](float x, float y) { return x * y; });
}

Tensor Tensor::div(const Tensor& a, const Tensor& b) {
    return apply_binary_elementwise(a, b, [](float x, float y) { return x / y; });
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    assert(a.rank() == 2 && b.rank() == 2);
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    assert(b.shape()[0] == K);
    int64_t N = b.shape()[1];

    Tensor res({M, N}, 0.0f);
    const auto& a_data = a.data();
    const auto& b_data = b.data();
    auto& res_data = res.data();

    // Cache-friendly i-k-j loop ordering
    for (int64_t i = 0; i < M; ++i) {
        int64_t a_row = i * K;
        int64_t c_row = i * N;
        for (int64_t k = 0; k < K; ++k) {
            float a_ik = a_data[a_row + k];
            int64_t b_row = k * N;
            for (int64_t j = 0; j < N; ++j) {
                res_data[c_row + j] += a_ik * b_data[b_row + j];
            }
        }
    }

    return res;
}

Tensor Tensor::relu(const Tensor& a) {
    std::vector<float> res_data(a.numel());
    for (int64_t i = 0; i < a.numel(); ++i) {
        res_data[i] = std::max(0.0f, a.at_flat(i));
    }
    return Tensor(a.shape(), std::move(res_data));
}

float Tensor::sum() const {
    float s = 0.0f;
    for (float v : data_) s += v;
    return s;
}

RuntimeValue Interpreter::execute_function(const IRFunction& fn, const std::vector<RuntimeValue>& args) {
    std::unordered_map<ValueId, RuntimeValue> env;

    assert(args.size() == fn.parameters.size());
    for (size_t i = 0; i < args.size(); ++i) {
        env[fn.parameters[i]] = args[i];
    }

    for (const auto& inst : fn.instructions) {
        switch (inst.opcode) {
            case Opcode::Parameter:
                break; // Already populated from args
            case Opcode::ConstantInt:
                env[inst.result_id] = static_cast<int32_t>(std::get<int64_t>(inst.literal));
                break;
            case Opcode::ConstantFloat:
                env[inst.result_id] = static_cast<float>(std::get<double>(inst.literal));
                break;
            case Opcode::ConstantBool:
                env[inst.result_id] = std::get<bool>(inst.literal);
                break;
            case Opcode::Add: {
                const auto& v1 = env[inst.operands[0]];
                const auto& v2 = env[inst.operands[1]];
                if (std::holds_alternative<Tensor>(v1) && std::holds_alternative<Tensor>(v2)) {
                    env[inst.result_id] = Tensor::add(std::get<Tensor>(v1), std::get<Tensor>(v2));
                } else if (std::holds_alternative<float>(v1) && std::holds_alternative<float>(v2)) {
                    env[inst.result_id] = std::get<float>(v1) + std::get<float>(v2);
                } else if (std::holds_alternative<int32_t>(v1) && std::holds_alternative<int32_t>(v2)) {
                    env[inst.result_id] = std::get<int32_t>(v1) + std::get<int32_t>(v2);
                }
                break;
            }
            case Opcode::Sub: {
                const auto& v1 = env[inst.operands[0]];
                const auto& v2 = env[inst.operands[1]];
                if (std::holds_alternative<Tensor>(v1) && std::holds_alternative<Tensor>(v2)) {
                    env[inst.result_id] = Tensor::sub(std::get<Tensor>(v1), std::get<Tensor>(v2));
                } else if (std::holds_alternative<float>(v1) && std::holds_alternative<float>(v2)) {
                    env[inst.result_id] = std::get<float>(v1) - std::get<float>(v2);
                } else if (std::holds_alternative<int32_t>(v1) && std::holds_alternative<int32_t>(v2)) {
                    env[inst.result_id] = std::get<int32_t>(v1) - std::get<int32_t>(v2);
                }
                break;
            }
            case Opcode::Mul: {
                const auto& v1 = env[inst.operands[0]];
                const auto& v2 = env[inst.operands[1]];
                if (std::holds_alternative<Tensor>(v1) && std::holds_alternative<Tensor>(v2)) {
                    env[inst.result_id] = Tensor::mul(std::get<Tensor>(v1), std::get<Tensor>(v2));
                } else if (std::holds_alternative<float>(v1) && std::holds_alternative<float>(v2)) {
                    env[inst.result_id] = std::get<float>(v1) * std::get<float>(v2);
                } else if (std::holds_alternative<int32_t>(v1) && std::holds_alternative<int32_t>(v2)) {
                    env[inst.result_id] = std::get<int32_t>(v1) * std::get<int32_t>(v2);
                }
                break;
            }
            case Opcode::Div: {
                const auto& v1 = env[inst.operands[0]];
                const auto& v2 = env[inst.operands[1]];
                if (std::holds_alternative<Tensor>(v1) && std::holds_alternative<Tensor>(v2)) {
                    env[inst.result_id] = Tensor::div(std::get<Tensor>(v1), std::get<Tensor>(v2));
                } else if (std::holds_alternative<float>(v1) && std::holds_alternative<float>(v2)) {
                    env[inst.result_id] = std::get<float>(v1) / std::get<float>(v2);
                } else if (std::holds_alternative<int32_t>(v1) && std::holds_alternative<int32_t>(v2)) {
                    env[inst.result_id] = std::get<int32_t>(v1) / std::get<int32_t>(v2);
                }
                break;
            }
            case Opcode::Matmul: {
                const auto& a = std::get<Tensor>(env[inst.operands[0]]);
                const auto& b = std::get<Tensor>(env[inst.operands[1]]);
                env[inst.result_id] = Tensor::matmul(a, b);
                break;
            }
            case Opcode::Relu: {
                const auto& val = env[inst.operands[0]];
                if (std::holds_alternative<Tensor>(val)) {
                    env[inst.result_id] = Tensor::relu(std::get<Tensor>(val));
                } else if (std::holds_alternative<float>(val)) {
                    env[inst.result_id] = std::max(0.0f, std::get<float>(val));
                }
                break;
            }
            case Opcode::Sum: {
                const auto& val = env[inst.operands[0]];
                if (std::holds_alternative<Tensor>(val)) {
                    env[inst.result_id] = std::get<Tensor>(val).sum();
                } else if (std::holds_alternative<float>(val)) {
                    env[inst.result_id] = val;
                }
                break;
            }
            case Opcode::Return:
                return env[inst.operands[0]];
        }
    }

    return std::monostate{};
}

} // namespace keryth
