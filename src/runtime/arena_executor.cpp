#include "keryth/arena_executor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>

namespace keryth {

RuntimeValue ArenaExecutor::execute(const IRFunction& fn,
                                    const MemoryPlan& plan,
                                    const std::vector<RuntimeValue>& args) {
    // 1. Allocate the single contiguous execution arena
    std::vector<uint8_t> arena(plan.total_arena_bytes, 0);
    uint8_t* arena_base = arena.data();

    auto get_ptr = [&](ValueId vid) -> void* {
        auto it = plan.offsets.find(vid);
        assert(it != plan.offsets.end());
        return arena_base + it->second;
    };

    auto get_float_ptr = [&](ValueId vid) -> float* {
        return static_cast<float*>(get_ptr(vid));
    };

    auto get_int_ptr = [&](ValueId vid) -> int32_t* {
        return static_cast<int32_t*>(get_ptr(vid));
    };

    auto get_bool_ptr = [&](ValueId vid) -> bool* {
        return static_cast<bool*>(get_ptr(vid));
    };

    // 2. Populate function parameters into their planned arena offsets
    assert(args.size() == fn.parameters.size());
    for (size_t i = 0; i < args.size(); ++i) {
        ValueId pid = fn.parameters[i];
        const auto& arg = args[i];
        void* dst = get_ptr(pid);

        if (std::holds_alternative<Tensor>(arg)) {
            const auto& t = std::get<Tensor>(arg);
            std::memcpy(dst, t.data().data(), t.numel() * sizeof(float));
        } else if (std::holds_alternative<float>(arg)) {
            *static_cast<float*>(dst) = std::get<float>(arg);
        } else if (std::holds_alternative<int32_t>(arg)) {
            *static_cast<int32_t*>(dst) = std::get<int32_t>(arg);
        } else if (std::holds_alternative<bool>(arg)) {
            *static_cast<bool*>(dst) = std::get<bool>(arg);
        }
    }

    // 3. Execute instructions in-place within the arena (ZERO heap allocations during execution)
    ValueId return_vid = INVALID_VALUE_ID;

    for (const auto& inst : fn.instructions) {
        switch (inst.opcode) {
            case Opcode::Parameter:
                break;

            case Opcode::ConstantInt:
                *get_int_ptr(inst.result_id) = static_cast<int32_t>(std::get<int64_t>(inst.literal));
                break;

            case Opcode::ConstantFloat:
                *get_float_ptr(inst.result_id) = static_cast<float>(std::get<double>(inst.literal));
                break;

            case Opcode::ConstantBool:
                *get_bool_ptr(inst.result_id) = std::get<bool>(inst.literal);
                break;

            case Opcode::ConstantTensor: {
                const auto& vec = std::get<std::vector<float>>(inst.literal);
                float* dst = get_float_ptr(inst.result_id);
                std::memcpy(dst, vec.data(), vec.size() * sizeof(float));
                break;
            }

            case Opcode::Transpose: {
                const float* src = get_float_ptr(inst.operands[0]);
                float* dst = get_float_ptr(inst.result_id);
                const auto* val_in = fn.get_value(inst.operands[0]);
                int64_t rows = val_in->type->dimensions()[0];
                int64_t cols = val_in->type->dimensions()[1];
                for (int64_t r = 0; r < rows; ++r) {
                    for (int64_t c = 0; c < cols; ++c) {
                        dst[c * rows + r] = src[r * cols + c];
                    }
                }
                break;
            }

            case Opcode::Reshape: {
                const float* src = get_float_ptr(inst.operands[0]);
                float* dst = get_float_ptr(inst.result_id);
                const auto* val = fn.get_value(inst.result_id);
                std::memcpy(dst, src, val->type->num_elements() * sizeof(float));
                break;
            }

            case Opcode::LoadTensor: {
                std::string path = std::get<std::string>(inst.literal);
                float* dst = get_float_ptr(inst.result_id);
                const auto* val = fn.get_value(inst.result_id);
                std::ifstream ifs(path, std::ios::binary);
                if (ifs.is_open()) {
                    ifs.read(reinterpret_cast<char*>(dst), val->type->num_elements() * sizeof(float));
                }
                break;
            }

            case Opcode::Add: {
                const auto* val = fn.get_value(inst.result_id);
                if (val && val->type && val->type->is_tensor()) {
                    float* res = get_float_ptr(inst.result_id);
                    const float* a = get_float_ptr(inst.operands[0]);
                    const float* b = get_float_ptr(inst.operands[1]);
                    const auto* val_a = fn.get_value(inst.operands[0]);
                    const auto* val_b = fn.get_value(inst.operands[1]);

                    int64_t total = val->type->num_elements();
                    if (val_a->type->dimensions() == val_b->type->dimensions()) {
                        for (int64_t i = 0; i < total; ++i) {
                            res[i] = a[i] + b[i];
                        }
                    } else {
                        int64_t m = val->type->dimensions()[0];
                        int64_t n = val->type->dimensions()[1];
                        for (int64_t i = 0; i < m; ++i) {
                            for (int64_t j = 0; j < n; ++j) {
                                res[i * n + j] = a[i * n + j] + b[j];
                            }
                        }
                    }
                } else if (val && val->type && val->type->element_kind() == ScalarKind::F32) {
                    *get_float_ptr(inst.result_id) = *get_float_ptr(inst.operands[0]) + *get_float_ptr(inst.operands[1]);
                } else {
                    *get_int_ptr(inst.result_id) = *get_int_ptr(inst.operands[0]) + *get_int_ptr(inst.operands[1]);
                }
                break;
            }

            case Opcode::Sub: {
                const auto* val = fn.get_value(inst.result_id);
                if (val && val->type && val->type->is_tensor()) {
                    float* res = get_float_ptr(inst.result_id);
                    const float* a = get_float_ptr(inst.operands[0]);
                    const float* b = get_float_ptr(inst.operands[1]);
                    int64_t total = val->type->num_elements();
                    for (int64_t i = 0; i < total; ++i) {
                        res[i] = a[i] - b[i];
                    }
                } else {
                    *get_float_ptr(inst.result_id) = *get_float_ptr(inst.operands[0]) - *get_float_ptr(inst.operands[1]);
                }
                break;
            }

            case Opcode::Mul: {
                const auto* val = fn.get_value(inst.result_id);
                if (val && val->type && val->type->is_tensor()) {
                    float* res = get_float_ptr(inst.result_id);
                    const float* a = get_float_ptr(inst.operands[0]);
                    const float* b = get_float_ptr(inst.operands[1]);
                    int64_t total = val->type->num_elements();
                    for (int64_t i = 0; i < total; ++i) {
                        res[i] = a[i] * b[i];
                    }
                } else {
                    *get_float_ptr(inst.result_id) = *get_float_ptr(inst.operands[0]) * *get_float_ptr(inst.operands[1]);
                }
                break;
            }

            case Opcode::Div: {
                const auto* val = fn.get_value(inst.result_id);
                if (val && val->type && val->type->is_tensor()) {
                    float* res = get_float_ptr(inst.result_id);
                    const float* a = get_float_ptr(inst.operands[0]);
                    const float* b = get_float_ptr(inst.operands[1]);
                    int64_t total = val->type->num_elements();
                    for (int64_t i = 0; i < total; ++i) {
                        res[i] = a[i] / b[i];
                    }
                } else {
                    *get_float_ptr(inst.result_id) = *get_float_ptr(inst.operands[0]) / *get_float_ptr(inst.operands[1]);
                }
                break;
            }

            case Opcode::Matmul: {
                const auto* val_a = fn.get_value(inst.operands[0]);
                const auto* val_b = fn.get_value(inst.operands[1]);
                const auto* val_c = fn.get_value(inst.result_id);

                int64_t M = val_a->type->dimensions()[0];
                int64_t K = val_a->type->dimensions()[1];
                int64_t N = val_b->type->dimensions()[1];

                const float* a = get_float_ptr(inst.operands[0]);
                const float* b = get_float_ptr(inst.operands[1]);
                float* c = get_float_ptr(inst.result_id);

                std::memset(c, 0, val_c->type->num_elements() * sizeof(float));

                for (int64_t i = 0; i < M; ++i) {
                    int64_t a_row = i * K;
                    int64_t c_row = i * N;
                    for (int64_t k = 0; k < K; ++k) {
                        float a_ik = a[a_row + k];
                        int64_t b_row = k * N;
                        for (int64_t j = 0; j < N; ++j) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
                break;
            }

            case Opcode::Relu: {
                const auto* val = fn.get_value(inst.result_id);
                float* res = get_float_ptr(inst.result_id);
                const float* in = get_float_ptr(inst.operands[0]);
                int64_t total = val->type->num_elements();
                for (int64_t i = 0; i < total; ++i) {
                    res[i] = std::max(0.0f, in[i]);
                }
                break;
            }

            case Opcode::FusedMatmulBiasRelu: {
                const auto* val_a = fn.get_value(inst.operands[0]);
                const auto* val_b = fn.get_value(inst.operands[1]);
                const auto* val_c = fn.get_value(inst.result_id);

                int64_t M = val_a->type->dimensions()[0];
                int64_t K = val_a->type->dimensions()[1];
                int64_t N = val_b->type->dimensions()[1];

                const float* a = get_float_ptr(inst.operands[0]);
                const float* b = get_float_ptr(inst.operands[1]);
                const float* bias = get_float_ptr(inst.operands[2]);
                float* c = get_float_ptr(inst.result_id);

                std::memset(c, 0, val_c->type->num_elements() * sizeof(float));

                for (int64_t i = 0; i < M; ++i) {
                    int64_t a_row = i * K;
                    int64_t c_row = i * N;
                    for (int64_t k = 0; k < K; ++k) {
                        float a_ik = a[a_row + k];
                        int64_t b_row = k * N;
                        for (int64_t j = 0; j < N; ++j) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                    for (int64_t j = 0; j < N; ++j) {
                        float val = c[c_row + j] + bias[j];
                        c[c_row + j] = std::max(0.0f, val);
                    }
                }
                break;
            }

            case Opcode::Sum: {
                const auto* in_val = fn.get_value(inst.operands[0]);
                const float* in = get_float_ptr(inst.operands[0]);
                float s = 0.0f;
                int64_t total = in_val->type->num_elements();
                for (int64_t i = 0; i < total; ++i) {
                    s += in[i];
                }
                *get_float_ptr(inst.result_id) = s;
                break;
            }

            case Opcode::Return:
                return_vid = inst.operands[0];
                break;
        }
    }

    assert(return_vid != INVALID_VALUE_ID);
    const auto* ret_val = fn.get_value(return_vid);
    assert(ret_val && ret_val->type);

    if (ret_val->type->is_tensor()) {
        int64_t total = ret_val->type->num_elements();
        const float* src = get_float_ptr(return_vid);
        std::vector<float> data(src, src + total);
        return Tensor(ret_val->type->dimensions(), std::move(data));
    }
    if (ret_val->type->element_kind() == ScalarKind::I32) {
        return *get_int_ptr(return_vid);
    }
    if (ret_val->type->element_kind() == ScalarKind::F32) {
        return *get_float_ptr(return_vid);
    }
    return *get_bool_ptr(return_vid);
}

} // namespace keryth
