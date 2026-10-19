#include "keryth/llvm_codegen.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace keryth {

LLVMCodeGen::LLVMCodeGen(std::string target_triple) noexcept
    : target_triple_(std::move(target_triple)) {}

std::string LLVMCodeGen::llvm_type(const Type* type) const {
    if (!type) return "void";
    if (type->is_tensor()) {
        return "ptr"; // Modern LLVM opaque pointer
    }
    if (type->element_kind() == ScalarKind::I32) {
        return "i32";
    }
    if (type->element_kind() == ScalarKind::F32) {
        return "float";
    }
    return "i1";
}

void LLVMCodeGen::emit_instruction(const Instruction& inst, const IRFunction& fn, std::string& out) const {
    std::ostringstream ss;

    switch (inst.opcode) {
        case Opcode::Parameter:
            break;

        case Opcode::ConstantInt: {
            int64_t val = std::get<int64_t>(inst.literal);
            ss << "  %v" << inst.result_id << " = add i32 0, " << val << "\n";
            break;
        }

        case Opcode::ConstantFloat: {
            double val = std::get<double>(inst.literal);
            ss << "  %v" << inst.result_id << " = fadd float 0.0, " << val << "\n";
            break;
        }

        case Opcode::ConstantBool: {
            bool val = std::get<bool>(inst.literal);
            ss << "  %v" << inst.result_id << " = or i1 0, " << (val ? "1" : "0") << "\n";
            break;
        }

        case Opcode::ConstantTensor: {
            const auto* res_val = fn.get_value(inst.result_id);
            int64_t numel = res_val->type->num_elements();
            const auto& vec = std::get<std::vector<float>>(inst.literal);

            ss << "  ; Constant Tensor Literal\n"
               << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n";

            for (int64_t i = 0; i < numel; ++i) {
                float v = (i < static_cast<int64_t>(vec.size())) ? vec[i] : 0.0f;
                ss << "  %elem_p_" << inst.result_id << "_" << i << " = getelementptr inbounds float, ptr %v" << inst.result_id << ", i64 " << i << "\n"
                   << "  store float " << v << ", ptr %elem_p_" << inst.result_id << "_" << i << "\n";
            }
            break;
        }

        case Opcode::Transpose: {
            const auto* in_val = fn.get_value(inst.operands[0]);
            int64_t rows = in_val->type->dimensions()[0];
            int64_t cols = in_val->type->dimensions()[1];
            int64_t numel = rows * cols;

            ss << "  ; Tensor Transpose: [" << rows << ", " << cols << "] -> [" << cols << ", " << rows << "]\n"
               << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n"
               << "  call void @keryth_runtime_transpose(ptr %v" << inst.operands[0]
               << ", ptr %v" << inst.result_id << ", i64 " << rows << ", i64 " << cols << ")\n";
            break;
        }

        case Opcode::Reshape: {
            const auto* res_val = fn.get_value(inst.result_id);
            int64_t numel = res_val->type->num_elements();

            ss << "  ; Tensor Reshape\n"
               << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n"
               << "  call void @llvm.memcpy.p0.p0.i64(ptr %v" << inst.result_id
               << ", ptr %v" << inst.operands[0] << ", i64 " << numel * sizeof(float) << ", i1 false)\n";
            break;
        }

        case Opcode::LoadTensor: {
            const auto* res_val = fn.get_value(inst.result_id);
            int64_t numel = res_val->type->num_elements();

            ss << "  ; Load Tensor from Checkpoint\n"
               << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n"
               << "  call void @llvm.memset.p0.i64(ptr %v" << inst.result_id << ", i8 0, i64 " << numel * sizeof(float) << ", i1 false)\n";
            break;
        }

        case Opcode::Add: {
            const auto* res_val = fn.get_value(inst.result_id);
            if (res_val && res_val->type && res_val->type->is_tensor()) {
                const auto* val_a = fn.get_value(inst.operands[0]);
                const auto* val_b = fn.get_value(inst.operands[1]);
                int64_t numel = res_val->type->num_elements();

                ss << "  ; Tensor Add\n"
                   << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n";

                if (val_a->type->dimensions() == val_b->type->dimensions()) {
                    ss << "  call void @keryth_runtime_tensor_add(ptr %v" << inst.operands[0]
                       << ", ptr %v" << inst.operands[1] << ", ptr %v" << inst.result_id
                       << ", i64 " << numel << ")\n";
                } else {
                    int64_t m = res_val->type->dimensions()[0];
                    int64_t n = res_val->type->dimensions()[1];
                    ss << "  call void @keryth_runtime_tensor_add_broadcast(ptr %v" << inst.operands[0]
                       << ", ptr %v" << inst.operands[1] << ", ptr %v" << inst.result_id
                       << ", i64 " << m << ", i64 " << n << ")\n";
                }
            } else if (res_val && res_val->type && res_val->type->element_kind() == ScalarKind::F32) {
                ss << "  %v" << inst.result_id << " = fadd float %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            } else {
                ss << "  %v" << inst.result_id << " = add i32 %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            }
            break;
        }

        case Opcode::Sub: {
            const auto* res_val = fn.get_value(inst.result_id);
            if (res_val && res_val->type && res_val->type->element_kind() == ScalarKind::F32) {
                ss << "  %v" << inst.result_id << " = fsub float %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            } else {
                ss << "  %v" << inst.result_id << " = sub i32 %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            }
            break;
        }

        case Opcode::Mul: {
            const auto* res_val = fn.get_value(inst.result_id);
            if (res_val && res_val->type && res_val->type->element_kind() == ScalarKind::F32) {
                ss << "  %v" << inst.result_id << " = fmul float %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            } else {
                ss << "  %v" << inst.result_id << " = mul i32 %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            }
            break;
        }

        case Opcode::Div: {
            const auto* res_val = fn.get_value(inst.result_id);
            if (res_val && res_val->type && res_val->type->element_kind() == ScalarKind::F32) {
                ss << "  %v" << inst.result_id << " = fdiv float %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            } else {
                ss << "  %v" << inst.result_id << " = sdiv i32 %v" << inst.operands[0] << ", %v" << inst.operands[1] << "\n";
            }
            break;
        }

        case Opcode::Relu: {
            const auto* res_val = fn.get_value(inst.result_id);
            if (res_val && res_val->type && res_val->type->is_tensor()) {
                int64_t numel = res_val->type->num_elements();
                ss << "  ; Tensor ReLU\n"
                   << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n"
                   << "  call void @keryth_runtime_tensor_relu(ptr %v" << inst.operands[0]
                   << ", ptr %v" << inst.result_id << ", i64 " << numel << ")\n";
            } else {
                ss << "  %cmp_" << inst.result_id << " = fcmp ogt float %v" << inst.operands[0] << ", 0.0\n"
                   << "  %v" << inst.result_id << " = select i1 %cmp_" << inst.result_id << ", float %v" << inst.operands[0] << ", float 0.0\n";
            }
            break;
        }

        case Opcode::Matmul: {
            const auto* a = fn.get_value(inst.operands[0]);
            const auto* b = fn.get_value(inst.operands[1]);
            const auto* c = fn.get_value(inst.result_id);

            int64_t M = a->type->dimensions()[0];
            int64_t K = a->type->dimensions()[1];
            int64_t N = b->type->dimensions()[1];

            ss << "  ; Matmul: [" << M << ", " << K << "] x [" << K << ", " << N << "]\n"
               << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << c->type->num_elements() * sizeof(float) << ")\n"
               << "  call void @llvm.memset.p0.i64(ptr %v" << inst.result_id << ", i8 0, i64 " << c->type->num_elements() * sizeof(float) << ", i1 false)\n"
               << "  call void @keryth_runtime_matmul(ptr %v" << inst.operands[0] << ", ptr %v" << inst.operands[1]
               << ", ptr %v" << inst.result_id << ", i64 " << M << ", i64 " << K << ", i64 " << N << ")\n";
            break;
        }

        case Opcode::FusedMatmulBiasRelu: {
            const auto* a = fn.get_value(inst.operands[0]);
            const auto* b = fn.get_value(inst.operands[1]);
            const auto* c = fn.get_value(inst.result_id);

            int64_t M = a->type->dimensions()[0];
            int64_t K = a->type->dimensions()[1];
            int64_t N = b->type->dimensions()[1];

            ss << "  ; Fused Matmul + Bias + ReLU: [" << M << ", " << K << "] x [" << K << ", " << N << "]\n"
               << "  %v" << inst.result_id << " = call ptr @malloc(i64 " << c->type->num_elements() * sizeof(float) << ")\n"
               << "  call void @keryth_runtime_fused_matmul_bias_relu(ptr %v" << inst.operands[0]
               << ", ptr %v" << inst.operands[1] << ", ptr %v" << inst.operands[2]
               << ", ptr %v" << inst.result_id << ", i64 " << M << ", i64 " << K << ", i64 " << N << ")\n";
            break;
        }

        case Opcode::Sum: {
            const auto* in = fn.get_value(inst.operands[0]);
            int64_t numel = in->type->num_elements();
            ss << "  ; Tensor Sum\n"
               << "  %v" << inst.result_id << " = call float @keryth_runtime_sum(ptr %v" << inst.operands[0] << ", i64 " << numel << ")\n";
            break;
        }

        case Opcode::Return: {
            const auto* ret_val = fn.get_value(inst.operands[0]);
            ss << "  ret " << llvm_type(ret_val->type.get()) << " %v" << inst.operands[0] << "\n";
            break;
        }
    }

    out += ss.str();
}

std::string LLVMCodeGen::emit_function(const IRFunction& fn) const {
    std::ostringstream ss;
    ss << "define " << llvm_type(fn.return_type.get()) << " @" << fn.name << "(";

    for (size_t i = 0; i < fn.parameters.size(); ++i) {
        ValueId pid = fn.parameters[i];
        const auto* val = fn.get_value(pid);
        ss << llvm_type(val->type.get()) << " %v" << pid;
        if (i + 1 < fn.parameters.size()) {
            ss << ", ";
        }
    }
    ss << ") {\nentry:\n";

    std::string body;
    for (const auto& inst : fn.instructions) {
        emit_instruction(inst, fn, body);
    }
    ss << body << "}\n";
    return ss.str();
}

std::string LLVMCodeGen::emit_module(const IRModule& module) const {
    std::ostringstream ss;

    ss << "; ModuleID = 'keryth_module'\n";
    if (!target_triple_.empty()) {
        ss << "target triple = \"" << target_triple_ << "\"\n";
    }

    // Standard runtime declarations
    ss << "\n; Declarations\n"
       << "declare ptr @malloc(i64)\n"
       << "declare void @free(ptr)\n"
       << "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)\n"
       << "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n"
       << "declare i32 @printf(ptr, ...)\n\n";

    // Tensor Transpose kernel
    ss << "; Runtime Kernels\n"
       << "define void @keryth_runtime_transpose(ptr %src, ptr %dst, i64 %rows, i64 %cols) {\n"
       << "entry:\n"
       << "  br label %loop_r\n"
       << "loop_r:\n"
       << "  %r = phi i64 [ 0, %entry ], [ %next_r, %latch_r ]\n"
       << "  %r_off = mul i64 %r, %cols\n"
       << "  br label %loop_c\n"
       << "loop_c:\n"
       << "  %c = phi i64 [ 0, %loop_r ], [ %next_c, %loop_c ]\n"
       << "  %src_idx = add i64 %r_off, %c\n"
       << "  %p_src = getelementptr inbounds float, ptr %src, i64 %src_idx\n"
       << "  %val = load float, ptr %p_src\n"
       << "  %dst_row_off = mul i64 %c, %rows\n"
       << "  %dst_idx = add i64 %dst_row_off, %r\n"
       << "  %p_dst = getelementptr inbounds float, ptr %dst, i64 %dst_idx\n"
       << "  store float %val, ptr %p_dst\n"
       << "  %next_c = add i64 %c, 1\n"
       << "  %cond_c = icmp slt i64 %next_c, %cols\n"
       << "  br i1 %cond_c, label %loop_c, label %latch_r\n"
       << "latch_r:\n"
       << "  %next_r = add i64 %r, 1\n"
       << "  %cond_r = icmp slt i64 %next_r, %rows\n"
       << "  br i1 %cond_r, label %loop_r, label %exit\n"
       << "exit:\n"
       << "  ret void\n"
       << "}\n\n";

    // Tensor elementwise Add kernel
    ss << "define void @keryth_runtime_tensor_add(ptr %a, ptr %b, ptr %c, i64 %numel) {\n"
       << "entry:\n"
       << "  br label %loop\n"
       << "loop:\n"
       << "  %i = phi i64 [ 0, %entry ], [ %next_i, %loop ]\n"
       << "  %pa = getelementptr inbounds float, ptr %a, i64 %i\n"
       << "  %va = load float, ptr %pa\n"
       << "  %pb = getelementptr inbounds float, ptr %b, i64 %i\n"
       << "  %vb = load float, ptr %pb\n"
       << "  %vc = fadd float %va, %vb\n"
       << "  %pc = getelementptr inbounds float, ptr %c, i64 %i\n"
       << "  store float %vc, ptr %pc\n"
       << "  %next_i = add i64 %i, 1\n"
       << "  %cond = icmp slt i64 %next_i, %numel\n"
       << "  br i1 %cond, label %loop, label %exit\n"
       << "exit:\n"
       << "  ret void\n"
       << "}\n\n";

    // Tensor Add with rank-1 broadcasting
    ss << "define void @keryth_runtime_tensor_add_broadcast(ptr %a, ptr %b, ptr %c, i64 %M, i64 %N) {\n"
       << "entry:\n"
       << "  br label %loop_i\n"
       << "loop_i:\n"
       << "  %i = phi i64 [ 0, %entry ], [ %next_i, %latch_i ]\n"
       << "  %row_off = mul i64 %i, %N\n"
       << "  br label %loop_j\n"
       << "loop_j:\n"
       << "  %j = phi i64 [ 0, %loop_i ], [ %next_j, %loop_j ]\n"
       << "  %idx = add i64 %j, %row_off\n"
       << "  %pa = getelementptr inbounds float, ptr %a, i64 %idx\n"
       << "  %va = load float, ptr %pa\n"
       << "  %pb = getelementptr inbounds float, ptr %b, i64 %j\n"
       << "  %vb = load float, ptr %pb\n"
       << "  %vc = fadd float %va, %vb\n"
       << "  %pc = getelementptr inbounds float, ptr %c, i64 %idx\n"
       << "  store float %vc, ptr %pc\n"
       << "  %next_j = add i64 %j, 1\n"
       << "  %cond_j = icmp slt i64 %next_j, %N\n"
       << "  br i1 %cond_j, label %loop_j, label %latch_i\n"
       << "latch_i:\n"
       << "  %next_i = add i64 %i, 1\n"
       << "  %cond_i = icmp slt i64 %next_i, %M\n"
       << "  br i1 %cond_i, label %loop_i, label %exit\n"
       << "exit:\n"
       << "  ret void\n"
       << "}\n\n";

    // Tensor ReLU kernel
    ss << "define void @keryth_runtime_tensor_relu(ptr %a, ptr %c, i64 %numel) {\n"
       << "entry:\n"
       << "  br label %loop\n"
       << "loop:\n"
       << "  %i = phi i64 [ 0, %entry ], [ %next_i, %loop ]\n"
       << "  %pa = getelementptr inbounds float, ptr %a, i64 %i\n"
       << "  %va = load float, ptr %pa\n"
       << "  %cmp = fcmp ogt float %va, 0.0\n"
       << "  %vact = select i1 %cmp, float %va, float 0.0\n"
       << "  %pc = getelementptr inbounds float, ptr %c, i64 %i\n"
       << "  store float %vact, ptr %pc\n"
       << "  %next_i = add i64 %i, 1\n"
       << "  %cond = icmp slt i64 %next_i, %numel\n"
       << "  br i1 %cond, label %loop, label %exit\n"
       << "exit:\n"
       << "  ret void\n"
       << "}\n\n";

    // Matrix Multiplication kernel
    ss << "define void @keryth_runtime_matmul(ptr %a, ptr %b, ptr %c, i64 %M, i64 %K, i64 %N) {\n"
       << "entry:\n"
       << "  br label %loop_i\n"
       << "loop_i:\n"
       << "  %i = phi i64 [ 0, %entry ], [ %next_i, %latch_i ]\n"
       << "  %a_row = mul i64 %i, %K\n"
       << "  %c_row = mul i64 %i, %N\n"
       << "  br label %loop_k\n"
       << "loop_k:\n"
       << "  %k = phi i64 [ 0, %loop_i ], [ %next_k, %latch_k ]\n"
       << "  %a_idx = add i64 %k, %a_row\n"
       << "  %a_p = getelementptr inbounds float, ptr %a, i64 %a_idx\n"
       << "  %a_val = load float, ptr %a_p\n"
       << "  %b_row = mul i64 %k, %N\n"
       << "  br label %loop_j\n"
       << "loop_j:\n"
       << "  %j = phi i64 [ 0, %loop_k ], [ %next_j, %loop_j ]\n"
       << "  %b_idx = add i64 %j, %b_row\n"
       << "  %b_p = getelementptr inbounds float, ptr %b, i64 %b_idx\n"
       << "  %b_val = load float, ptr %b_p\n"
       << "  %c_idx = add i64 %j, %c_row\n"
       << "  %c_p = getelementptr inbounds float, ptr %c, i64 %c_idx\n"
       << "  %c_val = load float, ptr %c_p\n"
       << "  %prod = fmul float %a_val, %b_val\n"
       << "  %new_c = fadd float %c_val, %prod\n"
       << "  store float %new_c, ptr %c_p\n"
       << "  %next_j = add i64 %j, 1\n"
       << "  %c_j = icmp slt i64 %next_j, %N\n"
       << "  br i1 %c_j, label %loop_j, label %latch_k\n"
       << "latch_k:\n"
       << "  %next_k = add i64 %k, 1\n"
       << "  %c_k = icmp slt i64 %next_k, %K\n"
       << "  br i1 %c_k, label %loop_k, label %latch_i\n"
       << "latch_i:\n"
       << "  %next_i = add i64 %i, 1\n"
       << "  %c_i = icmp slt i64 %next_i, %M\n"
       << "  br i1 %c_i, label %loop_i, label %exit\n"
       << "exit:\n"
       << "  ret void\n"
       << "}\n\n";

    // Fused Matmul + Bias + ReLU kernel
    ss << "define void @keryth_runtime_fused_matmul_bias_relu(ptr %a, ptr %b, ptr %bias, ptr %c, i64 %M, i64 %K, i64 %N) {\n"
       << "entry:\n"
       << "  call void @keryth_runtime_matmul(ptr %a, ptr %b, ptr %c, i64 %M, i64 %K, i64 %N)\n"
       << "  br label %f_i\n"
       << "f_i:\n"
       << "  %i = phi i64 [ 0, %entry ], [ %next_i, %f_latch_i ]\n"
       << "  %c_row = mul i64 %i, %N\n"
       << "  br label %f_j\n"
       << "f_j:\n"
       << "  %j = phi i64 [ 0, %f_i ], [ %next_j, %f_j ]\n"
       << "  %c_idx = add i64 %j, %c_row\n"
       << "  %c_p = getelementptr inbounds float, ptr %c, i64 %c_idx\n"
       << "  %c_val = load float, ptr %c_p\n"
       << "  %b_p = getelementptr inbounds float, ptr %bias, i64 %j\n"
       << "  %b_val = load float, ptr %b_p\n"
       << "  %sum = fadd float %c_val, %b_val\n"
       << "  %cmp = fcmp ogt float %sum, 0.0\n"
       << "  %act = select i1 %cmp, float %sum, float 0.0\n"
       << "  store float %act, ptr %c_p\n"
       << "  %next_j = add i64 %j, 1\n"
       << "  %c_j = icmp slt i64 %next_j, %N\n"
       << "  br i1 %c_j, label %f_j, label %f_latch_i\n"
       << "f_latch_i:\n"
       << "  %next_i = add i64 %i, 1\n"
       << "  %c_i = icmp slt i64 %next_i, %M\n"
       << "  br i1 %c_i, label %f_i, label %f_exit\n"
       << "f_exit:\n"
       << "  ret void\n"
       << "}\n\n";

    // Sum reduction kernel
    ss << "define float @keryth_runtime_sum(ptr %arr, i64 %len) {\n"
       << "entry:\n"
       << "  br label %sum_loop\n"
       << "sum_loop:\n"
       << "  %idx = phi i64 [ 0, %entry ], [ %next_idx, %sum_loop ]\n"
       << "  %acc = phi float [ 0.0, %entry ], [ %new_acc, %sum_loop ]\n"
       << "  %p = getelementptr inbounds float, ptr %arr, i64 %idx\n"
       << "  %val = load float, ptr %p\n"
       << "  %new_acc = fadd float %acc, %val\n"
       << "  %next_idx = add i64 %idx, 1\n"
       << "  %cond = icmp slt i64 %next_idx, %len\n"
       << "  br i1 %cond, label %sum_loop, label %exit\n"
       << "exit:\n"
       << "  ret float %new_acc\n"
       << "}\n\n";

    // Emit all module functions
    for (const auto& fn : module.functions) {
        ss << emit_function(fn) << "\n";
    }

    return ss.str();
}

bool LLVMCodeGen::compile_to_native_binary(
    const IRModule& module,
    const std::string& output_binary_path,
    int opt_level) const {

    std::string ll_ir = emit_module(module);

    // Add standalone main entry point if not already present
    if (!module.get_function("main")) {
        std::ostringstream ss;
        ss << "\n@fmt_int = private unnamed_addr constant [12 x i8] c\"Result: %d\\0A\\00\", align 1\n"
           << "@fmt_float = private unnamed_addr constant [12 x i8] c\"Result: %f\\0A\\00\", align 1\n"
           << "@fmt_tensor = private unnamed_addr constant [35 x i8] c\"Result: Tensor computed natively!\\0A\\00\", align 1\n\n"
           << "define i32 @main() {\n"
           << "entry:\n";

        if (!module.functions.empty()) {
            const auto& fn = module.functions[0];
            // Allocate dummy inputs
            for (size_t i = 0; i < fn.parameters.size(); ++i) {
                ValueId pid = fn.parameters[i];
                const auto* val = fn.get_value(pid);
                if (val && val->type && val->type->is_tensor()) {
                    int64_t numel = val->type->num_elements();
                    ss << "  %p" << pid << " = call ptr @malloc(i64 " << numel * sizeof(float) << ")\n"
                       << "  call void @llvm.memset.p0.i64(ptr %p" << pid << ", i8 0, i64 " << numel * sizeof(float) << ", i1 false)\n";
                } else if (val && val->type && val->type->element_kind() == ScalarKind::I32) {
                    ss << "  %p" << pid << " = add i32 0, 10\n";
                } else {
                    ss << "  %p" << pid << " = fadd float 0.0, 1.0\n";
                }
            }

            ss << "  %res = call " << llvm_type(fn.return_type.get()) << " @" << fn.name << "(";
            for (size_t i = 0; i < fn.parameters.size(); ++i) {
                ValueId pid = fn.parameters[i];
                const auto* val = fn.get_value(pid);
                ss << llvm_type(val->type.get()) << " %p" << pid;
                if (i + 1 < fn.parameters.size()) ss << ", ";
            }
            ss << ")\n";

            if (fn.return_type->is_tensor()) {
                ss << "  call i32 (ptr, ...) @printf(ptr @fmt_tensor)\n";
            } else if (fn.return_type->element_kind() == ScalarKind::I32) {
                ss << "  call i32 (ptr, ...) @printf(ptr @fmt_int, i32 %res)\n";
            } else {
                ss << "  %res_dbl = fpext float %res to double\n"
                   << "  call i32 (ptr, ...) @printf(ptr @fmt_float, double %res_dbl)\n";
            }
        }

        ss << "  ret i32 0\n"
           << "}\n";

        ll_ir += ss.str();
    }

    std::string tmp_ll = "/tmp/keryth_" + std::to_string(std::rand()) + ".ll";
    {
        std::ofstream ofs(tmp_ll);
        if (!ofs.is_open()) return false;
        ofs << ll_ir;
    }

    std::string clang_cmd = "/usr/bin/clang -O" + std::to_string(opt_level) + " " +
                            tmp_ll + " -o " + output_binary_path + " 2>&1";

    int ret = std::system(clang_cmd.c_str());
    std::remove(tmp_ll.c_str());

    return (ret == 0);
}

bool LLVMCodeGen::compile_to_object_file(
    const IRModule& module,
    const std::string& output_obj_path,
    int opt_level) const {

    std::string ll_ir = emit_module(module);
    std::string tmp_ll = "/tmp/keryth_" + std::to_string(std::rand()) + ".ll";
    {
        std::ofstream ofs(tmp_ll);
        if (!ofs.is_open()) return false;
        ofs << ll_ir;
    }

    std::string clang_cmd = "/usr/bin/clang -c -O" + std::to_string(opt_level) + " " +
                            tmp_ll + " -o " + output_obj_path + " 2>&1";

    int ret = std::system(clang_cmd.c_str());
    std::remove(tmp_ll.c_str());

    return (ret == 0);
}

} // namespace keryth
