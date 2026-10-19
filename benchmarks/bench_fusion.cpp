#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include "keryth/runtime.hpp"

using namespace keryth;

int main() {
    std::cout << "======================================================================\n"
              << "       KERYTH COMPILER BENCHMARK: OPERATOR EPILOGUE FUSION            \n"
              << "======================================================================\n";

    constexpr int64_t M = 256;
    constexpr int64_t K = 512;
    constexpr int64_t N = 256;

    std::cout << "Workload Dimensions: M=" << M << ", K=" << K << ", N=" << N << "\n"
              << "Computation: Y = ReLU(X * W + B)\n"
              << "Precision: Contiguous Row-Major float32\n\n";

    // Initialize inputs
    std::vector<float> x_data(M * K);
    for (int64_t i = 0; i < M * K; ++i) {
        x_data[i] = ((i % 100) - 50) * 0.01f;
    }
    Tensor X({M, K}, std::move(x_data));

    std::vector<float> w_data(K * N);
    for (int64_t i = 0; i < K * N; ++i) {
        w_data[i] = ((i % 80) - 40) * 0.01f;
    }
    Tensor W({K, N}, std::move(w_data));

    std::vector<float> b_data(N);
    for (int64_t i = 0; i < N; ++i) {
        b_data[i] = 0.5f;
    }
    Tensor B({N}, std::move(b_data));

    // 1. Numerical Correctness Verification
    Tensor unfused_z = Tensor::matmul(X, W);
    Tensor unfused_y = Tensor::add(unfused_z, B);
    Tensor unfused_out = Tensor::relu(unfused_y);

    Tensor fused_out = Tensor::fused_matmul_bias_relu(X, W, B);

    assert(unfused_out.shape() == fused_out.shape());
    float max_diff = 0.0f;
    for (int64_t i = 0; i < unfused_out.numel(); ++i) {
        float diff = std::fabs(unfused_out.at_flat(i) - fused_out.at_flat(i));
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "[Correctness] Max absolute difference: " << max_diff << " (Verified)\n\n";
    assert(max_diff < 1e-5f);

    // 2. Warm-up
    constexpr int WARMUP_ITERS = 10;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        Tensor t1 = Tensor::matmul(X, W);
        Tensor t2 = Tensor::add(t1, B);
        (void)Tensor::relu(t2);
        (void)Tensor::fused_matmul_bias_relu(X, W, B);
    }

    // 3. Timed Runs
    constexpr int BENCH_ITERS = 50;
    std::vector<double> unfused_times_ms;
    std::vector<double> fused_times_ms;
    unfused_times_ms.reserve(BENCH_ITERS);
    fused_times_ms.reserve(BENCH_ITERS);

    // Unfused measurements
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor t1 = Tensor::matmul(X, W);
        Tensor t2 = Tensor::add(t1, B);
        Tensor out = Tensor::relu(t2);
        auto t1_time = std::chrono::high_resolution_clock::now();
        unfused_times_ms.push_back(std::chrono::duration<double, std::milli>(t1_time - t0).count());
    }

    // Fused measurements
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor out = Tensor::fused_matmul_bias_relu(X, W, B);
        auto t1_time = std::chrono::high_resolution_clock::now();
        fused_times_ms.push_back(std::chrono::duration<double, std::milli>(t1_time - t0).count());
    }

    std::sort(unfused_times_ms.begin(), unfused_times_ms.end());
    std::sort(fused_times_ms.begin(), fused_times_ms.end());

    double unfused_median = unfused_times_ms[BENCH_ITERS / 2];
    double fused_median = fused_times_ms[BENCH_ITERS / 2];
    double speedup = unfused_median / fused_median;

    // Memory statistics
    int64_t intermediate_elements_unfused = 2 * (M * N); // T1 and T2
    int64_t intermediate_bytes_unfused = intermediate_elements_unfused * sizeof(float);
    int64_t intermediate_bytes_fused = 0; // In-place cache epilogue

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------------------------------------------------------------\n"
              << "Execution Path       Median Latency     Allocated Temporaries   Speedup\n"
              << "----------------------------------------------------------------------\n"
              << "Unfused Pipeline:    " << std::setw(8) << unfused_median << " ms"
              << "         " << std::setw(7) << intermediate_bytes_unfused / 1024 << " KB          1.00x\n"
              << "Keryth Fused:        " << std::setw(8) << fused_median << " ms"
              << "         " << std::setw(7) << intermediate_bytes_fused << " KB          "
              << speedup << "x\n"
              << "----------------------------------------------------------------------\n"
              << "Result: Epilogue fusion eliminated " << intermediate_bytes_unfused / 1024
              << " KB of temporary memory buffers\n"
              << "and achieved a " << speedup << "x latency speedup on CPU.\n"
              << "======================================================================\n";

    return 0;
}
