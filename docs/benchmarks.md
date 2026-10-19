# Keryth Benchmark Methodology & Empirical Results

> "Write once. Optimize for the machine."

This document details the benchmarking protocol, experimental setup, and empirical findings for the Keryth compiler's operator epilogue fusion pass.

---

## 1. Experimental Setup

* **Workload**: Dense Linear Neural Network Layer with Activation
  $$Y = \text{ReLU}(XW + B)$$
* **Tensor Dimensions**:
  * $X \in \mathbb{R}^{256 \times 512}$ (131,072 elements, 512 KB)
  * $W \in \mathbb{R}^{512 \times 256}$ (131,072 elements, 512 KB)
  * $B \in \mathbb{R}^{256}$ (256 elements, 1 KB)
  * $Y \in \mathbb{R}^{256 \times 256}$ (65,536 elements, 256 KB)
* **Hardware Environment**:
  * Platform: Apple Silicon ARM64 (macOS)
  * Compiler: Apple Clang targeting C++20 with `-O3` optimizations
  * Memory: Unified memory architecture with high-bandwidth L1/L2 caches
* **Measurement Protocol**:
  * Warm-up iterations: 20 discarded passes
  * Measurement window: 100 consecutive timed executions using `std::chrono::high_resolution_clock`
  * Metric: Median execution time and allocated temporary memory bytes

---

## 2. Compared Execution Paths

### 2.1 Unfused Baseline Pipeline
The standard sequential pipeline executes operations in separate passes:
1. **Matrix Multiplication**: $T_1 = XW$
   * Allocates temporary buffer $T_1$ of size $256 \times 256 \times 4 = 262,144$ bytes (256 KB).
   * Writes 256 KB to main memory / cache.
2. **Bias Addition**: $T_2 = T_1 + B$
   * Reads 256 KB from $T_1$.
   * Allocates temporary buffer $T_2$ of size 256 KB.
   * Writes 256 KB to main memory / cache.
3. **ReLU Activation**: $Y = \text{ReLU}(T_2)$
   * Reads 256 KB from $T_2$.
   * Allocates final output buffer $Y$ of size 256 KB.
   * Writes 256 KB to main memory / cache.

**Total intermediate allocations**: 2 temporary tensors = **512 KB per iteration**.  
**Total memory traffic**: $3 \times 256 = 768$ KB written, $2 \times 256 = 512$ KB read.

### 2.2 Keryth Fused Pipeline
Keryth's `OperatorFusionPass` merges the bias addition and ReLU activation directly into the write-back stage of the matrix multiplication row loop:
1. Accumulates matrix product $XW$ in local registers per row.
2. Adds bias vector $B[j]$ directly while the accumulator is in CPU L1 cache.
3. Applies $\max(0.0, \text{val})$ in-place in registers.
4. Writes directly to final output tensor $Y$.

**Total intermediate allocations**: **0 KB**.  
**Total memory traffic**: $1 \times 256 = 256$ KB written (a $66.7\%$ reduction in write traffic).

---

## 3. Empirical Results

```
======================================================================
       KERYTH COMPILER BENCHMARK: OPERATOR EPILOGUE FUSION            
======================================================================
Workload Dimensions: M=256, K=512, N=256
Computation: Y = ReLU(X * W + B)
Precision: Contiguous Row-Major float32

[Correctness] Max absolute difference: 0 (Verified)

----------------------------------------------------------------------
Execution Path       Median Latency     Allocated Temporaries   Speedup
----------------------------------------------------------------------
Unfused Pipeline:     108.770 ms             512 KB          1.000x
Keryth Fused:          96.519 ms               0 KB          1.127x
----------------------------------------------------------------------
Result: Epilogue fusion eliminated 512 KB of temporary memory buffers
and achieved a 1.127x latency speedup on CPU.
======================================================================
```

---

## 4. Analysis and Architectural Takeaways

1. **Elimination of Allocation Jitter**: Dynamic heap allocations (`malloc`/`free`) inside inner model evaluation loops introduce non-deterministic OS memory allocator overhead. Eliminating 512 KB of temporaries per inference step ensures deterministic tail latency.
2. **Cache Locality**: On CPU architectures, memory bandwidth is the primary bottleneck for elementwise operations. Epilogue fusion transforms memory-bandwidth-bound operations (Add, ReLU) into compute-bound operations executed essentially for free during the GEMM write-back.
3. **Future Hardware Scaling**: For GPU and accelerator backends (Triton/PTX), memory bandwidth advantages become even more pronounced ($10\times$ difference between SRAM register reuse and global HBM access).
