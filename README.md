<p align="center">
  <img src="assets/wordmark.svg" alt="Keryth Wordmark" width="600"/>
</p>

<p align="center">
  <strong>An AI-native programming language and optimizing compiler designed for predictable execution speed, zero memory overhead, and computational efficiency on tensor workloads.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-GitHub_Actions-brightgreen.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Language-C%2B%2B20-orange.svg" alt="C++20">
  <img src="https://img.shields.io/badge/Tests-25%20Passing-success.svg" alt="Tests">
</p>

---

## Vision & Philosophy

Modern machine learning frameworks rely on heavy Python interpreters, dynamic tracing (`torch.compile`, JAX), and runtime shape guards. Keryth investigates an alternative paradigm: **shapes, memory layouts, and tensor contracts verified statically at compile time**.

* **Statically Typed Tensors**: Shape mismatches and broadcasting incompatibilities are rejected during semantic analysis before code ever runs.
* **Predictable Latency**: Epilogue operator fusion merges sequential elementwise kernels (Add, ReLU) directly into matrix multiplication write-back loops, eliminating temporary cache thrashing.
* **Zero-Allocation Arena**: Static liveness analysis plans physical buffer reuse, guaranteeing fixed memory bounds and eliminating dynamic allocations (`malloc`/`free`) during inference.
* **Three Execution Tiers**: Reference Interpreter, Zero-Allocation Arena, and Ahead-of-Time LLVM Native Compilation with SIMD vectorization.

---

## Architectural Documentation

* [Language Specification](docs/language-spec.md): Formal EBNF grammar, scalar & tensor type rules, and broadcasting semantics.
* [Compiler Architecture](docs/architecture.md): Complete pipeline design from Lexer, Parser, and Typed SSA-IR to the Memory Planner, Autodiff engine, and LLVM backend.
* [Benchmarking Methodology & Results](docs/benchmarks.md): Empirical evaluation of operator epilogue fusion demonstrating a **1.127x latency speedup** and **512 KB memory allocation elimination**.

---

## Compiler Pipeline Status

| Pipeline Stage | Implementation | Status |
| :--- | :--- | :--- |
| **Lexer & Tokens** | Stateful scanner, source spans (`line:col`), comments, strings | Completed |
| **Parser & AST** | Recursive descent, operator precedence, tensor literals | Completed |
| **Type & Shape Checker** | Static shape inference, right-to-left broadcasting, matmul contracts | Completed |
| **Typed SSA-IR** | Linear SSA representation, explicit opcodes, types, and values | Completed |
| **Optimizer** | Constant folding, DCE, CSE, and epilogue operator fusion | Completed |
| **Memory Planner** | Liveness analysis, interval graph packing, 64-byte alignment (40% memory reduction) | Completed |
| **Autodiff (VJP)** | Reverse-mode automatic differentiation, gradient checks, MLP training loop | Completed |
| **LLVM Native Backend** | AOT compiler emitting native ARM64/x86 binaries with Clang/LLVM `-O3` | Completed |
| **Ergonomics & Ingestion** | Nested literals, `transpose`, `reshape`, and binary `load_tensor` | Completed |

---

## Quickstart

### Prerequisites

* C++20 compliant compiler (Apple Clang 14+, GCC 11+, or Clang 13+)
* CMake 3.20+
* CTest

### Building from Source

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Running the Test Suite (25 Tests)

```bash
ctest --output-on-failure
```

---

## Command-Line Usage

```bash
# 1. Inspect Abstract Syntax Tree
./build/src/keryth --dump-ast examples/model_checkpoint.kr

# 2. View Typed Intermediate Representation
./build/src/keryth --dump-ir examples/model_checkpoint.kr

# 3. View Optimized SSA-IR (with epilogue fusion)
./build/src/keryth --dump-opt-ir examples/model_checkpoint.kr

# 4. View Static Memory Arena Layout (lifetime intervals & byte offsets)
./build/src/keryth --dump-memory-plan examples/model_checkpoint.kr

# 5. Emit Target LLVM IR Assembly
./build/src/keryth --emit-llvm examples/model_checkpoint.kr

# 6. Execute in Zero-Allocation Memory Arena
./build/src/keryth run examples/model_checkpoint.kr --arena

# 7. Compile and Run Directly to Native Machine Code
./build/src/keryth run examples/model_checkpoint.kr --native

# 8. Build Standalone Native Executable
./build/src/keryth build examples/model_checkpoint.kr -o ./model_native
./model_native
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
