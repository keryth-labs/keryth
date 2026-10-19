# Keryth

> "Write once. Optimize for the machine."

Keryth is an AI-native programming language and optimizing compiler designed from first principles for predictable execution speed, minimal memory overhead, and computational efficiency on tensor workloads.

## Vision

Modern AI frameworks like PyTorch and JAX rely on complex layers of graph capture, trace-based jit compilation, and dynamic dispatch. Keryth investigates an alternative paradigm: a clean, statically-typed, tensor-aware language where shapes, dimensions, and operational semantics are first-class language constructs verified at compile time.

## Roadmap & Milestones

- **Milestone 1 — Repository and CLI Driver**: CMake, C++20, CLI options (`--help`, `--version`), and CTest infrastructure. *(Completed)*
- **Milestone 2 — Lexer**: Deterministic tokenizer for keywords, identifiers, numeric literals, punctuation, and source tracking.
- **Milestone 3 — Parser and AST**: Recursive descent parser with precise error diagnostics and source-location spans.
- **Milestone 4 — Types and Shape Verification**: Compile-time shape inference, dimensional compatibility, and static broadcasting rules.
- **Milestone 5 — Typed SSA-IR and Interpreter**: Typed linear SSA IR, canonical verifier, and contiguous row-major tensor interpreter.
- **Milestone 6 — Optimizations**: Constant folding, dead-code elimination, common-subexpression elimination, and kernel fusion.
- **Milestone 7 — Differentiation & ML Workload**: Compile-time reverse-mode automatic differentiation and a multi-layer perceptron benchmark.
- **Milestone 8 — Native Execution**: LLVM code generation, memory planning, and fair baseline comparisons against oneDNN/Eigen.

## Building and Testing

### Prerequisites

- C++20 compliant compiler (Apple Clang 14+, GCC 11+, or Clang 13+)
- CMake 3.20+
- CTest

### Build Instructions

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### Running Tests

```bash
ctest --output-on-failure
```

### Running the Compiler

```bash
./src/keryth --help
./src/keryth --version
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
