# Keryth Compiler Architecture

> "Write once. Optimize for the machine."

Keryth is engineered from first principles as an AI-native optimizing compiler. It eliminates runtime tracing, dynamic shape guards, and unmanaged garbage collection by validating tensor shapes, lifetimes, and operations statically at compile time.

---

## 1. End-to-End Pipeline Overview

```
                      Keryth Source (.kr)
                              │
                              ▼
                 [ Lexer & Source Manager ]
                 Token stream + SourceSpan
                              │
                              ▼
                [ Recursive Descent Parser ]
                       Abstract Syntax Tree
                              │
                              ▼
                [ Type & Shape Checker ]
           Symbol resolution & broadcasting solver
                              │
                              ▼
                 [ AST-to-IR Lowerer ]
              Linear Typed SSA Intermediate Rep
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
    [ Optimization Passes ]           [ Reverse-Mode Autodiff ]
    - Constant Folding                - VJP backward synthesis
    - CSE & DCE                       - Gradient table generation
    - Epilogue Fusion                          │
             │                                 ▼
             ├─────────────────────────────────┘
             │
             ▼
    [ Static Memory Planner ]
    - Liveness interval analysis
    - Interval graph arena packing
    - 64-byte SIMD alignment
             │
   ┌─────────┼─────────────────────────┐
   ▼         ▼                         ▼
Tier 1:   Tier 2:                   Tier 3:
Reference Zero-Allocation           LLVM Native Backend
Interp    Memory Arena Executor     Ahead-of-Time Binary (-O3)
```

---

## 2. Frontend Subsystems

### 2.1 Lexer & Source Manager (`src/frontend/lexer.cpp`)
* **Stateful scanning**: Ingests UTF-8 source text and tracks 1-indexed line and column coordinates.
* **Deterministic tokens**: Distinguishes keywords, identifiers, scalar numbers, strings, and multi-character operators (`->`, `==`).
* **SourceSpan tracking**: Every token and AST node preserves exact start and end coordinates for precise diagnostics.

### 2.2 Parser & AST (`src/frontend/parser.cpp`, `include/keryth/ast.hpp`)
* **Hand-crafted Recursive Descent**: Zero external parser dependencies.
* **Operator Precedence**: Multiplicative operators (`*`, `/`) bind tighter than additive operators (`+`, `-`).
* **Tensor Literals**: Parses nested bracketed lists (`[[1.0, 0.0], [0.0, 1.0]]`) with dimension uniformity validation.
* **Diagnostics**: Non-fatal syntax error recording with synchronization at statement boundaries.

### 2.3 Type & Shape Checker (`src/frontend/type_checker.cpp`)
* **Two-Pass Symbol Resolution**: Pass 1 registers all top-level signatures to support forward references; Pass 2 validates function bodies.
* **Static Broadcasting Solver**: Implements trailing right-to-left dimension alignment. Automatically validates dimensional expansions ($[2, 2] + [2] \to [2, 2]$) and rejects incompatible shapes at compile time.
* **Matrix Multiplication Contracts**: Enforces rank-2 tensors and inner dimension equality ($K_A == K_B$).

---

## 3. Intermediate Representation (`include/keryth/ir.hpp`)

Keryth uses a linear, typed Single Static Assignment (SSA) IR:

```
fn @linear(%0: Tensor<f32, [2, 3]>, %1: Tensor<f32, [3, 2]>, %2: Tensor<f32, [2]>) -> Tensor<f32, [2, 2]> {
  %0: Tensor<f32, [2, 3]> = param
  %1: Tensor<f32, [3, 2]> = param
  %2: Tensor<f32, [2]> = param
  %3: Tensor<f32, [2, 2]> = matmul %0, %1
  %4: Tensor<f32, [2, 2]> = add %3, %2
  %5: Tensor<f32, [2, 2]> = relu %4
  return %5
}
```

Every value `%id` is assigned exactly once and carries:
1. Immutable static type and shape metadata.
2. Generating opcode and operand value dependencies.
3. Original source location span for auditability.

---

## 4. Optimization Pipeline (`src/optimizer/optimizer.cpp`)

1. **Constant Folding**: Evaluates compile-time scalar arithmetic and identity transforms.
2. **Common Subexpression Elimination (CSE)**: Value numbering collapses duplicate subtrees to single executions.
3. **Dead Code Elimination (DCE)**: Reverse worklist pruning eliminates unused calculations.
4. **Epilogue Operator Fusion**: Identifies sequences of `Matmul` $\to$ `Add` $\to$ `ReLU` and fuses them into a single `fused_matmul_bias_relu` operation, eliminating intermediate buffer allocations.

---

## 5. Static Memory Planner (`src/optimizer/memory_planner.cpp`)

Because tensor dimensions are static, the compiler calculates the exact lifetime and byte size of every intermediate tensor:

1. **Liveness Analysis**: Computes `[first_def, last_use]` instruction spans for every SSA value.
2. **Interval Graph Packing**: Greedy first-fit memory packing assigns memory offsets with 64-byte alignment.
3. **Zero-Allocation Execution**: Non-overlapping values share physical memory offsets, reducing peak memory footprints by up to 40% with zero dynamic heap allocations during inference.

---

## 6. Execution Tiers

1. **Reference Interpreter (`src/runtime/interpreter.cpp`)**: Verifies mathematical semantics and provides debugger-friendly execution.
2. **Zero-Allocation Arena (`src/runtime/arena_executor.cpp`)**: Executes directly on raw arena pointers with zero `malloc`/`free` calls.
3. **LLVM Native Backend (`src/backend/llvm_codegen.cpp`)**: Emits optimized LLVM IR (`.ll`), compiling to native machine code binaries with hardware SIMD vectorization.
