# Keryth Language Specification (v0.1.0)

> "Write once. Optimize for the machine."

Keryth is a statically typed, tensor-aware programming language and optimizing compiler designed for predictable execution latency, zero dynamic memory overhead, and computational efficiency on tensor and machine-learning workloads.

---

## 1. Lexical Grammar

### 1.1 Whitespace and Comments

* Whitespace characters (` `, `\t`, `\r`, `\n`) separate tokens and are otherwise ignored.
* Single-line comments begin with `//` and extend to the end of the line.
* Block comments begin with `/*` and terminate with `*/`.

### 1.2 Identifiers and Keywords

* **Identifiers**: Match `[a-zA-Z_][a-zA-Z0-9_]*`.
* **Keywords**:
  `fn`, `let`, `return`, `Tensor`, `f32`, `i32`, `bool`, `true`, `false`.

### 1.3 Literals

* **Integer Literals**: Sequence of decimal digits `[0-9]+` (e.g., `0`, `42`, `1024`).
* **Floating-Point Literals**: Digits with fractional part or scientific exponent (e.g., `3.14`, `0.001`, `1e-4`, `2.5f`).
* **Boolean Literals**: `true`, `false`.
* **String Literals**: Double-quoted character sequences (e.g., `"weights/layer1.bin"`).
* **Tensor Literals**: Nested bracketed arrays of numbers (e.g., `[1.0, 2.0]` or `[[1.0, 0.0], [0.0, 1.0]]`).

### 1.4 Operators and Delimiters

* Delimiters: `(`, `)`, `[`, `]`, `{`, `}`, `<`, `>`, `:`, `;`, `,`.
* Operators: `->`, `=`, `+`, `-`, `*`, `/`.

---

## 2. Syntax Grammar (EBNF)

```ebnf
Program        ::= FunctionDecl*
FunctionDecl   ::= "fn" Identifier "(" ParameterList? ")" "->" Type "{" Statement* "}"
ParameterList  ::= Parameter ("," Parameter)*
Parameter      ::= Identifier ":" Type
Type           ::= ScalarType | TensorType
ScalarType     ::= "f32" | "i32" | "bool"
TensorType     ::= "Tensor" "<" ScalarType "," "[" DimensionList "]" ">"
DimensionList  ::= Integer ("," Integer)*

Statement      ::= LetStmt | ReturnStmt
LetStmt        ::= "let" Identifier ":" Type "=" Expression ";"
ReturnStmt     ::= "return" Expression ";"

Expression     ::= AdditiveExpr
AdditiveExpr   ::= MultiplicativeExpr ( ("+" | "-") MultiplicativeExpr )*
MultiplicativeExpr ::= PrimaryExpr ( ("*" | "/") PrimaryExpr )*

PrimaryExpr    ::= Identifier
                 | IntegerLiteral
                 | FloatLiteral
                 | BoolLiteral
                 | StringLiteral
                 | TensorLiteral
                 | CallExpr
                 | "(" Expression ")"

TensorLiteral  ::= "[" ( TensorLiteral | NumberList ) ("," ( TensorLiteral | NumberList ))* "]"
NumberList     ::= ("-")? ( FloatLiteral | IntegerLiteral ) ("," ("-")? ( FloatLiteral | IntegerLiteral ))*

CallExpr       ::= Identifier "(" (Expression ("," Expression)*)? ")"
```

---

## 3. Type System & Semantics

### 3.1 Scalar Types

* `f32`: 32-bit single-precision IEEE 754 floating-point number.
* `i32`: 32-bit signed two's complement integer.
* `bool`: Boolean truth value (`true` or `false`).

### 3.2 Fixed-Shape Tensor Types

* Tensors have static rank and compile-time integer dimension sizes: `Tensor<ScalarType, [D0, D1, ...]>`.
* All dimension sizes $D_i$ must be strictly positive integer constants ($D_i \ge 1$).

### 3.3 Static Broadcasting Rules

For binary elementwise operations ($A + B$, $A - B$, $A * B$, $A / B$):
1. Trailing dimensions are aligned from right to left.
2. Two dimensions are compatible if:
   * They are equal ($D_{1, i} == D_{2, i}$), or
   * One of them is `1` ($D_{1, i} == 1$ or $D_{2, i} == 1$).
3. Missing leading dimensions in a lower-rank tensor are prepended with `1`.
4. Scalars automatically broadcast across all dimensions of a tensor of matching scalar kind.

Example:
* $[2, 2] + [2] \implies [2, 2]$ (Rank-1 vector broadcasts across rows of Rank-2 matrix).
* $[4, 1] + [1, 5] \implies [4, 5]$ (Bidirectional expansion).
* $[2, 3] + [2, 4] \implies$ **Compile-time shape error**.

### 3.4 Matrix Multiplication (`matmul`)

* `matmul(A, B)` requires $A: \text{Tensor}\langle T, [M, K]\rangle$ and $B: \text{Tensor}\langle T, [K, N]\rangle$.
* Both arguments must have rank 2.
* Inner dimension $K$ must match exactly at compile time.
* Result type is $\text{Tensor}\langle T, [M, N]\rangle$.

### 3.5 Structural Operators

* `transpose(X)`: Swaps dimensions of a rank-2 matrix: $[M, K] \to [K, M]$.
* `reshape(X)`: View transformation preserving row-major ordering. Enforces `numel(X) == numel(target)`.
* `relu(X)`: Elementwise non-linear activation: $\max(0, x)$.
* `sum(X)`: Global reduction returning scalar of tensor element type.

### 3.6 Model Weight Ingestion (`load_tensor`)

* `load_tensor("filepath.bin")`: Ingests binary 32-bit float buffers directly from disk into static tensor allocations.
