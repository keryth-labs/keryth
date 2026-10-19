add_executable(test_version test_version.cpp)
target_link_libraries(test_version PRIVATE keryth_core)
add_test(NAME version_unit_test COMMAND test_version)

add_executable(test_lexer test_lexer.cpp)
target_link_libraries(test_lexer PRIVATE keryth_core)
add_test(NAME lexer_unit_test COMMAND test_lexer)

add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser PRIVATE keryth_core)
add_test(NAME parser_unit_test COMMAND test_parser)

add_executable(test_type_checker test_type_checker.cpp)
target_link_libraries(test_type_checker PRIVATE keryth_core)
add_test(NAME type_checker_unit_test COMMAND test_type_checker)

add_executable(test_interpreter test_interpreter.cpp)
target_link_libraries(test_interpreter PRIVATE keryth_core)
add_test(NAME interpreter_unit_test COMMAND test_interpreter)

add_executable(test_optimizer test_optimizer.cpp)
target_link_libraries(test_optimizer PRIVATE keryth_core)
add_test(NAME optimizer_unit_test COMMAND test_optimizer)

add_executable(test_autodiff test_autodiff.cpp)
target_link_libraries(test_autodiff PRIVATE keryth_core)
add_test(NAME autodiff_unit_test COMMAND test_autodiff)

add_executable(test_memory_planner test_memory_planner.cpp)
target_link_libraries(test_memory_planner PRIVATE keryth_core)
add_test(NAME memory_planner_unit_test COMMAND test_memory_planner)

add_executable(test_llvm_codegen test_llvm_codegen.cpp)
target_link_libraries(test_llvm_codegen PRIVATE keryth_core)
add_test(NAME llvm_codegen_unit_test COMMAND test_llvm_codegen)

add_executable(test_ergonomics test_ergonomics.cpp)
target_link_libraries(test_ergonomics PRIVATE keryth_core)
add_test(NAME ergonomics_unit_test COMMAND test_ergonomics)

# Integration tests for CLI flags
add_test(NAME cli_help_long COMMAND keryth --help)
set_tests_properties(cli_help_long PROPERTIES PASS_REGULAR_EXPRESSION "Usage: .*keryth")

add_test(NAME cli_help_short COMMAND keryth -h)
set_tests_properties(cli_help_short PROPERTIES PASS_REGULAR_EXPRESSION "Usage: .*keryth")

add_test(NAME cli_version_long COMMAND keryth --version)
set_tests_properties(cli_version_long PROPERTIES PASS_REGULAR_EXPRESSION "Keryth Compiler v0\\.1\\.0")

add_test(NAME cli_version_short COMMAND keryth -v)
set_tests_properties(cli_version_short PROPERTIES PASS_REGULAR_EXPRESSION "Keryth Compiler v0\\.1\\.0")

add_test(NAME cli_unknown_arg COMMAND keryth --invalid-flag-test)
set_tests_properties(cli_unknown_arg PROPERTIES WILL_FAIL TRUE)

# Integration tests for Compiler Pipeline on example files
add_test(NAME cli_dump_ast COMMAND keryth --dump-ast ${CMAKE_SOURCE_DIR}/examples/linear.kr)
set_tests_properties(cli_dump_ast PROPERTIES PASS_REGULAR_EXPRESSION "Parsed Program with 1 function")

add_test(NAME cli_dump_ir COMMAND keryth --dump-ir ${CMAKE_SOURCE_DIR}/examples/linear.kr)
set_tests_properties(cli_dump_ir PROPERTIES PASS_REGULAR_EXPRESSION "fn @linear")

add_test(NAME cli_dump_opt_ir COMMAND keryth --dump-opt-ir ${CMAKE_SOURCE_DIR}/examples/linear.kr)
set_tests_properties(cli_dump_opt_ir PROPERTIES PASS_REGULAR_EXPRESSION "fused_matmul_bias_relu")

add_test(NAME cli_dump_memory_plan COMMAND keryth --dump-memory-plan ${CMAKE_SOURCE_DIR}/examples/linear.kr)
set_tests_properties(cli_dump_memory_plan PROPERTIES PASS_REGULAR_EXPRESSION "KERYTH STATIC MEMORY ARENA PLAN")

add_test(NAME cli_emit_llvm COMMAND keryth --emit-llvm ${CMAKE_SOURCE_DIR}/examples/linear.kr)
set_tests_properties(cli_emit_llvm PROPERTIES PASS_REGULAR_EXPRESSION "define ptr @linear")

add_test(NAME cli_run_example COMMAND keryth run ${CMAKE_SOURCE_DIR}/examples/linear.kr --opt)
set_tests_properties(cli_run_example PROPERTIES PASS_REGULAR_EXPRESSION "Result: Tensor with shape \\[2, 2\\]")

add_test(NAME cli_run_arena COMMAND keryth run ${CMAKE_SOURCE_DIR}/examples/linear.kr --arena)
set_tests_properties(cli_run_arena PROPERTIES PASS_REGULAR_EXPRESSION "zero-allocation arena")

add_test(NAME cli_run_native COMMAND keryth run ${CMAKE_SOURCE_DIR}/examples/linear.kr --native)
set_tests_properties(cli_run_native PROPERTIES PASS_REGULAR_EXPRESSION "compiled natively")

add_test(NAME cli_run_model_checkpoint COMMAND keryth run ${CMAKE_SOURCE_DIR}/examples/model_checkpoint.kr --arena)
set_tests_properties(cli_run_model_checkpoint PROPERTIES PASS_REGULAR_EXPRESSION "Result: Tensor with shape \\[2\\]")
