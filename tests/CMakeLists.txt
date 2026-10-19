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
