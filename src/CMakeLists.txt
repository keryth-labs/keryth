add_library(keryth_core STATIC
    frontend/lexer.cpp
    frontend/parser.cpp
    frontend/type_checker.cpp
    ir/ir.cpp
    ir/lower.cpp
    runtime/interpreter.cpp
    runtime/arena_executor.cpp
    optimizer/optimizer.cpp
    optimizer/memory_planner.cpp
)

target_include_directories(keryth_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_executable(keryth main.cpp)
target_link_libraries(keryth PRIVATE keryth_core)

install(TARGETS keryth DESTINATION ${CMAKE_INSTALL_BINDIR})
