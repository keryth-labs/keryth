add_library(keryth_core STATIC
    frontend/lexer.cpp
)

target_include_directories(keryth_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_executable(keryth main.cpp)
target_link_libraries(keryth PRIVATE keryth_core)

install(TARGETS keryth DESTINATION ${CMAKE_INSTALL_BINDIR})
