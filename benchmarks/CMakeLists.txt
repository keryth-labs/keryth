add_executable(bench_fusion bench_fusion.cpp)
target_link_libraries(bench_fusion PRIVATE keryth_core)

add_test(NAME bench_fusion_test COMMAND bench_fusion)
