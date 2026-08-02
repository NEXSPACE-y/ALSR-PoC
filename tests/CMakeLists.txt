# テストは 1 つの実行ファイルにまとめ、Catch2 のタグで層を分ける。
#   [unit] [integration] [stress]
add_executable(alsr_tests
  unit/test_json.cpp
  unit/test_checksum.cpp
  unit/test_ggml_types.cpp
  unit/test_gguf.cpp
  unit/test_planner.cpp
  unit/test_ring_buffer.cpp
  unit/test_scheduler.cpp
  unit/test_telemetry.cpp
  integration/test_pipeline.cpp
  stress/test_stress.cpp
)

target_link_libraries(alsr_tests PRIVATE alsr_core Catch2::Catch2WithMain)
target_include_directories(alsr_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT MSVC)
  target_compile_options(alsr_tests PRIVATE -Wall -Wextra -Wpedantic)
  if(ALSR_WERROR)
    target_compile_options(alsr_tests PRIVATE -Werror)
  endif()
endif()

include(Catch)
catch_discover_tests(alsr_tests)
