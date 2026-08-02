cmake_minimum_required(VERSION 3.24)

project(ALSR_PoC
  VERSION 0.1.0
  DESCRIPTION "Adaptive Layer Streaming Runtime - Proof of Concept (Phase 0-2)"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# 言語・ビルド既定
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALSR_BUILD_TESTS   "Build unit/integration/stress tests" ON)
option(ALSR_WERROR        "Treat warnings as errors"            ON)
option(ALSR_ENABLE_HW_CRC "Use hardware CRC32C when available"  ON)

# ---------------------------------------------------------------------------
# コアライブラリ
#   NFR-04: コアロジックは GPU API から分離する。backends/ 配下のみが
#   プラットフォーム固有コードを持つ。
# ---------------------------------------------------------------------------
add_library(alsr_core STATIC
  src/telemetry/json.cpp
  src/telemetry/checksum.cpp
  src/telemetry/telemetry.cpp
  src/telemetry/env.cpp
  src/inspector/ggml_types.cpp
  src/inspector/gguf.cpp
  src/inspector/manifest.cpp
  src/planner/planner.cpp
  src/buffer/host_store.cpp
  src/buffer/ring_buffer.cpp
  src/scheduler/scheduler.cpp
  src/sim/bandwidth.cpp
  src/sim/sim.cpp
  backends/simulator/sim_transfer_backend.cpp
)

target_include_directories(alsr_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_definitions(alsr_core PUBLIC
  ALSR_VERSION_STRING="${PROJECT_VERSION}")

if(ALSR_ENABLE_HW_CRC)
  target_compile_definitions(alsr_core PRIVATE ALSR_ENABLE_HW_CRC=1)
endif()

if(MSVC)
  target_compile_options(alsr_core PRIVATE /W4)
  if(ALSR_WERROR)
    target_compile_options(alsr_core PRIVATE /WX)
  endif()
else()
  target_compile_options(alsr_core PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
  if(ALSR_WERROR)
    target_compile_options(alsr_core PRIVATE -Werror)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(alsr_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# CUDA backend (Phase 3)
#   本 Mission では実装しない。ディレクトリと README のみ用意する。
# ---------------------------------------------------------------------------
# add_subdirectory(backends/cuda)  # Phase 3 で有効化

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
foreach(tool inspect plan sim bw)
  add_executable(alsr-${tool} src/cli/alsr_${tool}.cpp)
  target_link_libraries(alsr-${tool} PRIVATE alsr_core)
  if(NOT MSVC)
    target_compile_options(alsr-${tool} PRIVATE -Wall -Wextra -Wpedantic)
    if(ALSR_WERROR)
      target_compile_options(alsr-${tool} PRIVATE -Werror)
    endif()
  endif()
endforeach()

# ---------------------------------------------------------------------------
# テスト
#   設計書 §9.1「テストフレームワークは1つに固定」に従い Catch2 v3 のみ。
#   再現性のため GIT_TAG は annotated tag の commit SHA で固定する。
# ---------------------------------------------------------------------------
if(ALSR_BUILD_TESTS)
  include(FetchContent)
  set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/_fetch"
      CACHE PATH "" FORCE)

  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    # v3.15.3
    GIT_TAG        8b08d4d79514f45f7e4ce2a607ac9c94e920d1bb
    GIT_SHALLOW    OFF
    SYSTEM)
  FetchContent_MakeAvailable(Catch2)

  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# 設定サマリー
# ---------------------------------------------------------------------------
message(STATUS "ALSR PoC ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  tests      : ${ALSR_BUILD_TESTS} (Catch2 v3.15.3 pinned)")
message(STATUS "  hw crc32c  : ${ALSR_ENABLE_HW_CRC}")
