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
option(ALSR_ENABLE_M1_LLAMA_TRACE
       "Build the M1 real llama.cpp graph trace tool" OFF)
option(ALSR_ENABLE_M2_EXECUTOR
       "Build the M2 M5 real Executor with the ALSR scheduler hook" OFF)
set(ALSR_LLAMA_CPP_SOURCE_DIR "" CACHE PATH
    "Exact b10091 llama.cpp source tree for the M1 trace tool")
set(ALSR_LLAMA_CPP_PATCH_FILE "" CACHE FILEPATH
    "Canonical ALSR patch applied to the pinned llama.cpp tree (M2 only)")

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
  src/inspector/real_graph_manifest.cpp
  src/planner/planner.cpp
  src/buffer/host_store.cpp
  src/buffer/ring_buffer.cpp
  src/supply/real_supply.cpp
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

add_executable(alsr-real-manifest src/cli/alsr_real_manifest.cpp)
target_link_libraries(alsr-real-manifest PRIVATE alsr_core)
if(NOT MSVC)
  target_compile_options(alsr-real-manifest PRIVATE -Wall -Wextra -Wpedantic)
  if(ALSR_WERROR)
    target_compile_options(alsr-real-manifest PRIVATE -Werror)
  endif()
endif()

if(ALSR_ENABLE_M1_LLAMA_TRACE OR ALSR_ENABLE_M2_EXECUTOR)
  if(NOT IS_DIRECTORY "${ALSR_LLAMA_CPP_SOURCE_DIR}")
    message(FATAL_ERROR
      "ALSR_ENABLE_M1_LLAMA_TRACE requires ALSR_LLAMA_CPP_SOURCE_DIR")
  endif()
  execute_process(
    COMMAND git -C "${ALSR_LLAMA_CPP_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE ALSR_LLAMA_CPP_ACTUAL_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ALSR_LLAMA_CPP_GIT_RESULT)
  set(ALSR_LLAMA_CPP_REQUIRED_COMMIT
      "b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45")
  if(NOT ALSR_LLAMA_CPP_GIT_RESULT EQUAL 0 OR
     NOT ALSR_LLAMA_CPP_ACTUAL_COMMIT STREQUAL ALSR_LLAMA_CPP_REQUIRED_COMMIT)
    message(FATAL_ERROR
      "M1 requires llama.cpp ${ALSR_LLAMA_CPP_REQUIRED_COMMIT}; got ${ALSR_LLAMA_CPP_ACTUAL_COMMIT}")
  endif()

  set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
  if(ALSR_ENABLE_M2_EXECUTOR)
    if(NOT EXISTS "${ALSR_LLAMA_CPP_PATCH_FILE}")
      message(FATAL_ERROR
        "M2 requires ALSR_LLAMA_CPP_PATCH_FILE")
    endif()
    file(SHA256 "${ALSR_LLAMA_CPP_PATCH_FILE}"
         ALSR_LLAMA_CPP_PATCH_SHA256)
    set(ALSR_LLAMA_CPP_CURRENT_PATCH
        "${CMAKE_BINARY_DIR}/llama.cpp-current.patch")
    execute_process(
      COMMAND git -C "${ALSR_LLAMA_CPP_SOURCE_DIR}" diff HEAD --binary
      OUTPUT_FILE "${ALSR_LLAMA_CPP_CURRENT_PATCH}"
      RESULT_VARIABLE ALSR_LLAMA_CPP_DIFF_RESULT)
    if(NOT ALSR_LLAMA_CPP_DIFF_RESULT EQUAL 0)
      message(FATAL_ERROR "could not fingerprint the M2 llama.cpp patch")
    endif()
    file(SHA256 "${ALSR_LLAMA_CPP_CURRENT_PATCH}"
         ALSR_LLAMA_CPP_CURRENT_PATCH_SHA256)
    execute_process(
      COMMAND git -C "${ALSR_LLAMA_CPP_SOURCE_DIR}" status
              --porcelain --untracked-files=all
      OUTPUT_VARIABLE ALSR_LLAMA_CPP_STATUS
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ALSR_LLAMA_CPP_STATUS MATCHES "(^|\n)\\?\\?")
      message(FATAL_ERROR
        "M2 pinned llama.cpp tree contains untracked files")
    endif()
    if(NOT ALSR_LLAMA_CPP_CURRENT_PATCH_SHA256 STREQUAL
           ALSR_LLAMA_CPP_PATCH_SHA256)
      message(FATAL_ERROR
        "M2 llama.cpp tree differs from canonical patch: expected ${ALSR_LLAMA_CPP_PATCH_SHA256}, got ${ALSR_LLAMA_CPP_CURRENT_PATCH_SHA256}")
    endif()
    # CPU repack creates a second model-sized allocation and bypasses the
    # host-mmap -> finite Metal split-copy path required by M2.
    set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
  endif()
  add_subdirectory("${ALSR_LLAMA_CPP_SOURCE_DIR}"
                   "${CMAKE_BINARY_DIR}/_deps/llama.cpp" EXCLUDE_FROM_ALL)

  if(ALSR_ENABLE_M1_LLAMA_TRACE)
  add_executable(alsr-llama-graph-trace src/cli/alsr_llama_graph_trace.cpp)
  target_link_libraries(alsr-llama-graph-trace PRIVATE alsr_core llama)
  target_include_directories(alsr-llama-graph-trace SYSTEM PRIVATE
    "${ALSR_LLAMA_CPP_SOURCE_DIR}/include"
    "${ALSR_LLAMA_CPP_SOURCE_DIR}/src"
    "${ALSR_LLAMA_CPP_SOURCE_DIR}/ggml/include")
  target_compile_definitions(alsr-llama-graph-trace PRIVATE
    ALSR_LLAMA_CPP_TAG="b10091"
    ALSR_LLAMA_CPP_COMMIT="${ALSR_LLAMA_CPP_REQUIRED_COMMIT}")
  if(NOT MSVC)
    target_compile_options(alsr-llama-graph-trace PRIVATE
      -Wall -Wextra -Wpedantic)
    if(ALSR_WERROR)
      target_compile_options(alsr-llama-graph-trace PRIVATE -Werror)
    endif()
  endif()
  endif()

  if(ALSR_ENABLE_M2_EXECUTOR)
    add_executable(alsr-m5-executor src/cli/alsr_m5_executor.cpp)
    target_link_libraries(alsr-m5-executor PRIVATE alsr_core llama)
    target_include_directories(alsr-m5-executor SYSTEM PRIVATE
      "${ALSR_LLAMA_CPP_SOURCE_DIR}/include"
      "${ALSR_LLAMA_CPP_SOURCE_DIR}/src"
      "${ALSR_LLAMA_CPP_SOURCE_DIR}/ggml/include")
    target_compile_definitions(alsr-m5-executor PRIVATE
      ALSR_LLAMA_CPP_TAG="b10091"
      ALSR_LLAMA_CPP_COMMIT="${ALSR_LLAMA_CPP_REQUIRED_COMMIT}"
      ALSR_LLAMA_CPP_PATCH_SHA256="${ALSR_LLAMA_CPP_PATCH_SHA256}"
      ALSR_M2_CPU_REPACK_DISABLED=1)
    if(NOT MSVC)
      target_compile_options(alsr-m5-executor PRIVATE
        -Wall -Wextra -Wpedantic)
      if(ALSR_WERROR)
        target_compile_options(alsr-m5-executor PRIVATE -Werror)
      endif()
    endif()
  endif()
endif()

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
