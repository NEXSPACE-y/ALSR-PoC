// ALSR — real-runtime graph trace to deterministic byte-range manifest (M1).
//
// The input trace must come from an instrumented real Runtime graph. This module
// deliberately does not infer execution order from tensor names.
#pragma once

#include "alsr/common.hpp"
#include "alsr/gguf.hpp"
#include "alsr/json.hpp"

namespace alsr {

struct RealGraphManifestOptions {
  Bytes target_chunk_bytes = 256 * MiB;
  Bytes read_buffer_bytes = 4 * MiB;
  bool compute_checksums = true;
};

/// Join a real Runtime graph trace to exact GGUF tensor ranges and produce a
/// deterministic, integrity-checked storage and demand plan.
[[nodiscard]] json::Value build_real_graph_manifest(
    const GgufFile& gguf, const json::Value& runtime_trace,
    const RealGraphManifestOptions& options = {});

/// Recheck structural coverage, tensor-fragment coverage, and mapped-use
/// invariants. Throws Error(Consistency) on any mismatch.
void verify_real_graph_manifest(const GgufFile& gguf,
                                const json::Value& manifest);

}  // namespace alsr
