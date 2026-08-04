// alsr-real-manifest — join an actual llama.cpp graph trace to GGUF ranges.
#include <fstream>
#include <iostream>

#include "alsr/gguf.hpp"
#include "alsr/json.hpp"
#include "alsr/real_graph_manifest.hpp"
#include "cli_common.hpp"

using namespace alsr;

namespace {

void usage() {
  std::cout << R"(alsr-real-manifest — real graph trace / GGUF range join

Usage:
  alsr-real-manifest --model <GGUF> --trace <trace.json> --out <manifest.json>
                     [--chunk-mib <n>] [--no-checksum]

The trace must use schema alsr.llama.graph-trace.v2. A heuristic tensor-name
order is rejected.
)";
}

}  // namespace

int main(int argc, char** argv) {
  return cli::run_main([&]() -> int {
    cli::Args args(argc, argv);
    if (args.flag("help") || argc == 1) {
      usage();
      return 0;
    }
    args.reject_unknown({"model", "trace", "out", "chunk-mib",
                         "no-checksum", "help"});
    if (!args.has("model") || !args.has("trace") || !args.has("out")) {
      fail(ErrorKind::Invalid, "--model, --trace, and --out are required");
    }
    const GgufFile gguf = read_gguf(args.get("model"));
    const json::Value trace = json::Value::parse_file(args.get("trace"));
    RealGraphManifestOptions options;
    options.target_chunk_bytes = args.get_u64("chunk-mib", 256) * MiB;
    options.compute_checksums = !args.flag("no-checksum");
    const json::Value manifest = build_real_graph_manifest(gguf, trace, options);
    std::ofstream out(args.get("out"), std::ios::trunc);
    if (!out) fail(ErrorKind::Io, "cannot write real graph manifest");
    out << manifest.dump(2) << '\n';
    std::cout << "wrote " << args.get("out") << "\n";
    std::cout << "structure sha256: "
              << manifest.at("manifest_structure_sha256").as_string() << "\n";
    return 0;
  });
}
