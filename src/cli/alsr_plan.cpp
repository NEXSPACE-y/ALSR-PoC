// alsr-plan — Chunk 計画の生成と検証 (FR-03)。
#include <iostream>

#include "alsr/checksum.hpp"
#include "alsr/gguf.hpp"
#include "alsr/manifest.hpp"
#include "alsr/planner.hpp"
#include "cli_common.hpp"

using namespace alsr;

namespace {

void usage() {
  std::cout << R"(alsr-plan — Chunk 計画の生成

使い方:
  alsr-plan (--model <path> | --ollama <name:tag>) [options]

オプション:
  --mode <fixed|variable>  分割方式 (既定 fixed)
  --chunk-mib <n>          fixed の目標 Chunk サイズ [MiB] (既定 256)
  --min-mib <n>            variable の下限 [MiB] (既定 16)
  --max-mib <n>            variable の上限 [MiB] (既定 1024)
  --no-checksum            CRC32C を計算しない（高速だが検証は弱くなる）
  --out <path>             plan.json の出力先
  --manifest-out <path>    manifest.json も同時に出力する
  --sha256                 モデル全体の SHA-256 を計算して plan に埋める
  --quiet

不変条件（毎回検証し、破れたら非ゼロ終了）:
  1. Chunk 群が [data_start, file_size) を隙間なく重複なく敷き詰める
  2. 各テンソルの各バイトがちょうど 1 つの Chunk に属する
  3. テンソル内部の切れ目が lcm(量子ブロック, alignment) の倍数
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
    args.reject_unknown({"model", "ollama", "mode", "chunk-mib", "min-mib",
                         "max-mib", "no-checksum", "out", "manifest-out",
                         "sha256", "quiet", "help"});

    const std::string path = cli::resolve_model_path(args);
    const bool quiet = args.flag("quiet");

    PlanOptions opt;
    opt.mode = chunk_mode_from_string(args.get("mode", "fixed"));
    opt.target_bytes = args.get_u64("chunk-mib", 256) * MiB;
    opt.min_bytes    = args.get_u64("min-mib", 16) * MiB;
    opt.max_bytes    = args.get_u64("max-mib", 1024) * MiB;
    opt.compute_checksums = !args.flag("no-checksum");

    const GgufFile gguf = read_gguf(path);
    Manifest manifest = build_manifest(gguf);
    if (args.flag("sha256")) {
      if (!quiet) std::cout << "computing SHA-256 ...\n";
      manifest.model_sha256 = sha256_file(path);
    }

    if (!quiet) {
      std::cout << "planning " << gguf.tensor_count << " tensors, mode="
                << to_string(opt.mode) << " ...\n";
    }
    const ChunkPlan plan = build_plan(gguf, manifest, opt);
    verify_plan(gguf, plan);  // 破れていればここで Error(Consistency)

    if (!quiet) {
      std::cout << "\n=== Chunk 計画 ===\n";
      std::cout << "  chunks              : " << plan.chunks.size() << "\n";
      std::cout << "  covered bytes       : " << plan.covered_bytes << " ("
                << human_bytes(plan.covered_bytes) << ")\n";
      std::cout << "  tensor bytes        : " << plan.tensor_bytes_total << "\n";
      std::cout << "  padding bytes       : " << plan.padding_bytes_total << "\n";
      std::cout << "  min / max chunk     : " << plan.min_chunk_bytes() << " / "
                << plan.max_chunk_bytes() << "\n";
      std::cout << "  split tensors       : " << plan.split_tensor_count << "\n";
      std::cout << "  exec-order 逆転     : " << plan.exec_order_inversions
                << " / " << (plan.chunks.empty() ? 0 : plan.chunks.size() - 1)
                << "\n";
      std::cout << "  checksums           : "
                << (opt.compute_checksums ? "computed" : "skipped") << "\n";
      std::cout << "  structure sha256    : " << plan.plan_structure_sha256 << "\n";
      std::cout << "  verify_plan()       : OK（被覆・重複・境界すべて検証済み）\n";
      std::cout << "\n  ※ exec-order 逆転が多いほど、ファイル順の逐次先読みが"
                   "実行順と噛み合わない。\n";
    }

    if (args.has("out")) {
      const std::string p = args.get("out");
      std::ofstream out(p, std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + p);
      out << to_json(plan).dump(2) << "\n";
      if (!quiet) std::cout << "\nwrote " << p << "\n";
    }
    if (args.has("manifest-out")) {
      const std::string p = args.get("manifest-out");
      std::ofstream out(p, std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + p);
      out << to_json(manifest).dump(2) << "\n";
      if (!quiet) std::cout << "wrote " << p << "\n";
    }
    return 0;
  });
}
