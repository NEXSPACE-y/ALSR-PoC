// alsr-inspect — GGUF 解析と実行順序マニフェスト生成 (FR-01 / FR-02)。
//
// モデルファイルは読み取り専用でしか触らない (NFR-03)。
#include <iostream>
#include <sstream>

#include "alsr/checksum.hpp"
#include "alsr/env.hpp"
#include "alsr/gguf.hpp"
#include "alsr/ggml_types.hpp"
#include "alsr/manifest.hpp"
#include "cli_common.hpp"

using namespace alsr;

namespace {

void usage() {
  std::cout << R"(alsr-inspect — GGUF 解析 / 実行順序マニフェスト

使い方:
  alsr-inspect (--model <path> | --ollama <name:tag>) [options]

オプション:
  --model <path>        GGUF ファイル
  --ollama <name:tag>   Ollama の blob を解決して使う (例: gemma4:12b)
  --manifest-out <path> manifest.json の出力先
  --gguf-out <path>     GGUF 生情報 JSON の出力先
  --env-out <path>      environment.json の出力先
  --sha256              モデル全体の SHA-256 を計算する（大きいと時間がかかる）
  --bandwidths <list>   T_max 表に使う帯域 [GB/s]、カンマ区切り
                        既定: 5,10,12,16,25,28,113
  --resident-mib <n>    常駐候補とみなすテンソルサイズ上限 [MiB] (既定 1)
  --quiet               標準出力のサマリーを抑制する
)";
}

std::vector<double> parse_bandwidths(const std::string& s) {
  std::vector<double> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(item.c_str(), &end);
    if (errno != 0 || end == nullptr || *end != '\0' || !(v > 0.0)) {
      fail(ErrorKind::Invalid, "--bandwidths に不正な値: '" + item + "'");
    }
    out.push_back(v * 1e9);  // GB/s -> byte/s
  }
  if (out.empty()) fail(ErrorKind::Invalid, "--bandwidths が空");
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  return cli::run_main([&]() -> int {
    cli::Args args(argc, argv);
    if (args.flag("help") || argc == 1) {
      usage();
      return 0;
    }
    args.reject_unknown({"model", "ollama", "manifest-out", "gguf-out", "env-out",
                         "sha256", "bandwidths", "resident-mib", "quiet", "help"});

    const std::string path = cli::resolve_model_path(args);
    const bool quiet = args.flag("quiet");

    if (!quiet) std::cout << "model: " << path << "\n";

    const GgufFile gguf = read_gguf(path);
    const Bytes resident_threshold = args.get_u64("resident-mib", 1) * MiB;
    Manifest manifest = build_manifest(gguf, resident_threshold);

    if (args.flag("sha256")) {
      if (!quiet) std::cout << "computing SHA-256 ...\n";
      manifest.model_sha256 = sha256_file(path);
    }

    // --- 標準出力サマリー ------------------------------------------------
    if (!quiet) {
      std::cout << "\n=== GGUF ===\n";
      std::cout << "  version        : " << gguf.version << "\n";
      std::cout << "  architecture   : " << gguf.architecture << "\n";
      std::cout << "  file size      : " << gguf.file_size << " ("
                << human_bytes(gguf.file_size) << ")\n";
      std::cout << "  alignment      : " << gguf.alignment << "\n";
      std::cout << "  header end     : " << gguf.header_end << "\n";
      std::cout << "  data start     : " << gguf.data_start << "\n";
      std::cout << "  tensors        : " << gguf.tensor_count << "\n";
      std::cout << "  metadata kv    : " << gguf.kv_count << "\n";
      if (!manifest.model_sha256.empty()) {
        std::cout << "  sha256         : " << manifest.model_sha256 << "\n";
      }

      std::cout << "\n=== レイアウト整合 (Phase 1 合格条件) ===\n";
      std::cout << "  data bytes     : " << gguf.layout.data_bytes << "\n";
      std::cout << "  tensor total   : " << gguf.layout.tensor_bytes_total << "\n";
      std::cout << "  padding        : " << gguf.layout.padding_bytes << "\n";
      std::cout << "  unexplained    : " << gguf.layout.unexplained_bytes << "\n";
      std::cout << "  issues         : " << gguf.layout.issues.size() << "\n";
      std::cout << "  => " << (gguf.layout.ok() ? "OK（重複／欠落なし・全て説明可能）"
                                                : "NG（下記 issue を確認）")
                << "\n";
      for (const auto& is : gguf.layout.issues) {
        std::cout << "     - [" << is.kind << "] @" << is.offset << " +" << is.length
                  << " : " << is.detail << "\n";
      }

      std::cout << "\n=== 実行順序マニフェスト ===\n";
      std::cout << "  block count    : " << manifest.block_count << "\n";
      std::cout << "  entries        : " << manifest.entries.size() << "\n";
      std::cout << "  unknown        : " << manifest.unknown_count << "\n";
      std::cout << "  human review   : "
                << (manifest.requires_human_review ? "REQUIRED" : "not required")
                << "\n";
      for (const auto& n : manifest.review_notes) {
        std::cout << "     ! " << n << "\n";
      }

      std::cout << "\n=== 物理量 (設計書 §2) ===\n";
      std::cout << "  total tensor bytes       : " << manifest.total_tensor_bytes
                << " (" << human_bytes(manifest.total_tensor_bytes) << ")\n";
      std::cout << "  resident candidate bytes : "
                << manifest.resident_candidate_bytes << " ("
                << human_bytes(manifest.resident_candidate_bytes) << ")\n";
      std::cout << "  S_stream lower           : " << manifest.per_token_bytes_lower
                << " (" << human_bytes(manifest.per_token_bytes_lower)
                << ") / token\n";
      std::cout << "  S_stream upper           : " << manifest.per_token_bytes_upper
                << " (" << human_bytes(manifest.per_token_bytes_upper)
                << ") / token\n";
      std::cout << "  separate output.weight   : "
                << (manifest.has_separate_output_tensor ? "yes" : "no (tied embedding の可能性)")
                << "\n";
    }

    const std::vector<double> bw =
        parse_bandwidths(args.get("bandwidths", "5,10,12,16,25,28,113"));
    const json::Value ceiling = bandwidth_ceiling_table(manifest, bw);

    if (!quiet) {
      std::cout << "\n=== 転送だけの理想上限 T_max = B / S_stream (§2.3) ===\n";
      std::cout << "  帯域[GB/s]   T_max(upper)[tok/s]   T_max(lower)[tok/s]\n";
      for (const json::Value& r : ceiling.at("rows").as_array()) {
        char line[160];
        std::snprintf(line, sizeof line, "  %9.2f   %19.4f   %19.4f\n",
                      r.at("bandwidth_gb_per_s").as_real(),
                      r.at("t_max_token_per_s_upper_bound_case").as_real(),
                      r.at("t_max_token_per_s_lower_bound_case").as_real());
        std::cout << line;
      }
      std::cout << "  ※ カーネル実行・同期・KV キャッシュ・再配置は含まない。"
                   "実効値は必ずこれより低い。\n";
    }

    // --- 出力 -----------------------------------------------------------
    if (args.has("gguf-out")) {
      const std::string p = args.get("gguf-out");
      std::ofstream out(p, std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + p);
      out << to_json(gguf).dump(2) << "\n";
      if (!quiet) std::cout << "\nwrote " << p << "\n";
    }
    if (args.has("manifest-out")) {
      json::Value mj = to_json(manifest);
      mj.set("bandwidth_ceiling", ceiling);
      mj.set("gguf_layout", to_json(gguf).at("layout"));
      const std::string p = args.get("manifest-out");
      std::ofstream out(p, std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + p);
      out << mj.dump(2) << "\n";
      if (!quiet) std::cout << "wrote " << p << "\n";
    }
    if (args.has("env-out")) {
      const std::string p = args.get("env-out");
      std::ofstream out(p, std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + p);
      out << collect_environment().dump(2) << "\n";
      if (!quiet) std::cout << "wrote " << p << "\n";
    }

    // レイアウト不整合や未分類は非ゼロ終了で知らせる（CI で拾えるように）。
    if (!gguf.layout.ok()) return 4;
    if (manifest.requires_human_review) return 5;
    return 0;
  });
}
