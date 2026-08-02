// alsr-sim — 仮想時間ストリーミングシミュレータ + Evidence Pack 生成
// (FR-04〜FR-07 / FR-10 / FR-11)。
#include <iostream>

#include "alsr/checksum.hpp"
#include "alsr/env.hpp"
#include "alsr/sim.hpp"
#include "alsr/telemetry.hpp"
#include "cli_common.hpp"

using namespace alsr;

namespace {

void usage() {
  std::cout << R"(alsr-sim — Chunk ストリーミングの仮想時間シミュレーション

使い方:
  alsr-sim --config <config.json> [options]

オプション:
  --config <path>     実験設定 JSON（必須）
  --results <dir>     Evidence Pack の出力先ルート (既定 results)
  --run-id <id>       Run ID を明示指定（省略時は設定内容から決定論的に生成）
  --label <name>      Run ID の接頭辞を上書き
  --no-evidence       Evidence Pack を書かない（テスト用）
  --print-summary     summary.json を標準出力にも出す
  --quiet

注意:
  これは M5 上の論理検証であり、CUDA 性能の予測ではない（設計書 §3.1）。
  backend.measured_on_real_hardware は常に false。
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
    args.reject_unknown({"config", "results", "run-id", "label", "no-evidence",
                         "print-summary", "quiet", "help"});

    const std::string config_path = args.require("config");
    const bool quiet = args.flag("quiet");

    const json::Value config_json = json::Value::parse_file(config_path);
    SimConfig cfg = SimConfig::from_json(config_json);
    if (args.has("label")) cfg.label = args.get("label");

    // 正規化した設定を残す。ここから run_id を作るので、同一設定 → 同一 ID。
    const json::Value normalized = cfg.to_json();
    const std::string run_id =
        args.has("run-id") ? args.get("run-id") : make_run_id(cfg.label, normalized);

    if (!quiet) {
      std::cout << "run_id : " << run_id << "\n";
      std::cout << "config : " << config_path << "\n";
    }

    const bool write_evidence = !args.flag("no-evidence");
    std::unique_ptr<EvidencePack> pack;
    std::unique_ptr<MetricsWriter> writer;

    if (write_evidence) {
      pack = std::make_unique<EvidencePack>(args.get("results", "results"), run_id);
      // 追記専用なので、同じ run_id の再実行で混ざらないよう既存を消す。
      std::error_code ec;
      std::filesystem::remove(pack->file("metrics.jsonl"), ec);
      writer = std::make_unique<MetricsWriter>(pack->file("metrics.jsonl"));
    }

    SimResult result = run_simulation(cfg, run_id, writer.get());
    if (writer) writer->close();

    const json::Value environment = collect_environment();
    result.summary.environment_recorded = write_evidence;
    if (write_evidence) {
      result.summary.metrics_sha256 = sha256_file(pack->file("metrics.jsonl"));
    }
    evaluate_mandatory_gates(result.summary);

    json::Value full_config = normalized;
    full_config.set("resolved_scheduler", result.scheduler_config);
    full_config.set("resolved_backend", result.backend_config);
    full_config.set("source_config_path", config_path);

    if (write_evidence) {
      pack->write_json("config.json", full_config);
      pack->write_json("environment.json", environment);
      pack->write_json("summary.json", result.summary.to_json());
      pack->write_text("summary.csv", result.summary.to_csv_header() + "\n" +
                                          result.summary.to_csv_row() + "\n");
      pack->write_text("report.md",
                       render_report(result.summary, full_config, environment));

      json::Value sched_log = json::Value::make_array();
      for (const auto& l : result.scheduler_change_log) {
        sched_log.push_back(json::Value(l));
      }
      json::Value sl = json::Value::make_object();
      sl.set("changes", std::move(sched_log));
      sl.set("note", "制御変更の理由 (FR-06)。振動の有無はここで確認する。");
      pack->write_json("scheduler_log.json", sl);
    }

    if (!quiet) {
      const RunSummary& s = result.summary;
      std::cout << "\n=== Run summary ===\n";
      std::cout << "  scheduler        : " << s.scheduler_name << "\n";
      std::cout << "  chunks           : " << s.chunk_count << "\n";
      std::cout << "  bytes            : " << s.total_bytes << " ("
                << human_bytes(s.total_bytes) << ")\n";
      std::cout << "  virtual seconds  : " << s.virtual_seconds << "\n";
      std::cout << "  wall seconds     : " << s.wall_seconds << "\n";
      std::cout << "  effective H2D    : " << s.effective_h2d_gbps << " GB/s\n";
      if (s.per_token_bytes > 0) {
        std::cout << "  tokens/s         : " << s.tokens_per_s << "\n";
      }
      std::cout << "  GPU wait ratio   : " << (s.gpu_wait_ratio * 100.0) << " %\n";
      std::cout << "  late wait ratio  : " << (s.late_wait_ratio * 100.0) << " %\n";
      std::cout << "  overlap ratio    : "
                << (s.copy_compute_overlap_ratio * 100.0) << " %\n";
      std::cout << "  starvation       : " << s.starvation_events << " events, "
                << s.starvation_total_s << " s total\n";
      std::cout << "  checksum fail    : " << s.checksum_failures << "\n";
      std::cout << "  order violations : " << s.order_violations << "\n";
      std::cout << "  wrap-arounds     : " << s.ring_wrap_arounds << "\n";
      std::cout << "  sched changes    : " << s.scheduler_changes << "\n";
      std::cout << "  mandatory gates  : " << (s.gate_pass ? "PASS" : "FAIL") << "\n";
      for (const auto& f : s.gate_failures) std::cout << "     - " << f << "\n";
      if (result.cancelled) {
        std::cout << "  cancelled        : yes (all slots released: "
                  << (result.cancel_released_all_slots ? "yes" : "NO") << ")\n";
      }
      if (write_evidence) std::cout << "\nevidence: " << pack->dir() << "\n";
    }

    if (args.flag("print-summary")) {
      std::cout << result.summary.to_json().dump(2) << "\n";
    }

    // ゲート失敗は非ゼロ終了。ただし cancel テストは意図的な中断なので 0。
    if (!result.summary.gate_pass && !result.cancelled) return 6;
    return 0;
  });
}
