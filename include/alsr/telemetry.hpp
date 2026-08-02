// ALSR PoC — Telemetry (FR-10) と Evidence Pack (FR-11)。
//
// results/<run_id>/
//   config.json / environment.json / manifest.json / metrics.jsonl
//   summary.json / summary.csv / report.md / stdout.log / stderr.log
//
// §16.4 結果の改変防止:
//   - raw metrics は追記専用で書く
//   - summary に raw metrics の SHA-256 を記録する
//   - 集計は必ずこのコードを通す（手作業で数値を書き換えない）
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/json.hpp"

namespace alsr {

struct ChunkMetric {
  u64   chunk_id = 0;
  Bytes bytes = 0;
  std::size_t queue_depth_before = 0;
  std::size_t queue_depth_after = 0;

  double prepare_ms = 0.0;
  double h2d_ms = 0.0;
  double compute_ms = 0.0;
  double wait_ms = 0.0;             ///< この Chunk の直前に GPU が待った時間
  double effective_h2d_gbps = 0.0;

  bool        checksum_ok = true;
  std::string scheduler_action = "keep";

  // 診断用の絶対時刻（仮想秒）
  double t_prepare_start_s = 0.0;
  double t_copy_start_s = 0.0;
  double t_copy_end_s = 0.0;
  double t_compute_start_s = 0.0;
  double t_compute_end_s = 0.0;

  std::size_t slot_id = 0;
  u64         slot_generation = 0;
  Bytes       chunk_size_setting = 0;
  std::size_t prefetch_depth_setting = 0;
};

/// JSONL を追記専用で書く。ホットパスを塞がないようバッファリングする。
class MetricsWriter {
public:
  explicit MetricsWriter(std::string path);
  ~MetricsWriter();

  MetricsWriter(const MetricsWriter&) = delete;
  MetricsWriter& operator=(const MetricsWriter&) = delete;

  void write(const std::string& run_id, const ChunkMetric& m);
  void flush();
  void close();

  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] u64 lines_written() const noexcept { return lines_; }

private:
  std::string   path_;
  std::ofstream out_;
  u64           lines_ = 0;
};

struct Percentiles {
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double max = 0.0;

  [[nodiscard]] json::Value to_json() const;
};

/// nearest-rank 法。空なら全て 0。
[[nodiscard]] Percentiles compute_percentiles(std::vector<double> values);

struct RunSummary {
  std::string run_id;
  std::string scheduler_name;
  std::string backend_name;

  u64   chunk_count = 0;
  Bytes total_bytes = 0;
  double virtual_seconds = 0.0;   ///< 仮想時間での総実行時間
  double wall_seconds = 0.0;      ///< 実測の壁時計（計測オーバーヘッド把握用）

  double chunks_per_s = 0.0;
  double effective_h2d_gbps = 0.0;
  double gpu_wait_ratio = 0.0;
  double copy_compute_overlap_ratio = 0.0;

  /// per_token_bytes が既知なら導出する。0 なら未算出。
  Bytes  per_token_bytes = 0;
  double tokens_per_s = 0.0;

  u64    starvation_events = 0;
  double starvation_total_s = 0.0;
  double starvation_max_s = 0.0;
  /// 後半 / 前半の 1 Chunk あたり平均待機時間。前半が 0 なら 0。
  double starvation_progression_ratio = 0.0;
  /// 最後の 1/4 区間での GPU 待機率。0 より大きいなら定常状態で枯渇している。
  double late_wait_ratio = 0.0;

  Percentiles copy_ms;
  Percentiles compute_ms;
  Percentiles prepare_ms;
  Percentiles wait_ms;

  Bytes pinned_bytes = 0;
  Bytes peak_host_bytes = 0;
  Bytes peak_device_bytes = 0;

  u64 checksum_failures = 0;
  u64 missing_chunks = 0;
  u64 duplicate_chunks = 0;
  u64 order_violations = 0;
  u64 unhandled_errors = 0;
  u64 scheduler_changes = 0;
  u64 ring_wrap_arounds = 0;

  std::string metrics_sha256;
  bool        environment_recorded = false;
  /// 実データを流したか。false の Run は正確性の証拠にならない。
  bool        data_materialized = true;

  // §12.1 必須ゲート
  bool gate_pass = false;
  std::vector<std::string> gate_failures;

  [[nodiscard]] json::Value to_json() const;
  [[nodiscard]] std::string to_csv_header() const;
  [[nodiscard]] std::string to_csv_row() const;
};

/// §12.1 の必須ゲートを判定して gate_pass / gate_failures を埋める。
void evaluate_mandatory_gates(RunSummary& s);

/// Evidence Pack の書き出し先を管理する。
class EvidencePack {
public:
  EvidencePack(std::string results_root, std::string run_id);

  [[nodiscard]] const std::string& dir() const noexcept { return dir_; }
  [[nodiscard]] std::string file(const std::string& name) const;

  void write_json(const std::string& name, const json::Value& v) const;
  void write_text(const std::string& name, const std::string& text) const;

private:
  std::string dir_;
};

/// report.md を生成する。数値は summary.json と同じ出所しか使わない。
[[nodiscard]] std::string render_report(const RunSummary& s,
                                        const json::Value& config,
                                        const json::Value& environment);

/// 決定論的な run_id。config の内容から作るので、同一設定なら同一 ID になる。
[[nodiscard]] std::string make_run_id(const std::string& label,
                                      const json::Value& config);

}  // namespace alsr
