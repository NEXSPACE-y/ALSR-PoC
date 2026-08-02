#include "alsr/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "alsr/checksum.hpp"

namespace alsr {
namespace {

std::string fmt(double v, int prec = 3) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}

/// 0 除算を黙って inf/NaN にしない。分母が 0 なら 0 を返す。
double safe_div(double a, double b) noexcept {
  if (!(b > 0.0) || !std::isfinite(b) || !std::isfinite(a)) return 0.0;
  const double r = a / b;
  return std::isfinite(r) ? r : 0.0;
}

}  // namespace

// ---------------------------------------------------------------------------
// MetricsWriter
// ---------------------------------------------------------------------------
MetricsWriter::MetricsWriter(std::string path) : path_(std::move(path)) {
  out_.open(path_, std::ios::out | std::ios::app | std::ios::binary);
  if (!out_) fail(ErrorKind::Io, "cannot open metrics file for append: " + path_);
}

MetricsWriter::~MetricsWriter() {
  try {
    close();
  } catch (...) {
    // デストラクタから例外を投げない。close() の失敗は flush 時に検出される。
  }
}

void MetricsWriter::write(const std::string& run_id, const ChunkMetric& m) {
  json::Value j = json::Value::make_object();
  j.set("run_id", run_id);
  j.set("chunk_id", m.chunk_id);
  j.set("bytes", m.bytes);
  j.set("queue_depth_before", m.queue_depth_before);
  j.set("queue_depth_after", m.queue_depth_after);
  j.set("prepare_ms", m.prepare_ms);
  j.set("h2d_ms", m.h2d_ms);
  j.set("compute_ms", m.compute_ms);
  j.set("wait_ms", m.wait_ms);
  j.set("effective_h2d_gbps", m.effective_h2d_gbps);
  j.set("checksum_ok", m.checksum_ok);
  j.set("scheduler_action", m.scheduler_action);
  j.set("t_prepare_start_s", m.t_prepare_start_s);
  j.set("t_copy_start_s", m.t_copy_start_s);
  j.set("t_copy_end_s", m.t_copy_end_s);
  j.set("t_compute_start_s", m.t_compute_start_s);
  j.set("t_compute_end_s", m.t_compute_end_s);
  j.set("slot_id", m.slot_id);
  j.set("slot_generation", m.slot_generation);
  j.set("chunk_size_setting", m.chunk_size_setting);
  j.set("prefetch_depth_setting", m.prefetch_depth_setting);

  out_ << j.dump() << '\n';
  if (!out_) fail(ErrorKind::Io, "metrics write failed: " + path_);
  ++lines_;
}

void MetricsWriter::flush() {
  out_.flush();
  if (!out_) fail(ErrorKind::Io, "metrics flush failed: " + path_);
}

void MetricsWriter::close() {
  if (out_.is_open()) {
    out_.flush();
    const bool ok = static_cast<bool>(out_);
    out_.close();
    if (!ok) fail(ErrorKind::Io, "metrics close failed: " + path_);
  }
}

// ---------------------------------------------------------------------------
// Percentiles
// ---------------------------------------------------------------------------
json::Value Percentiles::to_json() const {
  json::Value j = json::Value::make_object();
  j.set("p50", p50);
  j.set("p95", p95);
  j.set("p99", p99);
  j.set("mean", mean);
  j.set("max", max);
  return j;
}

Percentiles compute_percentiles(std::vector<double> values) {
  Percentiles p;
  if (values.empty()) return p;
  std::sort(values.begin(), values.end());

  const auto nearest_rank = [&values](double q) {
    const double n = static_cast<double>(values.size());
    auto idx = static_cast<std::size_t>(std::ceil(q * n));
    if (idx == 0) idx = 1;
    if (idx > values.size()) idx = values.size();
    return values[idx - 1];
  };

  p.p50 = nearest_rank(0.50);
  p.p95 = nearest_rank(0.95);
  p.p99 = nearest_rank(0.99);
  p.max = values.back();
  double sum = 0.0;
  for (const double v : values) sum += v;
  p.mean = safe_div(sum, static_cast<double>(values.size()));
  return p;
}

// ---------------------------------------------------------------------------
// RunSummary
// ---------------------------------------------------------------------------
json::Value RunSummary::to_json() const {
  json::Value j = json::Value::make_object();
  j.set("run_id", run_id);
  j.set("scheduler", scheduler_name);
  j.set("backend", backend_name);

  json::Value th = json::Value::make_object();
  th.set("chunk_count", chunk_count);
  th.set("total_bytes", total_bytes);
  th.set("virtual_seconds", virtual_seconds);
  th.set("wall_seconds", wall_seconds);
  th.set("chunks_per_s", chunks_per_s);
  th.set("effective_h2d_gbps", effective_h2d_gbps);
  th.set("per_token_bytes", per_token_bytes);
  th.set("tokens_per_s", tokens_per_s);
  j.set("throughput", std::move(th));

  json::Value ov = json::Value::make_object();
  ov.set("gpu_wait_ratio", gpu_wait_ratio);
  ov.set("copy_compute_overlap_ratio", copy_compute_overlap_ratio);
  ov.set("starvation_events", starvation_events);
  ov.set("starvation_total_s", starvation_total_s);
  ov.set("starvation_max_s", starvation_max_s);
  ov.set("starvation_progression_ratio", starvation_progression_ratio);
  ov.set("late_wait_ratio", late_wait_ratio);
  j.set("overlap_and_starvation", std::move(ov));

  json::Value lat = json::Value::make_object();
  lat.set("copy_ms", copy_ms.to_json());
  lat.set("compute_ms", compute_ms.to_json());
  lat.set("prepare_ms", prepare_ms.to_json());
  lat.set("wait_ms", wait_ms.to_json());
  j.set("latency", std::move(lat));

  json::Value mem = json::Value::make_object();
  mem.set("pinned_bytes", pinned_bytes);
  mem.set("peak_host_bytes", peak_host_bytes);
  mem.set("peak_device_bytes", peak_device_bytes);
  mem.set("resident_bytes", resident_bytes);
  mem.set("resident_chunk_count", resident_chunk_count);
  mem.set("non_per_token_bytes", non_per_token_bytes);
  j.set("memory", std::move(mem));

  json::Value corr = json::Value::make_object();
  corr.set("checksum_failures", checksum_failures);
  corr.set("missing_chunks", missing_chunks);
  corr.set("duplicate_chunks", duplicate_chunks);
  corr.set("order_violations", order_violations);
  corr.set("unhandled_errors", unhandled_errors);
  corr.set("ring_wrap_arounds", ring_wrap_arounds);
  j.set("correctness", std::move(corr));

  j.set("scheduler_changes", scheduler_changes);
  j.set("metrics_sha256", metrics_sha256);
  j.set("model_digest", model_digest);
  j.set("model_digest_source", model_digest_source);
  j.set("environment_recorded", environment_recorded);
  j.set("data_materialized", data_materialized);

  json::Value g = json::Value::make_object();
  g.set("pass", gate_pass);
  json::Value f = json::Value::make_array();
  for (const auto& s : gate_failures) f.push_back(json::Value(s));
  g.set("failures", std::move(f));
  g.set("note",
        "§12.1 の必須ゲート。1 つでも失敗した Run は性能比較から除外する。");
  j.set("mandatory_gates", std::move(g));

  return j;
}

std::string RunSummary::to_csv_header() const {
  return "run_id,scheduler,backend,chunk_count,total_bytes,virtual_seconds,"
         "chunks_per_s,tokens_per_s,effective_h2d_gbps,gpu_wait_ratio,"
         "copy_compute_overlap_ratio,starvation_events,starvation_total_s,"
         "starvation_max_s,starvation_progression_ratio,late_wait_ratio,"
         "copy_p50_ms,copy_p95_ms,"
         "copy_p99_ms,compute_p50_ms,compute_p95_ms,compute_p99_ms,"
         "peak_device_bytes,checksum_failures,scheduler_changes,gate_pass";
}

std::string RunSummary::to_csv_row() const {
  std::ostringstream os;
  os << run_id << ',' << scheduler_name << ',' << backend_name << ','
     << chunk_count << ',' << total_bytes << ',' << fmt(virtual_seconds, 6) << ','
     << fmt(chunks_per_s, 6) << ',' << fmt(tokens_per_s, 6) << ','
     << fmt(effective_h2d_gbps, 6) << ',' << fmt(gpu_wait_ratio, 6) << ','
     << fmt(copy_compute_overlap_ratio, 6) << ',' << starvation_events << ','
     << fmt(starvation_total_s, 6) << ',' << fmt(starvation_max_s, 6) << ','
     << fmt(starvation_progression_ratio, 6) << ',' << fmt(late_wait_ratio, 6)
     << ',' << fmt(copy_ms.p50) << ','
     << fmt(copy_ms.p95) << ',' << fmt(copy_ms.p99) << ',' << fmt(compute_ms.p50)
     << ',' << fmt(compute_ms.p95) << ',' << fmt(compute_ms.p99) << ','
     << peak_device_bytes << ',' << checksum_failures << ',' << scheduler_changes
     << ',' << (gate_pass ? 1 : 0);
  return os.str();
}

void evaluate_mandatory_gates(RunSummary& s) {
  s.gate_failures.clear();
  if (s.checksum_failures != 0) {
    s.gate_failures.push_back("checksum 一致率が 100% でない (" +
                              std::to_string(s.checksum_failures) + " 件失敗)");
  }
  if (s.missing_chunks != 0) {
    s.gate_failures.push_back("Chunk 欠落 " + std::to_string(s.missing_chunks));
  }
  if (s.duplicate_chunks != 0) {
    s.gate_failures.push_back("Chunk 重複 " + std::to_string(s.duplicate_chunks));
  }
  if (s.order_violations != 0) {
    s.gate_failures.push_back("順序違反 " + std::to_string(s.order_violations));
  }
  if (s.unhandled_errors != 0) {
    s.gate_failures.push_back("unhandled error " +
                              std::to_string(s.unhandled_errors));
  }
  if (!s.environment_recorded) {
    s.gate_failures.push_back("environment が保存されていない");
  }
  if (s.chunk_count == 0) {
    s.gate_failures.push_back("Chunk が 1 つも処理されていない");
  }
  if (!s.data_materialized) {
    s.gate_failures.push_back(
        "実データを流していない Run（materialize_data=false）。"
        "スケジューラ挙動の観察専用であり、正確性の証拠にはならない");
  }
  s.gate_pass = s.gate_failures.empty();
}

// ---------------------------------------------------------------------------
// EvidencePack
// ---------------------------------------------------------------------------
EvidencePack::EvidencePack(std::string results_root, std::string run_id) {
  namespace fs = std::filesystem;
  const fs::path p = fs::path(results_root) / run_id;
  std::error_code ec;
  fs::create_directories(p, ec);
  if (ec) {
    fail(ErrorKind::Io,
         "cannot create evidence directory " + p.string() + ": " + ec.message());
  }
  dir_ = p.string();
}

std::string EvidencePack::file(const std::string& name) const {
  return (std::filesystem::path(dir_) / name).string();
}

void EvidencePack::write_json(const std::string& name, const json::Value& v) const {
  write_text(name, v.dump(2) + "\n");
}

void EvidencePack::write_text(const std::string& name, const std::string& text) const {
  const std::string p = file(name);
  std::ofstream out(p, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!out) fail(ErrorKind::Io, "cannot write " + p);
  out << text;
  out.flush();
  if (!out) fail(ErrorKind::Io, "write failed for " + p);
}

// ---------------------------------------------------------------------------
// report.md
// ---------------------------------------------------------------------------
std::string render_report(const RunSummary& s, const json::Value& config,
                          const json::Value& environment) {
  std::ostringstream o;
  o << "# ALSR Run Report — `" << s.run_id << "`\n\n";

  o << "> 本レポートの数値は `summary.json` と同一の出所から生成している。\n"
    << "> 手作業で数値を書き換えないこと (設計書 §16.4)。\n\n";

  o << "## 必須ゲート (§12.1)\n\n";
  o << (s.gate_pass ? "**PASS** — 性能比較に使用してよい。\n"
                    : "**FAIL** — この Run は性能比較から除外する。\n");
  if (!s.gate_failures.empty()) {
    o << "\n";
    for (const auto& f : s.gate_failures) o << "- " << f << "\n";
  }
  o << "\n";

  o << "## スループット\n\n";
  o << "| 指標 | 値 |\n|---|---|\n";
  o << "| Chunk 数 | " << s.chunk_count << " |\n";
  o << "| 総転送バイト | " << s.total_bytes << " (" << human_bytes(s.total_bytes)
    << ") |\n";
  o << "| 仮想実行時間 | " << fmt(s.virtual_seconds, 6) << " s |\n";
  o << "| 実測壁時計 | " << fmt(s.wall_seconds, 6) << " s |\n";
  o << "| Chunk/s | " << fmt(s.chunks_per_s, 3) << " |\n";
  if (s.per_token_bytes > 0) {
    o << "| S_stream | " << s.per_token_bytes << " byte/token |\n";
    o << "| token/s (導出) | " << fmt(s.tokens_per_s, 4) << " |\n";
  }
  o << "| 実効 H2D | " << fmt(s.effective_h2d_gbps, 3) << " GB/s |\n\n";

  o << "## 重複と枯渇\n\n";
  o << "| 指標 | 値 |\n|---|---|\n";
  o << "| GPU 待機率 | " << fmt(s.gpu_wait_ratio * 100.0, 3) << " % |\n";
  o << "| copy/compute 重複率 | " << fmt(s.copy_compute_overlap_ratio * 100.0, 3)
    << " % |\n";
  o << "| starvation 回数 | " << s.starvation_events << " |\n";
  o << "| starvation 合計 | " << fmt(s.starvation_total_s, 6) << " s |\n";
  o << "| starvation 最長 | " << fmt(s.starvation_max_s, 6) << " s |\n";
  o << "| 進行性指標 (後半/前半の待機) | " << fmt(s.starvation_progression_ratio, 3)
    << " |\n";
  o << "| 終盤 1/4 の待機率 | " << fmt(s.late_wait_ratio * 100.0, 3) << " % |\n\n";
  o << "進行性指標が 1 を大きく超える場合、平均供給帯域が平均消費帯域を"
       "下回っている。有限深度のバッファでは必ず枯渇する (設計書 §2.2)。\n\n";

  o << "## 分位点 [ms]\n\n";
  o << "| 区間 | p50 | p95 | p99 | mean | max |\n|---|---|---|---|---|---|\n";
  const auto row = [&o](const char* n, const Percentiles& p) {
    o << "| " << n << " | " << fmt(p.p50) << " | " << fmt(p.p95) << " | "
      << fmt(p.p99) << " | " << fmt(p.mean) << " | " << fmt(p.max) << " |\n";
  };
  row("prepare", s.prepare_ms);
  row("h2d copy", s.copy_ms);
  row("compute", s.compute_ms);
  row("wait", s.wait_ms);
  o << "\n";

  o << "## 正確性\n\n";
  o << "| 指標 | 値 |\n|---|---|\n";
  o << "| checksum 失敗 | " << s.checksum_failures << " |\n";
  o << "| Chunk 欠落 | " << s.missing_chunks << " |\n";
  o << "| Chunk 重複 | " << s.duplicate_chunks << " |\n";
  o << "| 順序違反 | " << s.order_violations << " |\n";
  o << "| ring wrap-around | " << s.ring_wrap_arounds << " |\n";
  o << "| scheduler 変更回数 | " << s.scheduler_changes << " |\n\n";

  o << "## メモリ\n\n";
  o << "| 指標 | 値 |\n|---|---|\n";
  o << "| 最大デバイス側 | " << s.peak_device_bytes << " ("
    << human_bytes(s.peak_device_bytes) << ") |\n";
  o << "| 最大ホスト側 | " << s.peak_host_bytes << " ("
    << human_bytes(s.peak_host_bytes) << ") |\n";
  o << "| pinned 相当 | " << s.pinned_bytes << " (" << human_bytes(s.pinned_bytes)
    << ") |\n";
  o << "| VRAM 常駐重み | " << s.resident_bytes << " ("
    << human_bytes(s.resident_bytes) << ", " << s.resident_chunk_count
    << " chunk) |\n\n";
  if (s.resident_bytes > 0) {
    o << "常駐分は定常状態のストリーミング対象から外している。起動時の初期"
         "ロード転送量はこの Run の総転送バイトに含まれない（設計書 §2.4）。\n\n";
  }

  o << "## 証拠\n\n";
  o << "- `metrics.jsonl` SHA-256: `" << s.metrics_sha256 << "`\n";
  o << "- モデル digest: `" << (s.model_digest.empty() ? "未記録" : s.model_digest)
    << "` (" << s.model_digest_source << ")\n";
  o << "- environment 記録: " << (s.environment_recorded ? "あり" : "なし") << "\n\n";

  o << "## 設定\n\n```json\n" << config.dump(2) << "\n```\n\n";
  o << "## 環境\n\n```json\n" << environment.dump(2) << "\n```\n";
  return o.str();
}

std::string make_run_id(const std::string& label, const json::Value& config) {
  const std::string dumped = config.dump();
  const std::string h = sha256_hex(dumped.data(), dumped.size());
  return label + "-" + h.substr(0, 12);
}

}  // namespace alsr
