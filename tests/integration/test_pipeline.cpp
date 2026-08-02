#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <string>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/gguf.hpp"
#include "alsr/host_store.hpp"
#include "alsr/manifest.hpp"
#include "alsr/planner.hpp"
#include "alsr/sim.hpp"
#include "alsr/telemetry.hpp"
#include "test_helpers.hpp"

using namespace alsr;
using namespace alsr::test;

namespace {

std::vector<TensorSpec> small_model() {
  std::vector<TensorSpec> t;
  t.push_back({"output_norm.weight", 0, {3840}});
  t.push_back({"token_embd.weight", 14, {3840, 512}});
  for (int layer = 0; layer < 3; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    t.push_back({p + "attn_norm.weight", 0, {3840}});
    t.push_back({p + "attn_q.weight", 12, {3840, 256}});
    t.push_back({p + "attn_k.weight", 12, {3840, 128}});
    t.push_back({p + "attn_v.weight", 12, {3840, 128}});
    t.push_back({p + "attn_output.weight", 12, {3840, 256}});
    t.push_back({p + "ffn_gate.weight", 12, {3840, 512}});
    t.push_back({p + "ffn_up.weight", 12, {3840, 512}});
    t.push_back({p + "ffn_down.weight", 14, {3840, 512}});
  }
  return t;
}

struct Built {
  TempDir     tmp;
  std::string model;
  std::string plan_path;
  GgufFile    gguf;
  Manifest    manifest;
  ChunkPlan   plan;

  explicit Built(const std::string& tag, Bytes chunk_bytes = 2 * MiB)
      : tmp(tag) {
    model = tmp.file("model.gguf");
    write_test_gguf(model, small_model());
    gguf = read_gguf(model);
    manifest = build_manifest(gguf);

    PlanOptions opt;
    opt.mode = ChunkMode::Fixed;
    opt.target_bytes = chunk_bytes;
    opt.compute_checksums = true;
    plan = build_plan(gguf, manifest, opt);
    verify_plan(gguf, plan);

    plan_path = tmp.file("plan.json");
    std::ofstream out(plan_path, std::ios::trunc);
    out << to_json(plan).dump(2);
  }
};

/// 供給が消費を十分上回る設定。
SimConfig healthy_config(const Built& b) {
  SimConfig c;
  c.label = "it";
  c.plan_path = b.plan_path;
  c.model_path = b.model;
  c.host_store_mode = HostStoreMode::Mmap;
  c.ring_depth = 4;
  c.prefetch_depth = 3;
  c.scheduler = "fixed";
  c.fixed.chunk_size = 2 * MiB;
  c.fixed.prefetch_depth = 3;
  c.backend.h2d.bytes_per_s = 40e9;
  c.backend.h2d.fixed_latency_s = 0.0;
  c.backend.prepare.bytes_per_s = 40e9;
  c.backend.prepare.fixed_latency_s = 0.0;
  c.backend.consume_bytes_per_s = 10e9;
  c.backend.verify_checksums = true;
  c.backend.pinned_limit_bytes = 1 * GiB;
  c.backend.device_memory_bytes = 12 * GiB;
  return c;
}

}  // namespace

TEST_CASE("統合: Inspector → Planner → Host Store が一貫している",
          "[integration]") {
  Built b("it-inspect");

  // Planner が主張する範囲を Host Store で実際に読めて、checksum も一致する
  HostStore store(b.model, HostStoreMode::Mmap);
  REQUIRE(store.file_size() == b.gguf.file_size);

  std::vector<u8> buf;
  Bytes total = 0;
  for (const Chunk& c : b.plan.chunks) {
    buf.resize(static_cast<std::size_t>(c.bytes));
    store.read_into(c.source_offset, c.bytes, buf.data());
    REQUIRE(crc32c(buf.data(), buf.size()) == c.crc32c);
    total += c.bytes;
  }
  REQUIRE(total == b.gguf.file_size - b.gguf.data_start);
}

TEST_CASE("統合: Producer → Queue → Transfer → Consumer が全 Chunk を通す",
          "[integration]") {
  Built b("it-pipeline");
  const SimConfig cfg = healthy_config(b);
  const SimResult r = run_simulation(cfg, "run-pipeline", nullptr);

  REQUIRE(r.summary.chunk_count == b.plan.chunks.size());
  REQUIRE(r.summary.total_bytes == b.plan.covered_bytes);
  REQUIRE(r.summary.checksum_failures == 0);
  REQUIRE(r.summary.order_violations == 0);
  REQUIRE(r.summary.missing_chunks == 0);

  RunSummary s = r.summary;
  s.environment_recorded = true;
  evaluate_mandatory_gates(s);
  REQUIRE(s.gate_pass);
}

TEST_CASE("統合: 同一 seed / 同一設定で summary が完全再現する",
          "[integration]") {
  Built b("it-repro");
  const SimConfig cfg = healthy_config(b);

  const SimResult a = run_simulation(cfg, "run-repro", nullptr);
  const SimResult c = run_simulation(cfg, "run-repro", nullptr);

  // wall_seconds は実測値なので一致しない。それ以外は完全一致すること。
  // 「仮想時間は再現する / 壁時計は再現しない」を分離して測れること自体が
  // 仮想時間を採用した理由 (DR-0004 / NFR-05)。
  RunSummary sa = a.summary;
  RunSummary sc = c.summary;
  REQUIRE(sa.wall_seconds > 0.0);
  REQUIRE(sc.wall_seconds > 0.0);
  sa.wall_seconds = 0.0;
  sc.wall_seconds = 0.0;
  REQUIRE(sa.to_json().dump() == sc.to_json().dump());
  REQUIRE(a.metrics.size() == c.metrics.size());
  for (std::size_t i = 0; i < a.metrics.size(); ++i) {
    REQUIRE(a.metrics[i].h2d_ms == c.metrics[i].h2d_ms);
    REQUIRE(a.metrics[i].wait_ms == c.metrics[i].wait_ms);
  }

  // seed が変われば（jitter がある設定では）結果も変わり得る
  SimConfig jitter = cfg;
  jitter.backend.h2d.jitter_fraction = 0.2;
  jitter.seed = 1;
  const SimResult d = run_simulation(jitter, "run-j1", nullptr);
  jitter.seed = 2;
  const SimResult e = run_simulation(jitter, "run-j2", nullptr);
  REQUIRE(d.summary.virtual_seconds != e.summary.virtual_seconds);
}

TEST_CASE("統合: cancellation で全 slot が安全に解放される", "[integration]") {
  Built b("it-cancel");
  SimConfig cfg = healthy_config(b);
  cfg.cancel_at_chunk = 4;

  const SimResult r = run_simulation(cfg, "run-cancel", nullptr);
  REQUIRE(r.cancelled);
  REQUIRE(r.cancel_released_all_slots);
  // 中断したので未消費 Chunk が残る = ゲートは落ちる（これが正しい挙動）
  REQUIRE(r.summary.missing_chunks > 0);

  RunSummary s = r.summary;
  s.environment_recorded = true;
  evaluate_mandatory_gates(s);
  REQUIRE_FALSE(s.gate_pass);
}

TEST_CASE("統合: timeout（仮想時間上限）で打ち切れる", "[integration]") {
  Built b("it-timeout");
  SimConfig cfg = healthy_config(b);
  cfg.repeat_passes = 1000;
  cfg.max_virtual_seconds = 0.05;

  const SimResult r = run_simulation(cfg, "run-timeout", nullptr);
  REQUIRE(r.summary.chunk_count > 0);
  REQUIRE(r.summary.virtual_seconds < 1.0);
  // 上限が無ければ 1000 周するはずのところで止まっている
  REQUIRE(r.summary.chunk_count < b.plan.chunks.size() * 1000);
}

TEST_CASE("統合: max_chunks で打ち切れる", "[integration]") {
  Built b("it-maxchunks");
  SimConfig cfg = healthy_config(b);
  cfg.repeat_passes = 100;
  cfg.max_chunks = 7;

  const SimResult r = run_simulation(cfg, "run-maxchunks", nullptr);
  REQUIRE(r.summary.chunk_count == 7);
}

TEST_CASE("統合: OOM 模擬 — pinned 上限超過は起動時に止まる", "[integration]") {
  Built b("it-oom");
  SimConfig cfg = healthy_config(b);
  cfg.ring_depth = 8;
  cfg.prefetch_depth = 8;
  cfg.backend.pinned_limit_bytes = 1024;  // 明らかに足りない

  try {
    (void)run_simulation(cfg, "run-oom", nullptr);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Invalid);
    REQUIRE(e.raw().find("pinned") != std::string::npos);
  }
}

TEST_CASE("統合: VRAM 上限超過も起動時に止まる", "[integration]") {
  Built b("it-vram");
  SimConfig cfg = healthy_config(b);
  cfg.ring_depth = 4;
  cfg.prefetch_depth = 4;
  cfg.backend.pinned_limit_bytes = 0;      // 無制限
  cfg.backend.device_memory_bytes = 1024;  // 足りない
  REQUIRE_THROWS_AS(run_simulation(cfg, "run-vram", nullptr), Error);
}

TEST_CASE("統合: 壊れたモデルは checksum で検出される", "[integration]") {
  Built b("it-corrupt");
  SimConfig cfg = healthy_config(b);

  // 計画を作った後にデータ領域を書き換える（テスト用の合成ファイル）
  {
    std::fstream f(b.model, std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(f.good());
    f.seekp(static_cast<std::streamoff>(b.gguf.data_start + 1024));
    const char poison = static_cast<char>(0xFF);
    f.write(&poison, 1);
    f.flush();
  }

  const SimResult r = run_simulation(cfg, "run-corrupt", nullptr);
  REQUIRE(r.summary.checksum_failures > 0);

  RunSummary s = r.summary;
  s.environment_recorded = true;
  evaluate_mandatory_gates(s);
  REQUIRE_FALSE(s.gate_pass);
}

TEST_CASE("統合: telemetry writer が全 Chunk 分を残す", "[integration]") {
  Built b("it-telemetry");
  const SimConfig cfg = healthy_config(b);
  TempDir out("it-telemetry-out");
  const std::string path = out.file("metrics.jsonl");

  SimResult r;
  {
    MetricsWriter w(path);
    r = run_simulation(cfg, "run-tel", &w);
    REQUIRE(w.lines_written() == r.summary.chunk_count);
  }

  std::ifstream in(path);
  std::string line;
  std::size_t n = 0;
  while (std::getline(in, line)) {
    const json::Value v = json::Value::parse(line);
    REQUIRE(v.at("run_id").as_string() == "run-tel");
    REQUIRE(v.at("bytes").as_int() > 0);
    REQUIRE(v.at("checksum_ok").as_bool());
    ++n;
  }
  REQUIRE(n == r.summary.chunk_count);

  // §16.4: raw metrics のハッシュが取れる
  REQUIRE(sha256_file(path).size() == 64);
}

TEST_CASE("統合: mmap と read で結果が一致する", "[integration]") {
  Built b("it-modes");
  SimConfig m = healthy_config(b);
  m.host_store_mode = HostStoreMode::Mmap;
  SimConfig r = healthy_config(b);
  r.host_store_mode = HostStoreMode::Read;

  const SimResult a = run_simulation(m, "run-mmap", nullptr);
  const SimResult c = run_simulation(r, "run-read", nullptr);
  REQUIRE(a.summary.checksum_failures == 0);
  REQUIRE(c.summary.checksum_failures == 0);
  REQUIRE(a.summary.total_bytes == c.summary.total_bytes);
  REQUIRE(a.summary.virtual_seconds == c.summary.virtual_seconds);
}

TEST_CASE("統合: 設定の JSON 往復", "[integration]") {
  Built b("it-config");
  const SimConfig cfg = healthy_config(b);
  const json::Value j = cfg.to_json();
  const SimConfig back = SimConfig::from_json(j);
  REQUIRE(back.to_json().dump() == j.dump());

  // 不正設定は読み込み時に落ちる
  json::Value bad = j;
  json::Value ring = json::Value::make_object();
  ring.set("ring_depth", 2);
  ring.set("prefetch_depth", 5);
  bad.set("ring", ring);
  REQUIRE_THROWS_AS(SimConfig::from_json(bad), Error);
}

TEST_CASE("統合: resident_bytes は定常状態の転送量を減らす", "[integration]") {
  Built b("it-resident");

  const SimConfig base = healthy_config(b);
  const SimResult full = run_simulation(base, "run-res-0", nullptr);

  // 総バイトのおよそ半分を常駐させる
  SimConfig half = base;
  half.resident_bytes = full.summary.total_bytes / 2;
  const SimResult r = run_simulation(half, "run-res-half", nullptr);

  REQUIRE(r.summary.resident_bytes > 0);
  REQUIRE(r.summary.resident_chunk_count > 0);
  REQUIRE(r.summary.resident_bytes <= half.resident_bytes);
  // 常駐分は流れない
  REQUIRE(r.summary.total_bytes ==
          full.summary.total_bytes - r.summary.resident_bytes);
  REQUIRE(r.summary.checksum_failures == 0);
  REQUIRE(r.summary.order_violations == 0);
  REQUIRE(r.summary.chunk_count < full.summary.chunk_count);
}

TEST_CASE("統合: 全部常駐させようとしたら明示的に失敗する", "[integration]") {
  Built b("it-resident-all");
  SimConfig cfg = healthy_config(b);
  cfg.resident_bytes = 1ull << 40;  // 明らかに全部載る

  try {
    (void)run_simulation(cfg, "run-res-all", nullptr);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Invalid);
    REQUIRE(e.raw().find("ALSR が不要") != std::string::npos);
  }
}

TEST_CASE("統合: 合成 Chunk では resident_bytes を拒否する", "[integration]") {
  SimConfig cfg;
  cfg.plan_path = "";              // synthetic
  cfg.synthetic_total_bytes = 64 * MiB;
  cfg.synthetic_chunk_bytes = 8 * MiB;
  cfg.fixed.chunk_size = 8 * MiB;
  cfg.ring_depth = 4;
  cfg.prefetch_depth = 3;
  cfg.resident_bytes = 16 * MiB;
  cfg.backend.pinned_limit_bytes = 1 * GiB;
  cfg.backend.device_memory_bytes = 1 * GiB;

  REQUIRE_THROWS_AS(run_simulation(cfg, "run-res-synth", nullptr), Error);
}
