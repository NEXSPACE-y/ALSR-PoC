// Stress / 物理条件のテスト。
//
// ここで固定したい主張は 2 つある（設計書 §2.2 / §17 D）。
//   1. 平均供給帯域が平均消費帯域を下回るなら、有限深度のバッファは必ず枯渇する。
//      深度をいくら増やしても定常状態では消えない。
//   2. 供給が十分で jitter がバッファ余裕の内側なら、ウォームアップ後は枯渇しない。
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "alsr/sim.hpp"
#include "alsr/telemetry.hpp"

using namespace alsr;

namespace {

/// モデル不要の合成 Chunk でスケジューラとキューだけを検証する。
SimConfig synthetic_config(Bytes total, Bytes chunk, std::size_t depth,
                           std::size_t prefetch, double h2d_gbps,
                           double consume_gbps) {
  SimConfig c;
  c.label = "stress";
  c.seed = 20260802;
  c.synthetic_total_bytes = total;
  c.synthetic_chunk_bytes = chunk;
  c.ring_depth = depth;
  c.prefetch_depth = prefetch;
  c.scheduler = "fixed";
  c.fixed.chunk_size = chunk;
  c.fixed.prefetch_depth = prefetch;
  c.backend.h2d.bytes_per_s = h2d_gbps * 1e9;
  c.backend.h2d.fixed_latency_s = 0.0;
  c.backend.h2d.jitter_fraction = 0.0;
  c.backend.prepare.bytes_per_s = 1e12;  // 準備は律速にしない
  c.backend.prepare.fixed_latency_s = 0.0;
  c.backend.consume_bytes_per_s = consume_gbps * 1e9;
  c.backend.verify_checksums = false;  // 大量の Chunk を速く回す
  c.backend.pinned_limit_bytes = 4 * GiB;
  c.backend.device_memory_bytes = 4 * GiB;
  return c;
}

}  // namespace

TEST_CASE("定常帯域不足なら、どの深度でも必ず枯渇する", "[stress][physics]") {
  // 供給 5 GB/s < 消費 10 GB/s。深度を 2..16 まで振っても解決しない。
  for (const std::size_t depth : {std::size_t{2}, std::size_t{3}, std::size_t{4},
                                  std::size_t{8}, std::size_t{16}}) {
    const SimConfig cfg = synthetic_config(
        /*total=*/512 * MiB, /*chunk=*/8 * MiB, depth, depth, 5.0, 10.0);
    const SimResult r = run_simulation(cfg, "starve-" + std::to_string(depth),
                                       nullptr);

    INFO("depth=" << depth << " wait_ratio=" << r.summary.gpu_wait_ratio
                  << " late=" << r.summary.late_wait_ratio);
    REQUIRE(r.summary.starvation_events > 0);
    REQUIRE(r.summary.gpu_wait_ratio > 0.0);
    // 終盤でも待機が消えない = 定常状態で枯渇している
    REQUIRE(r.summary.late_wait_ratio > 0.1);
  }
}

TEST_CASE("帯域不足では深度を増やしても待機率がほぼ改善しない",
          "[stress][physics]") {
  const SimResult shallow = run_simulation(
      synthetic_config(512 * MiB, 8 * MiB, 2, 2, 5.0, 10.0), "d2", nullptr);
  const SimResult deep = run_simulation(
      synthetic_config(512 * MiB, 8 * MiB, 16, 16, 5.0, 10.0), "d16", nullptr);

  INFO("shallow late=" << shallow.summary.late_wait_ratio
                       << " deep late=" << deep.summary.late_wait_ratio);
  // 定常状態の待機率は深度にほとんど依存しない（バッファは平均不足を隠せない）
  REQUIRE(deep.summary.late_wait_ratio > 0.3);
  REQUIRE(shallow.summary.late_wait_ratio > 0.3);
}

TEST_CASE("供給が十分なら、ウォームアップ後は枯渇しない", "[stress][physics]") {
  // 供給 40 GB/s > 消費 10 GB/s、prefetch 3 なので余裕は compute の 2 倍分
  const SimConfig cfg =
      synthetic_config(512 * MiB, 8 * MiB, 4, 3, 40.0, 10.0);
  const SimResult r = run_simulation(cfg, "healthy", nullptr);

  INFO("wait_ratio=" << r.summary.gpu_wait_ratio
                     << " events=" << r.summary.starvation_events);
  REQUIRE(r.summary.starvation_events == 0);
  REQUIRE(r.summary.late_wait_ratio == 0.0);
  REQUIRE(r.summary.copy_compute_overlap_ratio > 0.9);
}

TEST_CASE("jitter がバッファ余裕の内側なら枯渇しない", "[stress][physics]") {
  SimConfig cfg = synthetic_config(512 * MiB, 8 * MiB, 4, 3, 40.0, 10.0);
  cfg.backend.h2d.jitter_fraction = 0.2;  // ±20%
  const SimResult r = run_simulation(cfg, "jitter-ok", nullptr);
  REQUIRE(r.summary.starvation_events == 0);
}

TEST_CASE("jitter がバッファ余裕を超えると枯渇が現れる", "[stress][physics]") {
  // 供給 11 GB/s と消費 10 GB/s は僅差。prefetch=1 で余裕ゼロにし、
  // 大きな jitter を入れると個別 Chunk で待ちが出る。
  SimConfig cfg = synthetic_config(512 * MiB, 8 * MiB, 2, 1, 11.0, 10.0);
  cfg.backend.h2d.jitter_fraction = 0.5;
  const SimResult r = run_simulation(cfg, "jitter-ng", nullptr);
  REQUIRE(r.summary.starvation_events > 0);
}

TEST_CASE("一時停止（OS 揺らぎ）はバッファで吸収され得る", "[stress]") {
  SimConfig cfg = synthetic_config(512 * MiB, 8 * MiB, 8, 6, 40.0, 10.0);
  cfg.backend.h2d.pause_period_s = 0.02;
  cfg.backend.h2d.pause_duration_s = 0.001;
  const SimResult r = run_simulation(cfg, "pause", nullptr);
  // 完全には消えないかもしれないが、待機率は小さいままであること
  REQUIRE(r.summary.gpu_wait_ratio < 0.2);
  REQUIRE(r.summary.checksum_failures == 0);
}

TEST_CASE("小さい Chunk を大量に流しても壊れない（wrap-around 多数）",
          "[stress]") {
  // 64 KiB x 8192 chunk、深度 4 → wrap-around は 8000 回超
  const SimConfig cfg =
      synthetic_config(512 * MiB, 64 * KiB, 4, 3, 40.0, 10.0);
  const SimResult r = run_simulation(cfg, "small-chunks", nullptr);

  REQUIRE(r.summary.chunk_count == 8192);
  REQUIRE(r.summary.ring_wrap_arounds == 8192 - 4);
  REQUIRE(r.summary.order_violations == 0);
  REQUIRE(r.summary.missing_chunks == 0);
}

TEST_CASE("長時間の定常運転で待機が進行性に増えない", "[stress]") {
  // 仮想時間で 600 秒（10 分相当）の定常運転。
  // 消費 10 GB/s x 600 s = 6 TB 相当になるため実バイトは流さない
  // （materialize_data=false）。ここで見たいのはキューと制御則の挙動だけで、
  // その代わりこの Run は必須ゲートを必ず落とす（正確性の証拠にしない）。
  SimConfig cfg = synthetic_config(256 * MiB, 8 * MiB, 4, 3, 40.0, 10.0);
  cfg.repeat_passes = 100000;
  cfg.max_virtual_seconds = 600.0;
  cfg.materialize_data = false;

  const SimResult r = run_simulation(cfg, "steady-600s", nullptr);
  INFO("virtual=" << r.summary.virtual_seconds
                  << " chunks=" << r.summary.chunk_count);
  REQUIRE(r.summary.virtual_seconds >= 600.0);
  REQUIRE(r.summary.starvation_events == 0);
  REQUIRE(r.summary.late_wait_ratio == 0.0);
  REQUIRE(r.summary.order_violations == 0);

  // 実データを流していない Run は必ずゲートで落ちること（黙って通さない）
  RunSummary s = r.summary;
  s.environment_recorded = true;
  evaluate_mandatory_gates(s);
  REQUIRE_FALSE(s.gate_pass);
}

TEST_CASE("適応スケジューラは枯渇時に反応し、振動しない", "[stress][scheduler]") {
  SimConfig cfg = synthetic_config(512 * MiB, 64 * MiB, 8, 2, 6.0, 10.0);
  cfg.scheduler = "adaptive";
  cfg.adaptive.initial_chunk_size = 64 * MiB;
  cfg.adaptive.initial_prefetch_depth = 2;
  cfg.adaptive.max_prefetch_depth = 8;
  cfg.adaptive.min_interval_chunks = 2;
  cfg.adaptive.max_chunk_size = 64 * MiB;   // slot 容量を実測サイズに抑える

  const SimResult r = run_simulation(cfg, "adaptive-starve", nullptr);

  // 反応している
  REQUIRE(r.summary.scheduler_changes > 0);
  // 変更回数が Chunk 数に対して過大でない（ヒステリシスが効いている）
  REQUIRE(r.summary.scheduler_changes <= r.summary.chunk_count / 2 + 1);
  REQUIRE_FALSE(r.scheduler_change_log.empty());
  // それでも物理的な不足は解消できない
  REQUIRE(r.summary.late_wait_ratio > 0.0);
}

TEST_CASE("適応と固定を同条件で比較できる", "[stress][scheduler]") {
  const SimConfig base = synthetic_config(512 * MiB, 32 * MiB, 8, 2, 12.0, 10.0);

  SimConfig fixed = base;
  fixed.scheduler = "fixed";
  SimConfig adaptive = base;
  adaptive.scheduler = "adaptive";
  adaptive.adaptive.initial_chunk_size = 32 * MiB;
  adaptive.adaptive.initial_prefetch_depth = 2;
  adaptive.adaptive.max_prefetch_depth = 8;
  adaptive.adaptive.min_interval_chunks = 4;
  adaptive.adaptive.max_chunk_size = 32 * MiB;

  const SimResult f = run_simulation(fixed, "cmp-fixed", nullptr);
  const SimResult a = run_simulation(adaptive, "cmp-adaptive", nullptr);

  // 適応側は Chunk サイズを変えるので Chunk 数は一致しない。
  // 一致すべきなのは「同じ総バイトを、欠落も順序違反もなく流したこと」。
  REQUIRE(f.summary.total_bytes == a.summary.total_bytes);
  REQUIRE(f.summary.missing_chunks == 0);
  REQUIRE(a.summary.missing_chunks == 0);
  REQUIRE(f.summary.order_violations == 0);
  REQUIRE(a.summary.order_violations == 0);
  // 適応側が悪化していないこと（改善の有無は実験で判断する）
  INFO("fixed=" << f.summary.gpu_wait_ratio
                << " adaptive=" << a.summary.gpu_wait_ratio);
  REQUIRE(a.summary.gpu_wait_ratio <= f.summary.gpu_wait_ratio + 1e-9);
}

TEST_CASE("checksum 検証の有無で結果（仮想時間）が変わらない", "[stress]") {
  SimConfig on = synthetic_config(64 * MiB, 8 * MiB, 4, 3, 40.0, 10.0);
  on.backend.verify_checksums = true;
  SimConfig off = on;
  off.backend.verify_checksums = false;

  const SimResult a = run_simulation(on, "crc-on", nullptr);
  const SimResult b = run_simulation(off, "crc-off", nullptr);

  // 仮想時間は計測オーバーヘッドの影響を受けない（これが仮想時間を使う理由）
  REQUIRE(a.summary.virtual_seconds == b.summary.virtual_seconds);
  REQUIRE(a.summary.checksum_failures == 0);
  // 壁時計は当然変わる（NFR-05: 計測オーバーヘッドを測れること）
  REQUIRE(a.summary.wall_seconds > 0.0);
}

TEST_CASE("深度 2..16 を通しで走らせても不変条件が保たれる", "[stress]") {
  for (std::size_t depth = 2; depth <= 16; depth += 2) {
    const SimConfig cfg =
        synthetic_config(128 * MiB, 4 * MiB, depth, depth - 1, 40.0, 10.0);
    const SimResult r =
        run_simulation(cfg, "depth-" + std::to_string(depth), nullptr);
    INFO("depth=" << depth);
    REQUIRE(r.summary.chunk_count == 32);
    REQUIRE(r.summary.order_violations == 0);
    REQUIRE(r.summary.missing_chunks == 0);
    REQUIRE(r.summary.peak_device_bytes <= 4 * MiB * (depth - 1));
  }
}
