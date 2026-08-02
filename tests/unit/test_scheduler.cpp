#include <catch2/catch_test_macros.hpp>

#include "alsr/scheduler.hpp"

using namespace alsr;

namespace {

SchedulerInput healthy_input(u64 idx, Bytes chunk, std::size_t depth) {
  SchedulerInput in;
  in.chunk_index = idx;
  in.queue_depth = depth;
  in.ring_depth = 8;
  in.free_slots = 8 - depth;
  in.p50_compute_ms = 100.0;
  in.p50_copy_ms = 10.0;
  in.p95_copy_ms = 12.0;
  in.recent_copy_cv = 0.05;
  in.wait_ratio = 0.0;
  in.starvation_detected = false;
  in.vram_used_bytes = 1 * GiB;
  in.vram_capacity_bytes = 12 * GiB;
  in.pinned_used_bytes = 1 * GiB;
  in.pinned_limit_bytes = 4 * GiB;
  in.current_chunk_size = chunk;
  in.current_prefetch_depth = depth;
  return in;
}

}  // namespace

TEST_CASE("Scheduler: 階梯の上下移動", "[unit][scheduler]") {
  REQUIRE(next_higher_size(16 * MiB) == 32 * MiB);
  REQUIRE(next_higher_size(256 * MiB) == 512 * MiB);
  REQUIRE(next_higher_size(1024 * MiB) == 1024 * MiB);  // 端で止まる
  REQUIRE(next_lower_size(1024 * MiB) == 512 * MiB);
  REQUIRE(next_lower_size(16 * MiB) == 16 * MiB);
  // 階梯上にない値でも近い段へ動く
  REQUIRE(next_higher_size(100 * MiB) == 128 * MiB);
  REQUIRE(next_lower_size(100 * MiB) == 64 * MiB);
}

TEST_CASE("Scheduler: fixed は何があっても動かない", "[unit][scheduler]") {
  FixedSchedulerConfig cfg;
  cfg.chunk_size = 128 * MiB;
  cfg.prefetch_depth = 3;
  auto s = make_fixed_scheduler(cfg);

  for (u64 i = 0; i < 100; ++i) {
    SchedulerInput in = healthy_input(i, 128 * MiB, 3);
    in.starvation_detected = (i % 2 == 0);
    in.wait_ratio = 0.5;
    const SchedulerDecision d = s->decide(in);
    REQUIRE(d.chunk_size == 128 * MiB);
    REQUIRE(d.prefetch_depth == 3);
    REQUIRE_FALSE(d.changed);
  }
  REQUIRE(s->change_count() == 0);
}

TEST_CASE("Scheduler: 枯渇したら先読みを深くし Chunk を細かくする",
          "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.initial_chunk_size = 256 * MiB;
  cfg.initial_prefetch_depth = 2;
  cfg.min_interval_chunks = 4;
  auto s = make_adaptive_scheduler(cfg);

  SchedulerInput in = healthy_input(10, 256 * MiB, 0);
  in.starvation_detected = true;
  in.wait_ratio = 0.4;

  const SchedulerDecision d = s->decide(in);
  REQUIRE(d.changed);
  REQUIRE(d.prefetch_depth == 3);
  REQUIRE(d.chunk_size == 128 * MiB);
  REQUIRE(d.transfer_priority == 1);
  REQUIRE_FALSE(d.reason.empty());
  REQUIRE(s->change_count() == 1);
}

TEST_CASE("Scheduler: ヒステリシスで連続変更を抑える（振動対策）",
          "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.initial_chunk_size = 256 * MiB;
  cfg.initial_prefetch_depth = 2;
  cfg.min_interval_chunks = 10;
  auto s = make_adaptive_scheduler(cfg);

  SchedulerInput in = healthy_input(0, 256 * MiB, 0);
  in.starvation_detected = true;
  in.wait_ratio = 0.4;

  REQUIRE(s->decide(in).changed);
  // 直後の 9 chunk では変更しない
  for (u64 i = 1; i < 10; ++i) {
    in.chunk_index = i;
    const SchedulerDecision d = s->decide(in);
    REQUIRE_FALSE(d.changed);
    REQUIRE(d.action == "emergency");  // 転送優先度だけは上げる
  }
  in.chunk_index = 10;
  REQUIRE(s->decide(in).changed);
  REQUIRE(s->change_count() == 2);
}

TEST_CASE("Scheduler: 余裕があれば Chunk を大きくする", "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.initial_chunk_size = 64 * MiB;
  cfg.initial_prefetch_depth = 3;
  cfg.min_interval_chunks = 1;
  auto s = make_adaptive_scheduler(cfg);

  SchedulerInput in = healthy_input(100, 64 * MiB, 3);
  const SchedulerDecision d = s->decide(in);
  REQUIRE(d.changed);
  REQUIRE(d.action == "grow_chunk");
  REQUIRE(d.chunk_size == 128 * MiB);
}

TEST_CASE("Scheduler: メモリ圧迫では先読みを浅くする", "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.initial_chunk_size = 256 * MiB;
  cfg.initial_prefetch_depth = 4;
  cfg.min_interval_chunks = 1;
  auto s = make_adaptive_scheduler(cfg);

  SchedulerInput in = healthy_input(50, 256 * MiB, 4);
  in.vram_used_bytes = 11 * GiB;  // 12GiB の 91%
  const SchedulerDecision d = s->decide(in);
  REQUIRE(d.changed);
  REQUIRE(d.action == "shallow");
  REQUIRE(d.prefetch_depth == 3);
}

TEST_CASE("Scheduler: 上下限を越えない", "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.initial_chunk_size = 16 * MiB;
  cfg.initial_prefetch_depth = 1;
  cfg.min_prefetch_depth = 1;
  cfg.max_prefetch_depth = 2;
  cfg.min_chunk_size = 16 * MiB;
  cfg.min_interval_chunks = 1;
  auto s = make_adaptive_scheduler(cfg);

  SchedulerInput in = healthy_input(10, 16 * MiB, 0);
  in.starvation_detected = true;
  in.wait_ratio = 0.9;

  for (u64 i = 10; i < 30; ++i) {
    in.chunk_index = i;
    const SchedulerDecision d = s->decide(in);
    REQUIRE(d.prefetch_depth >= 1);
    REQUIRE(d.prefetch_depth <= 2);
    REQUIRE(d.chunk_size >= 16 * MiB);
  }
}

TEST_CASE("Scheduler: chunk サイズ変更を禁止できる（実 plan 使用時）",
          "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.initial_chunk_size = 64 * MiB;
  cfg.initial_prefetch_depth = 3;
  cfg.min_interval_chunks = 1;
  cfg.allow_chunk_size_change = false;
  auto s = make_adaptive_scheduler(cfg);

  for (u64 i = 0; i < 20; ++i) {
    SchedulerInput in = healthy_input(i, 64 * MiB, 3);
    const SchedulerDecision d = s->decide(in);
    REQUIRE(d.chunk_size == 64 * MiB);
  }
}

TEST_CASE("Scheduler: 不正な設定を拒否する", "[unit][scheduler]") {
  AdaptiveSchedulerConfig cfg;
  cfg.min_prefetch_depth = 0;
  REQUIRE_THROWS_AS(make_adaptive_scheduler(cfg), Error);

  AdaptiveSchedulerConfig cfg2;
  cfg2.min_prefetch_depth = 5;
  cfg2.max_prefetch_depth = 2;
  REQUIRE_THROWS_AS(make_adaptive_scheduler(cfg2), Error);
}

TEST_CASE("Scheduler: config_json に制御パラメータが残る", "[unit][scheduler]") {
  auto s = make_adaptive_scheduler(AdaptiveSchedulerConfig{});
  const json::Value j = s->config_json();
  REQUIRE(j.at("type").as_string() == "adaptive");
  REQUIRE(j.contains("min_interval_chunks"));
  REQUIRE(j.contains("starvation_wait_ratio"));
}
