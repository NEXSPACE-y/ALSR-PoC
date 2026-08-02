// ALSR PoC — Scheduler (FR-06)。
//
//   - 静的 / 適応型の両モードを持つ
//   - 入力: queue depth, GPU 処理時間, H2D 時間, 待機率, VRAM 余量, 直近の分散
//   - 出力: 次 Chunk サイズ, 先読み距離, 転送優先度
//   - 初期実装は決定論的な制御則。機械学習は使わない。
//   - 制御変更の理由をログへ残す（FR-06 / §8.3）
//
// 振動対策 (§14 リスク表「Scheduler振動」):
//   - 変更間隔の下限 (min_interval_chunks)
//   - 直前の変更と逆向きの変更に対するクールダウン
//   - 判定に p50/p95 を使い、単発の外れ値では動かない
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/json.hpp"

namespace alsr {

/// Chunk サイズの階梯（設計書 §11 Phase 3 のスイープ範囲に合わせる）。
[[nodiscard]] const std::vector<Bytes>& chunk_size_ladder();

struct SchedulerInput {
  u64         chunk_index = 0;
  std::size_t queue_depth = 0;        ///< 現在の in-flight スロット数
  std::size_t ring_depth = 0;
  std::size_t free_slots = 0;

  double p50_compute_ms = 0.0;
  double p50_copy_ms = 0.0;
  double p95_copy_ms = 0.0;
  double recent_copy_cv = 0.0;        ///< 直近 copy 時間の変動係数（分散の指標）
  double wait_ratio = 0.0;            ///< 直近窓での GPU 待機率 [0,1]
  bool   starvation_detected = false;

  Bytes vram_used_bytes = 0;
  Bytes vram_capacity_bytes = 0;
  Bytes pinned_used_bytes = 0;
  Bytes pinned_limit_bytes = 0;

  Bytes current_chunk_size = 0;
  std::size_t current_prefetch_depth = 0;
};

struct SchedulerDecision {
  Bytes       chunk_size = 0;
  std::size_t prefetch_depth = 0;
  int         transfer_priority = 0;  ///< 0=通常, 1=緊急（枯渇時）
  std::string action = "keep";        ///< keep | grow_chunk | shrink_chunk |
                                      ///< deepen | shallow | emergency
  std::string reason;                 ///< 人間が読む理由。ログへ残す。
  bool        changed = false;
};

class Scheduler {
public:
  virtual ~Scheduler() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  virtual SchedulerDecision decide(const SchedulerInput& in) = 0;
  [[nodiscard]] virtual u64 change_count() const noexcept = 0;
  [[nodiscard]] virtual json::Value config_json() const = 0;
};

struct FixedSchedulerConfig {
  Bytes       chunk_size = 256 * MiB;
  std::size_t prefetch_depth = 2;
};

struct AdaptiveSchedulerConfig {
  Bytes       initial_chunk_size = 256 * MiB;
  std::size_t initial_prefetch_depth = 2;
  std::size_t min_prefetch_depth = 1;
  std::size_t max_prefetch_depth = 8;
  Bytes       min_chunk_size = 16 * MiB;
  Bytes       max_chunk_size = 1024 * MiB;

  /// 変更を許すまでの最小 Chunk 間隔（ヒステリシス）
  u64    min_interval_chunks = 8;
  /// この待機率を超えたら枯渇とみなす
  double starvation_wait_ratio = 0.02;
  /// p95_copy < grow_ratio * p50_compute なら Chunk を大きくしてよい
  double grow_ratio = 0.5;
  /// VRAM / pinned がこの割合を超えたら圧迫とみなす
  double pressure_ratio = 0.9;
  /// chunk_size は可変か（実 plan を使う Run では固定される）
  bool allow_chunk_size_change = true;
};

[[nodiscard]] std::unique_ptr<Scheduler> make_fixed_scheduler(
    const FixedSchedulerConfig& cfg);
[[nodiscard]] std::unique_ptr<Scheduler> make_adaptive_scheduler(
    const AdaptiveSchedulerConfig& cfg);

/// 階梯上で 1 段上/下のサイズ。端では同じ値を返す。
[[nodiscard]] Bytes next_higher_size(Bytes current);
[[nodiscard]] Bytes next_lower_size(Bytes current);

}  // namespace alsr
