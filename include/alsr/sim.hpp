// ALSR PoC — 仮想時間ストリーミングシミュレータ (Phase 2)。
//
// 目的 (設計書 §11 Phase 2):
//   キューと制御則の検証。CUDA 性能の予測ではない。
//
// なぜ仮想時間か (docs/DECISIONS.md DR-0004):
//   「同一 seed で summary が再現」(§17 完了条件) を満たすには、OS スケジュー
//   リングの揺らぎを結果に混ぜてはならない。時間は決定論的な漸化式で進め、
//   実データの読み出し・checksum 検証だけを実測で行う。壁時計は計測オーバー
//   ヘッドの把握用に別途記録する (NFR-05)。
//
// パイプライン漸化式（3 つの直列エンジンが互いに並行に動く）:
//   prepare_end[i] = max(producer_busy,  slot_released[slot(i)]) + prepare(i)
//   copy_end[i]    = max(transfer_busy,  prepare_end[i])         + h2d(i)
//   compute_end[i] = max(consumer_busy,  copy_end[i])            + compute(i)
//   wait[i]        = max(consumer_busy, copy_end[i]) - consumer_busy
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/host_store.hpp"
#include "alsr/json.hpp"
#include "alsr/planner.hpp"
#include "alsr/scheduler.hpp"
#include "alsr/telemetry.hpp"
#include "alsr/transfer_backend.hpp"

namespace alsr {

struct SimConfig {
  std::string label = "sim";
  u64         seed = 20260802;

  // --- データ源 -------------------------------------------------------
  /// plan.json のパス。空なら synthetic チャンクを生成する。
  std::string   plan_path;
  /// モデルファイル。空なら plan.json の model_path を使う。
  std::string   model_path;
  HostStoreMode host_store_mode = HostStoreMode::Mmap;
  Bytes         synthetic_total_bytes = 4 * GiB;
  Bytes         synthetic_chunk_bytes = 256 * MiB;

  // --- リングバッファ -------------------------------------------------
  std::size_t ring_depth = 4;
  std::size_t prefetch_depth = 2;

  // --- Scheduler ------------------------------------------------------
  std::string             scheduler = "fixed";  ///< fixed | adaptive
  FixedSchedulerConfig    fixed;
  AdaptiveSchedulerConfig adaptive;
  std::size_t             stats_window = 32;

  // --- Backend --------------------------------------------------------
  SimulatorBackendConfig backend;

  // --- 実行条件 -------------------------------------------------------
  /// Chunk 列を何周するか。定常運転（10 分等）はここを増やす。
  u64    repeat_passes = 1;
  /// 仮想時間の上限 [s]。0 で無制限。
  double max_virtual_seconds = 0.0;
  /// 処理する Chunk 数の上限。0 で無制限。
  u64    max_chunks = 0;
  /// S_stream [byte/token]。0 でなければ token/s を導出する。
  Bytes  per_token_bytes = 0;
  /// VRAM へ常駐させる重みのバイト数（設計書 §2.4「大きな重み常駐率」）。
  /// 実行順の先頭から積んでいき、この量に収まる Chunk は「起動時に一度だけ
  /// 載せる」ものとして定常状態のストリーミング対象から外す。
  ///
  /// 注意: これは『重みに割ける VRAM』であって VRAM 総容量ではない。
  /// KV キャッシュ・アクティベーション・CUDA コンテキストは呼び出し側で引く。
  ///
  /// Batch=1 の Dense では各重みが 1 トークンにつき 1 回しか使われないので、
  /// 「どれを常駐させるか」は総転送量に影響しない。効くのはバイト数だけ。
  Bytes  resident_bytes = 0;
  /// 「毎トークン読まれない」Chunk（視覚エンコーダ・投影器など）を定常状態の
  /// ストリーミング対象から外すか。テキスト生成の token/s を測るなら true。
  /// 画像入力を含む経路を測るなら false にして全部流す。
  bool   exclude_non_per_token_chunks = true;
  /// この Chunk 数を処理した時点で cancel_all() する。0 で無効。
  u64    cancel_at_chunk = 0;
  /// 実バイトを Host Store から slot へ流すか。
  /// false にするとスケジューラとキューの挙動だけを長時間回せるが、
  /// 正確性は一切検証していないことになるため必須ゲートは必ず落ちる。
  bool   materialize_data = true;

  [[nodiscard]] json::Value to_json() const;
  static SimConfig from_json(const json::Value& v);
  void validate() const;
};

struct SimResult {
  RunSummary               summary;
  std::vector<ChunkMetric> metrics;
  bool                     cancelled = false;
  /// cancel 後にリングバッファが全 FREE へ戻ったか（FR-05 の安全解放）
  bool                     cancel_released_all_slots = false;
  json::Value              scheduler_config;
  json::Value              backend_config;
  /// Chunk サイズ設定の変遷（振動の確認用）
  std::vector<std::string> scheduler_change_log;
};

/// シミュレーションを実行する。writer が非 null なら 1 Chunk ごとに JSONL 出力。
[[nodiscard]] SimResult run_simulation(const SimConfig& cfg,
                                       const std::string& run_id,
                                       MetricsWriter* writer);

}  // namespace alsr
