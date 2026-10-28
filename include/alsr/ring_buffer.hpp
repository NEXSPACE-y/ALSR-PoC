// ALSR PoC — Queue / Ring Buffer (FR-05)。
//
// 要件:
//   - EMPTY → PREPARING → READY → COPYING → RESIDENT → IN_USE →
//     RETIRED → EMPTY の
//     状態遷移を持つ
//   - 単一所有権または世代番号により ABA 問題を防ぐ
//   - producer が consumer を上書きしない
//   - キャンセル時に全 slot を安全に解放する
//
// 設計:
//   スロットは SlotRef{slot_id, generation} 経由でのみ触る。generation は
//   スロットが FREE へ戻るたび、および cancel 時に増える。古い SlotRef で
//   触ろうとすると Error(State) になる（ABA 検出）。
//   FREE スロットの取得は acquire_free() のみ。producer は取得できた
//   スロットしか書けないので、CONSUMING 中のスロットを踏むことはない。
#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "alsr/common.hpp"

namespace alsr {

enum class SlotState {
  Empty = 0,
  Preparing,
  Ready,
  Copying,
  Resident,
  InUse,
  Retired,
  Error,
};

std::string_view to_string(SlotState s) noexcept;

/// 世代付きハンドル。値で持ち回る。
struct SlotRef {
  std::size_t slot_id = 0;
  u64         generation = 0;

  friend bool operator==(const SlotRef& a, const SlotRef& b) noexcept {
    return a.slot_id == b.slot_id && a.generation == b.generation;
  }
};

struct BufferSlot {
  std::size_t slot_id = 0;
  u64         generation = 0;
  SlotState   state = SlotState::Empty;

  // ペイロード
  u64   chunk_id = 0;
  Bytes source_offset = 0;
  Bytes payload_bytes = 0;
  u32   checksum = 0;
  bool  checksum_expected_known = false;
  u32   checksum_expected = 0;

  // 仮想時刻（秒）
  double fill_start_s = 0.0;
  double fill_end_s = 0.0;
  double copy_start_s = 0.0;
  double copy_end_s = 0.0;
  double compute_start_s = 0.0;
  double compute_end_s = 0.0;

  // 直前にこのスロットを使った Chunk の消費完了時刻（スロット解放時刻）
  double released_at_s = 0.0;

  [[nodiscard]] SlotRef ref() const noexcept { return SlotRef{slot_id, generation}; }
};

struct RingBufferStats {
  std::size_t depth = 0;
  Bytes       slot_capacity = 0;
  u64         acquisitions = 0;      ///< acquire_free() の成功回数
  u64         wrap_arounds = 0;      ///< スロットが再利用された回数
  u64         cancellations = 0;
  u64         illegal_transition_attempts = 0;
  std::size_t peak_in_flight = 0;    ///< 同時に Free 以外だったスロット数の最大
};

class RingBuffer {
public:
  RingBuffer(std::size_t depth, Bytes slot_capacity);

  [[nodiscard]] std::size_t depth() const noexcept { return slots_.size(); }
  [[nodiscard]] Bytes slot_capacity() const noexcept { return slot_capacity_; }
  [[nodiscard]] std::size_t free_count() const noexcept { return free_ids_.size(); }
  [[nodiscard]] std::size_t in_flight() const noexcept {
    return slots_.size() - free_ids_.size();
  }
  [[nodiscard]] const RingBufferStats& stats() const noexcept { return stats_; }

  /// EMPTY のスロットを 1 つ確保し PREPARING にする。無ければ std::nullopt 相当で
  /// generation==0 かつ slot_id==npos の SlotRef を返さず、has_free() で事前確認すること。
  [[nodiscard]] bool has_free() const noexcept { return !free_ids_.empty(); }

  /// EMPTY → PREPARING。EMPTY が無ければ Error(State)。
  SlotRef acquire_free();

  /// 状態遷移。from が現在状態と違う、または遷移表に無ければ Error(State)。
  void transition(const SlotRef& ref, SlotState from, SlotState to);

  /// 書き込み用アクセス。世代不一致は Error(State)。
  BufferSlot&       slot(const SlotRef& ref);
  const BufferSlot& slot(const SlotRef& ref) const;

  /// ホスト側バッファ（実データ）。capacity バイト。
  u8* host_buffer(const SlotRef& ref);

  /// 全スロットを EMPTY へ戻し、世代を進める。既存 SlotRef はすべて無効になる。
  void cancel_all();

  /// 状態別のスロット数（テスト・診断用）。
  [[nodiscard]] std::vector<std::size_t> state_histogram() const;

private:
  void check_ref(const SlotRef& ref) const;

  std::vector<BufferSlot>      slots_;
  std::vector<std::vector<u8>> buffers_;
  std::deque<std::size_t>      free_ids_;
  Bytes                        slot_capacity_ = 0;
  RingBufferStats              stats_;
};

}  // namespace alsr
