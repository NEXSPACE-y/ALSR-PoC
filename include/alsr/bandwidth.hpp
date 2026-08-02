// ALSR PoC — Bandwidth Emulator (FR-07)。
//
//   - 帯域、固定レイテンシ、jitter、burst、一時停止を設定できる
//   - 5, 10, 15, 20, 25, 28 GB/s 等を再現できる
//   - M5 上で CUDA の成立を主張しない。スケジューラの論理試験にのみ使う。
//
// 決定論:
//   乱数は seed 固定の mt19937_64。呼び出し順が同じなら同じ系列になる。
//   これにより「同一 seed で summary が再現」を満たす。
#pragma once

#include <random>
#include <string>

#include "alsr/common.hpp"
#include "alsr/json.hpp"

namespace alsr {

struct BandwidthProfile {
  /// 持続帯域 [byte/s]。0 は不許可。
  double bytes_per_s = 25e9;
  /// 転送 1 回ごとの固定レイテンシ [s]（発行・同期のオーバーヘッド）
  double fixed_latency_s = 20e-6;
  /// 転送時間へ乗る一様乱数の幅。0.2 なら ±20%。
  double jitter_fraction = 0.0;
  /// 一時停止の周期 [s]。0 で無効。OS 揺らぎ・他プロセス干渉の模擬。
  double pause_period_s = 0.0;
  /// 一時停止の長さ [s]
  double pause_duration_s = 0.0;

  [[nodiscard]] json::Value to_json() const;
  static BandwidthProfile from_json(const json::Value& v);
  void validate() const;
};

class BandwidthModel {
public:
  BandwidthModel(BandwidthProfile profile, u64 seed);

  /// n バイト転送に要する仮想時間 [s]。
  /// now は転送開始時刻。pause 判定に使う（時刻依存を持たせるため）。
  [[nodiscard]] double transfer_time_s(Bytes n, double now_s);

  [[nodiscard]] const BandwidthProfile& profile() const noexcept { return profile_; }
  [[nodiscard]] u64 pause_hits() const noexcept { return pause_hits_; }
  /// jitter を無視した理想時間（オーバーヘッド比の算出に使う）。
  [[nodiscard]] double ideal_time_s(Bytes n) const noexcept {
    return static_cast<double>(n) / profile_.bytes_per_s;
  }

private:
  BandwidthProfile      profile_;
  std::mt19937_64       rng_;
  u64                   pause_hits_ = 0;
  double                next_pause_at_s_ = 0.0;
};

}  // namespace alsr
