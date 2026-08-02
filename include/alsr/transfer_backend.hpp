// ALSR PoC — Transfer Backend 抽象 (NFR-04)。
//
//   「M5 の simulation backend と CUDA backend が同一 scheduler interface を
//     使う」ため、Host→Device 転送はこのインタフェース越しにのみ行う。
//   Phase 3 で backends/cuda/ が同じインタフェースを実装する。
//
// 仮想時間について:
//   simulator backend は実測時間ではなく仮想時間を返す。これは M5 に
//   独立 VRAM への PCIe H2D 経路が無く、実測しても CUDA の予測にならない
//   ためである（設計書 §3.1）。CUDA backend は CUDA Event の実測値を返す。
#pragma once

#include <memory>
#include <string>

#include "alsr/bandwidth.hpp"
#include "alsr/common.hpp"
#include "alsr/json.hpp"

namespace alsr {

struct TransferCapabilities {
  bool  device_overlap = false;      ///< copy と kernel を重ねられるか
  int   async_engine_count = 0;      ///< コピーエンジン数
  bool  pinned_memory_supported = false;
  bool  measured_on_real_hardware = false;  ///< false なら性能主張に使わない
  Bytes pinned_limit_bytes = 0;
  Bytes device_memory_bytes = 0;
};

class TransferBackend {
public:
  virtual ~TransferBackend() = default;

  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual TransferCapabilities capabilities() const = 0;

  /// n バイトの H2D にかかる時間 [s]。now_s は開始時刻（仮想）。
  [[nodiscard]] virtual double h2d_time_s(Bytes n, double now_s) = 0;
  /// ホスト側で Chunk を準備する時間 [s]（RAM 読み出し・再配置）。
  [[nodiscard]] virtual double prepare_time_s(Bytes n, double now_s) = 0;
  /// GPU が n バイト分の重みを消費する時間 [s]。
  [[nodiscard]] virtual double compute_time_s(Bytes n, double now_s) = 0;

  /// 実データを「デバイス側」へ渡す。simulator では checksum 検証のみ行う。
  virtual u32 consume(const u8* host_data, Bytes n) = 0;

  [[nodiscard]] virtual json::Value config_json() const = 0;
};

struct SimulatorBackendConfig {
  BandwidthProfile h2d;         ///< RAM → VRAM
  BandwidthProfile prepare;     ///< Host Store → pinned slab
  /// GPU 消費速度 [byte/s]。S_stream 相当の重みを何 byte/s で食えるか。
  double consume_bytes_per_s = 30e9;
  double consume_jitter_fraction = 0.0;
  Bytes  device_memory_bytes = 12ull * GiB;   ///< RTX 3060 12GB を模した既定
  Bytes  pinned_limit_bytes = 4ull * GiB;
  bool   verify_checksums = true;
  u64    seed = 20260802;
};

[[nodiscard]] std::unique_ptr<TransferBackend> make_simulator_backend(
    const SimulatorBackendConfig& cfg);

}  // namespace alsr
