// ALSR PoC — チェックサム。
//
//  - CRC-32C (Castagnoli, 反転多項式 0x82F63B78)
//      Chunk 完全性検証用 (FR-03 / NFR-01)。ホットパスで毎 Chunk 走るため
//      速度が要る。ARMv8 CRC32 命令があれば使い、無ければ slicing-by-8。
//      両経路が同一値を返すことをテストで固定する。
//  - SHA-256
//      証拠管理用 (§16.4)。raw metrics のハッシュ、モデル digest 検証。
//      速度より「外部ツール(shasum -a 256)と一致すること」が重要。
#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "alsr/common.hpp"

namespace alsr {

// ---------------------------------------------------------------------------
// CRC-32C
// ---------------------------------------------------------------------------

/// 逐次計算。分割されたバッファをまたいで更新できる。
class Crc32c {
public:
  Crc32c() = default;
  void   update(const void* data, std::size_t len) noexcept;
  [[nodiscard]] u32 value() const noexcept { return ~state_; }
  void   reset() noexcept { state_ = 0xFFFFFFFFu; }

private:
  u32 state_ = 0xFFFFFFFFu;
};

[[nodiscard]] u32 crc32c(const void* data, std::size_t len) noexcept;

/// ハードウェア命令を使わない参照実装。テストでの相互検証用に公開する。
[[nodiscard]] u32 crc32c_software(const void* data, std::size_t len) noexcept;

/// ビルド構成でハードウェア CRC32C 経路が有効か。environment.json へ記録する。
[[nodiscard]] bool crc32c_hardware_enabled() noexcept;

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

class Sha256 {
public:
  Sha256() { reset(); }
  void reset() noexcept;
  void update(const void* data, std::size_t len) noexcept;
  /// 呼び出し後の状態は未規定。再利用時は reset() すること。
  [[nodiscard]] std::array<u8, 32> digest() noexcept;
  [[nodiscard]] std::string        hex();

private:
  std::array<u32, 8> h_{};
  std::array<u8, 64> buf_{};
  std::size_t        buf_len_ = 0;
  u64                total_bits_ = 0;

  void compress(const u8* block) noexcept;
};

[[nodiscard]] std::string sha256_hex(const void* data, std::size_t len);

/// ファイル全体の SHA-256。巨大ファイルでも一定メモリで走る。
/// 進捗が要る場合は呼び出し側で分割すること。
[[nodiscard]] std::string sha256_file(const std::string& path);

}  // namespace alsr
