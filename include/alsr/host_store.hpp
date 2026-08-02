// ALSR PoC — Host Store (FR-04)。
//
//   - モデル全体または対象範囲を RAM 上で参照できる
//   - mmap と明示 read の両方を比較できる
//   - NFR-03: モデルファイルは読み取り専用。書き込み・変換・削除をしない。
//
// Pinned Memory は CUDA backend (Phase 3) の担当。M5 では該当経路が無いため、
// ここでは「pinned 相当のホスト側スラブ」を確保量の上限つきで模擬するだけで、
// 効果の合否判定はしない（設計書 §3.1）。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "alsr/common.hpp"

namespace alsr {

enum class HostStoreMode {
  Mmap,       ///< mmap(PROT_READ, MAP_PRIVATE)
  Read,       ///< pread でその都度読む
  Synthetic,  ///< モデル無し。決定論的パターンを生成する
};

std::string_view to_string(HostStoreMode m) noexcept;
[[nodiscard]] HostStoreMode host_store_mode_from_string(std::string_view s);

class HostStore {
public:
  /// Synthetic モードでは path を無視し、synthetic_size バイトの仮想ファイルを作る。
  HostStore(std::string path, HostStoreMode mode, Bytes synthetic_size = 0);
  ~HostStore();

  HostStore(const HostStore&) = delete;
  HostStore& operator=(const HostStore&) = delete;

  /// [offset, offset+len) を dst へコピーする。
  /// Mmap では memcpy、Read では pread、Synthetic では生成。
  void read_into(Bytes offset, Bytes len, u8* dst);

  [[nodiscard]] Bytes         file_size() const noexcept { return file_size_; }
  [[nodiscard]] HostStoreMode mode() const noexcept { return mode_; }
  [[nodiscard]] Bytes         bytes_read() const noexcept { return bytes_read_; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

  /// Synthetic データの生成規則（テストが同じ値を再現できるように公開する）。
  static u8 synthetic_byte(Bytes absolute_offset) noexcept {
    // 単純な線形合同。offset だけから決まるので順序に依存しない。
    const u64 x = absolute_offset * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<u8>((x >> 33) & 0xFFu);
  }

private:
  std::string   path_;
  HostStoreMode mode_;
  Bytes         file_size_ = 0;
  Bytes         bytes_read_ = 0;
  int           fd_ = -1;
  void*         map_ = nullptr;
  std::size_t   map_len_ = 0;
};

}  // namespace alsr
