// ALSR PoC — 共通型とエラー。
//
// 方針 (FR-01 / NFR-03):
//   推測で処理を続行しない。解釈できない入力は例外で明示的に失敗させる。
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace alsr {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

/// バイト数。オフセットも長さもこの型で表す。
using Bytes = std::uint64_t;

inline constexpr Bytes KiB = 1024ull;
inline constexpr Bytes MiB = 1024ull * 1024ull;
inline constexpr Bytes GiB = 1024ull * 1024ull * 1024ull;

/// 失敗の分類。ログとレポートで機械的に集計できるようにする。
enum class ErrorKind {
  Io,             ///< ファイルが開けない、read が短い等
  Format,         ///< GGUF の magic/version/構造が想定外
  Unsupported,    ///< 既知だが本 PoC が扱わない形式（黙って推測しない）
  Invalid,        ///< 設定値・引数が不正
  Consistency,    ///< 自己検証に失敗（被覆漏れ、checksum 不一致など）
  State,          ///< 状態機械の遷移違反
};

inline std::string_view to_string(ErrorKind k) noexcept {
  switch (k) {
    case ErrorKind::Io:          return "io";
    case ErrorKind::Format:      return "format";
    case ErrorKind::Unsupported: return "unsupported";
    case ErrorKind::Invalid:     return "invalid";
    case ErrorKind::Consistency: return "consistency";
    case ErrorKind::State:       return "state";
  }
  return "unknown";
}

class Error : public std::runtime_error {
public:
  Error(ErrorKind kind, std::string message)
      : std::runtime_error(std::string(to_string(kind)) + ": " + message),
        kind_(kind), raw_(std::move(message)) {}

  [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }
  [[nodiscard]] const std::string& raw() const noexcept { return raw_; }

private:
  ErrorKind   kind_;
  std::string raw_;
};

[[noreturn]] inline void fail(ErrorKind k, std::string msg) {
  throw Error(k, std::move(msg));
}

inline void require(bool cond, ErrorKind k, std::string msg) {
  if (!cond) fail(k, std::move(msg));
}

/// [offset, offset + length) の半開区間。
struct ByteRange {
  Bytes offset = 0;
  Bytes length = 0;

  [[nodiscard]] Bytes end() const noexcept { return offset + length; }
  [[nodiscard]] bool  empty() const noexcept { return length == 0; }

  [[nodiscard]] bool overlaps(const ByteRange& o) const noexcept {
    return offset < o.end() && o.offset < end();
  }
  friend bool operator==(const ByteRange& a, const ByteRange& b) noexcept {
    return a.offset == b.offset && a.length == b.length;
  }
};

/// value 以上で最小の alignment の倍数。alignment は 2 の冪でなくてもよい。
inline constexpr Bytes align_up(Bytes value, Bytes alignment) noexcept {
  if (alignment <= 1) return value;
  const Bytes rem = value % alignment;
  return rem == 0 ? value : value + (alignment - rem);
}

/// value 以下で最大の alignment の倍数。
inline constexpr Bytes align_down(Bytes value, Bytes alignment) noexcept {
  if (alignment <= 1) return value;
  return value - (value % alignment);
}

/// 人間可読なバイト表記（レポート用。数値は常に生バイトも併記すること）。
inline std::string human_bytes(Bytes n) {
  static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(n);
  int u = 0;
  while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
  char buf[64];
  if (u == 0) std::snprintf(buf, sizeof buf, "%llu B",
                            static_cast<unsigned long long>(n));
  else        std::snprintf(buf, sizeof buf, "%.2f %s", v, kUnits[u]);
  return std::string(buf);
}

}  // namespace alsr
