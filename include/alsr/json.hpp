// ALSR PoC — 最小 JSON DOM（parser + writer）。
//
// なぜ自前か (docs/DECISIONS.md DR-0003):
//   - 設定・manifest・plan・metrics すべてが JSON/JSONL であり、依存物が少ない
//     ほど再現性が上がる。ビルド時のネットワーク取得は Catch2 のみに限定する。
//   - オブジェクトのキー順を「挿入順」で保存する。std::map による並べ替えを
//     避け、同一入力から同一バイト列の出力を得る（NFR-02 再現性）。
//
// 実装メモ:
//   オブジェクトは「キー配列」と「値配列」の 2 本立てで保持する。
//   vector<pair<string, Value>> だと Value が不完全型の時点で pair の実体化が
//   必要になり libc++ で破綻する。vector<Value> は不完全型でも成立する。
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "alsr/common.hpp"

namespace alsr::json {

class Value;

using Array = std::vector<Value>;

enum class Type { Null, Bool, Int, Real, String, Array, Object };

std::string_view to_string(Type t) noexcept;

class Value {
public:
  Value() = default;  ///< null
  Value(std::nullptr_t) {}  // NOLINT(google-explicit-constructor)

  Value(bool b) : t_(Type::Bool), b_(b) {}  // NOLINT

  template <class T, std::enable_if_t<std::is_integral_v<T> &&
                                          !std::is_same_v<std::decay_t<T>, bool>,
                                      int> = 0>
  Value(T v) : t_(Type::Int), i_(static_cast<std::int64_t>(v)) {}  // NOLINT

  template <class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  Value(T v) : t_(Type::Real), d_(static_cast<double>(v)) {}  // NOLINT

  Value(const char* s) : t_(Type::String), s_(s ? s : "") {}     // NOLINT
  Value(std::string s) : t_(Type::String), s_(std::move(s)) {}   // NOLINT
  Value(std::string_view s) : t_(Type::String), s_(s) {}         // NOLINT

  static Value make_array(Array a = {});
  static Value make_object();

  [[nodiscard]] Type type() const noexcept { return t_; }
  [[nodiscard]] bool is_null()   const noexcept { return t_ == Type::Null; }
  [[nodiscard]] bool is_bool()   const noexcept { return t_ == Type::Bool; }
  [[nodiscard]] bool is_int()    const noexcept { return t_ == Type::Int; }
  [[nodiscard]] bool is_real()   const noexcept { return t_ == Type::Real; }
  [[nodiscard]] bool is_number() const noexcept { return is_int() || is_real(); }
  [[nodiscard]] bool is_string() const noexcept { return t_ == Type::String; }
  [[nodiscard]] bool is_array()  const noexcept { return t_ == Type::Array; }
  [[nodiscard]] bool is_object() const noexcept { return t_ == Type::Object; }

  // 型が違えば Error(Invalid)。silent な既定値化はしない。
  [[nodiscard]] bool               as_bool()   const;
  [[nodiscard]] std::int64_t       as_int()    const;
  [[nodiscard]] double             as_real()   const;  ///< Int も受け付ける
  [[nodiscard]] const std::string& as_string() const;
  [[nodiscard]] const Array&       as_array()  const;
  Array& as_array();

  // --- 配列 ---------------------------------------------------------------
  void push_back(Value v);

  // --- オブジェクト -------------------------------------------------------
  /// 既存キーがあれば上書き、なければ末尾へ追加。挿入順を保つ。
  void set(std::string key, Value v);

  [[nodiscard]] std::size_t        object_size() const;
  [[nodiscard]] const std::string& key_at(std::size_t i) const;
  [[nodiscard]] const Value&       value_at(std::size_t i) const;

  [[nodiscard]] const Value* find(std::string_view key) const noexcept;
  [[nodiscard]] bool contains(std::string_view key) const noexcept {
    return find(key) != nullptr;
  }
  /// 必須キー。無ければ Error(Invalid)。
  [[nodiscard]] const Value& at(std::string_view key) const;

  // 既定値付き取得（設定読み込み用）
  [[nodiscard]] bool         get_bool(std::string_view k, bool def) const;
  [[nodiscard]] std::int64_t get_int(std::string_view k, std::int64_t def) const;
  [[nodiscard]] double       get_real(std::string_view k, double def) const;
  [[nodiscard]] std::string  get_string(std::string_view k, std::string def) const;

  /// indent < 0 で 1 行（JSONL 用）、>= 0 で整形。
  [[nodiscard]] std::string dump(int indent = -1) const;

  /// 解析失敗は Error(Format)。
  static Value parse(std::string_view text);
  static Value parse_file(const std::string& path);

private:
  Type                     t_ = Type::Null;
  bool                     b_ = false;
  std::int64_t             i_ = 0;
  double                   d_ = 0.0;
  std::string              s_;
  std::vector<Value>       arr_;   ///< Array の要素 / Object の値
  std::vector<std::string> keys_;  ///< Object のキー（arr_ と同じ長さ）

  void dump_to(std::string& out, int indent, int depth) const;
};

/// double を最短往復表現へ整形する（決定論的）。非有限値は Error(Invalid)。
std::string format_real(double v);

/// JSON 文字列としてエスケープ（前後の " を含む）。
std::string quote(std::string_view s);

}  // namespace alsr::json
