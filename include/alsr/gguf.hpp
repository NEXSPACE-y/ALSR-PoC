// ALSR PoC — GGUF 読み取り専用パーサ (FR-01)。
//
// 制約:
//   - ファイルは読み取り専用で開く。書き込み・変換・移動を一切行わない。
//   - magic / version / 構造が想定外なら明示的に失敗する。
//   - 未対応の量子形式は推測せず Error(Unsupported)。
//   - 「解析対象 byte 範囲の重複／欠落が説明可能」であること（Phase 1 合格条件）を
//     GgufLayoutReport として機械可読に出す。
#pragma once

#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/json.hpp"

namespace alsr {

/// GGUF メタデータ値型 ID。
enum class GgufValueType : u32 {
  Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3, Uint32 = 4, Int32 = 5,
  Float32 = 6, Bool = 7, String = 8, Array = 9, Uint64 = 10, Int64 = 11,
  Float64 = 12,
};

std::string_view gguf_value_type_name(GgufValueType t) noexcept;

struct GgufTensorInfo {
  std::string      name;
  std::vector<u64> dims;         ///< ne[0..rank-1]（ggml の順序のまま）
  u32              type_id = 0;
  Bytes            rel_offset = 0;  ///< データ領域先頭からの相対
  Bytes            abs_offset = 0;  ///< ファイル先頭からの絶対
  Bytes            nbytes = 0;      ///< 型テーブルから算出
  std::size_t      file_index = 0;  ///< テンソル情報の出現順

  [[nodiscard]] ByteRange range() const noexcept { return {abs_offset, nbytes}; }
};

/// 被覆検査で見つかった説明できない領域。
struct GgufLayoutIssue {
  std::string kind;    ///< "overlap" | "gap" | "tail" | "out_of_range"
  Bytes       offset = 0;
  Bytes       length = 0;
  std::string detail;
};

struct GgufLayoutReport {
  Bytes data_start         = 0;  ///< テンソルデータ領域の先頭（alignment 適用後）
  Bytes data_bytes         = 0;  ///< file_size - data_start
  Bytes tensor_bytes_total = 0;  ///< Σ nbytes
  Bytes padding_bytes      = 0;  ///< alignment で説明できる埋め
  Bytes unexplained_bytes  = 0;  ///< 説明できない残り
  std::vector<GgufLayoutIssue> issues;

  [[nodiscard]] bool ok() const noexcept {
    return issues.empty() && unexplained_bytes == 0;
  }
};

struct GgufFile {
  std::string path;
  Bytes       file_size   = 0;
  u32         version     = 0;
  u64         tensor_count = 0;
  u64         kv_count    = 0;
  Bytes       alignment   = 32;   ///< general.alignment（既定 32）
  Bytes       header_end  = 0;    ///< テンソル情報セクションの終端
  Bytes       data_start  = 0;    ///< align_up(header_end, alignment)
  std::string architecture;       ///< general.architecture

  std::vector<GgufTensorInfo> tensors;   ///< ファイル出現順
  json::Value                 metadata;  ///< KV を JSON 化したもの
  GgufLayoutReport            layout;

  [[nodiscard]] const GgufTensorInfo* find_tensor(std::string_view name) const noexcept;
};

struct GgufReadOptions {
  /// この長さを超える配列メタデータは要約のみ記録する
  /// （tokenizer.ggml.tokens 等が manifest を数百 MB にするのを防ぐ）。
  std::size_t array_inline_limit = 32;
};

/// 読み取り専用で GGUF を解析する。失敗は Error。
[[nodiscard]] GgufFile read_gguf(const std::string& path,
                                 const GgufReadOptions& opt = {});

/// manifest.json へ書ける形へ変換する。
[[nodiscard]] json::Value to_json(const GgufFile& f);

}  // namespace alsr
