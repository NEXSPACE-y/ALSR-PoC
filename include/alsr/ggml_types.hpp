// ALSR PoC — ggml テンソル型テーブル。
//
// FR-01: 「未対応量子形式は黙って推測せず、明示的に失敗する」
//
// 本テーブルは ggml の block レイアウト（1 ブロックあたりの要素数と
// バイト数）を持つ。値が正しいかどうかは、最終的に
//   Σ tensor_bytes + padding == file_size - data_start
// の整合チェック（inspector）で機械的に検証される。テーブルが誤っていれば
// そこで必ず落ちる設計にしてある。
#pragma once

#include <string>
#include <vector>

#include "alsr/common.hpp"

namespace alsr {

/// GGUF / ggml のテンソル型 ID。ここに無い ID は Unsupported として扱う。
struct GgmlTypeInfo {
  u32         id            = 0;
  const char* name          = "";
  u64         block_elems   = 0;  ///< 1 ブロックあたりの要素数
  u64         block_bytes   = 0;  ///< 1 ブロックあたりのバイト数
  bool        storable      = false;  ///< GGUF に格納され得る型か
  const char* note          = "";
};

/// 未知の ID なら nullptr。
[[nodiscard]] const GgmlTypeInfo* ggml_type_info(u32 id) noexcept;

/// 表示用。未知なら "unknown(<id>)"。
[[nodiscard]] std::string ggml_type_name(u32 id);

/// テンソルのバイト数。
/// 未知型・非格納型・ne[0] がブロック境界に揃っていない場合は Error を投げる。
[[nodiscard]] Bytes ggml_tensor_bytes(u32 type_id, const std::vector<u64>& dims);

/// 量子ブロック 1 個分のバイト数（Chunk 分割の最小単位に使う）。
/// 未知型・非格納型は Error。
[[nodiscard]] Bytes ggml_block_bytes(u32 type_id);

/// テーブル全体（environment.json / manifest への記録用）。
[[nodiscard]] const std::vector<GgmlTypeInfo>& ggml_type_table();

}  // namespace alsr
