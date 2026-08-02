// ALSR PoC — Chunk Planner (FR-03)。
//
//   - 固定サイズと可変サイズを切り替えられる
//   - テンソル境界を優先し、分割が必要な場合は量子ブロック境界と
//     alignment を守る
//   - 各 Chunk へ連番・元 offset・length・checksum・関連テンソルを付与する
//   - 同じ入力と設定から同じ計画を生成する（決定論）
//
// 不変条件（テストで固定する）:
//   1. Chunk 群は [data_start, file_size) を隙間なく重複なく敷き詰める。
//   2. 各テンソルの各バイトはちょうど 1 つの Chunk に属する。
//   3. テンソル内部の切れ目は必ず lcm(block_bytes, alignment) の倍数。
#pragma once

#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/gguf.hpp"
#include "alsr/json.hpp"
#include "alsr/manifest.hpp"

namespace alsr {

enum class ChunkMode {
  Fixed,     ///< target_bytes に近い大きさへ揃える
  Variable,  ///< テンソル境界優先。min_bytes 未満は併合、max_bytes 超は分割
};

std::string_view to_string(ChunkMode m) noexcept;
[[nodiscard]] ChunkMode chunk_mode_from_string(std::string_view s);

struct PlanOptions {
  ChunkMode mode         = ChunkMode::Fixed;
  Bytes     target_bytes = 256 * MiB;  ///< Fixed 用
  Bytes     min_bytes    = 16 * MiB;   ///< Variable 用
  Bytes     max_bytes    = 1024 * MiB; ///< Variable 用
  bool      compute_checksums = true;
  Bytes     read_buffer_bytes = 8 * MiB;
};

/// Chunk 内に含まれるテンソル（またはその一部）。
struct ChunkTensorRef {
  std::size_t tensor_index = 0;
  std::string name;
  Bytes       offset_in_chunk = 0;   ///< Chunk 先頭からの相対
  Bytes       offset_in_tensor = 0;  ///< テンソル先頭からの相対
  Bytes       bytes = 0;
  bool        partial = false;       ///< テンソルが複数 Chunk に跨るか
  u64         exec_order = 0;
  /// このテンソルがテキスト生成で毎トークン読まれるか（manifest 由来）。
  /// 視覚エンコーダ・投影器・埋め込み lookup は false。
  bool        per_token = false;
};

struct Chunk {
  u64   chunk_id = 0;         ///< ファイルオフセット順の連番
  Bytes source_offset = 0;    ///< ファイル先頭からの絶対オフセット
  Bytes bytes = 0;
  u32   crc32c = 0;
  bool  checksum_computed = false;
  Bytes tensor_bytes = 0;     ///< うちテンソル実体
  Bytes padding_bytes = 0;    ///< うちアライメント埋め
  /// うち「毎トークン読まれる」テンソルの実体バイト。
  /// 0 なら、この Chunk は定常状態のストリーミング対象ではない。
  Bytes per_token_bytes = 0;
  [[nodiscard]] bool is_per_token() const noexcept { return per_token_bytes > 0; }
  u64   first_exec_order = 0;
  u64   last_exec_order = 0;
  std::vector<ChunkTensorRef> tensors;
};

struct ChunkPlan {
  std::string  model_path;
  std::string  model_sha256;   ///< 空なら未計算
  PlanOptions  options;
  Bytes        alignment = 32;
  Bytes        data_start = 0;
  Bytes        file_size = 0;

  std::vector<Chunk> chunks;              ///< offset 昇順 == chunk_id 昇順
  std::vector<std::size_t> exec_sequence; ///< 実行順に並べた chunks の添字

  // 検証結果
  Bytes covered_bytes = 0;
  Bytes tensor_bytes_total = 0;
  Bytes padding_bytes_total = 0;
  /// 毎トークン読まれるテンソルの合計（manifest の per_token_demand と一致する）
  Bytes per_token_bytes_total = 0;
  /// 毎トークン読まれない Chunk の合計バイト（視覚エンコーダ等）
  Bytes non_per_token_chunk_bytes = 0;
  std::size_t non_per_token_chunk_count = 0;
  std::size_t split_tensor_count = 0;
  /// 実行順とファイル順が逆転している箇所の数（逐次先読みの効きやすさの指標）
  std::size_t exec_order_inversions = 0;

  /// checksum を除いた計画構造のハッシュ。決定論の検証に使う。
  std::string plan_structure_sha256;

  [[nodiscard]] Bytes min_chunk_bytes() const noexcept;
  [[nodiscard]] Bytes max_chunk_bytes() const noexcept;
};

/// 計画を作る。compute_checksums=true ならモデルを読み取り専用で走査する。
[[nodiscard]] ChunkPlan build_plan(const GgufFile& f, const Manifest& m,
                                   const PlanOptions& opt);

/// 不変条件を検証する。破れていれば Error(Consistency)。
void verify_plan(const GgufFile& f, const ChunkPlan& plan);

[[nodiscard]] json::Value to_json(const ChunkPlan& plan);

/// plan.json を読み戻す（alsr-sim が実チャンクを使うため）。
[[nodiscard]] ChunkPlan plan_from_json(const json::Value& v);

}  // namespace alsr
