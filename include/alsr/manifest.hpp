// ALSR PoC — 実行順序マニフェスト (FR-02)。
//
//   - テンソルを論理レイヤーへ関連付ける
//   - 実行順序、常駐候補を manifest へ記録する
//   - Embedding / output / norm / attention / FFN を区別する
//   - 解析できない名前規則は unknown として保存し、人間の確認を要求する
//
// 重要（推測を混ぜないための方針）:
//   「1 トークンあたり何バイト転送が要るか (S_stream)」は、埋め込み行列を
//   lm_head として全読みするかどうかで倍近く変わる。GGUF だけからは断定
//   できないため、下限と上限の両方を出し、どちらを採用するかは実行経路の
//   実測（Phase 5）で確定させる。
#pragma once

#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/gguf.hpp"
#include "alsr/json.hpp"

namespace alsr {

enum class TensorRole {
  Embedding,   ///< token_embd。Batch=1 では行の lookup のみ
  Output,      ///< lm_head / output.weight
  Norm,        ///< 各種 norm
  Attention,   ///< q/k/v/o 射影
  Ffn,         ///< gate/up/down
  MoeRouter,   ///< ffn_gate_inp 等（MoE 研究用。Dense PoC では出現しない想定）
  MoeExpert,   ///< *_exps
  Rope,        ///< rope_freqs 等の事前計算テーブル
  Scale,       ///< layer_output_scale 等のスカラ
  Unknown,     ///< 人間の確認が必要
};

std::string_view to_string(TensorRole r) noexcept;

struct ManifestEntry {
  std::size_t tensor_index = 0;   ///< GgufFile::tensors の添字
  std::string name;
  std::string role_detail;        ///< "attn_q" など、blk.N. を除いた識別子
  TensorRole  role = TensorRole::Unknown;
  int         layer = -1;         ///< ブロック番号。非ブロックは -1
  u64         exec_order = 0;     ///< 全体を通した実行順（0 始まり）
  Bytes       bytes = 0;
  u32         type_id = 0;

  /// Batch=1 の 1 トークンで全量が必要か（常識的なDense実行を仮定）。
  bool per_token_full_read = false;
  /// VRAM 常駐候補（小さく、毎トークン使う）。初期版は単純な閾値ヒューリスティック。
  bool resident_candidate = false;
};

/// 実行段。0 = 前処理、1..block_count = 各ブロック、block_count+1 = 後処理。
struct ExecStage {
  std::string              name;
  int                      layer = -1;
  std::vector<std::size_t> entries;  ///< Manifest::entries の添字
};

struct Manifest {
  std::string model_path;
  std::string architecture;
  std::string model_sha256;   ///< 空なら未計算
  int         block_count = 0;

  std::vector<ManifestEntry> entries;  ///< exec_order 昇順
  std::vector<ExecStage>     stages;

  std::size_t unknown_count = 0;
  bool        requires_human_review = false;
  std::vector<std::string> review_notes;

  // --- 物理量（§2 の式へ直接入る値） ---------------------------------
  Bytes total_tensor_bytes = 0;
  Bytes resident_candidate_bytes = 0;
  /// 埋め込み行列を lm_head として全読みしない場合の下限
  Bytes per_token_bytes_lower = 0;
  /// 埋め込み行列を tied lm_head として全読みする場合の上限
  Bytes per_token_bytes_upper = 0;
  bool  has_separate_output_tensor = false;
};

/// GGUF から実行順序マニフェストを構築する。
/// `resident_threshold_bytes` 未満のテンソルを常駐候補とみなす。
[[nodiscard]] Manifest build_manifest(const GgufFile& f,
                                      Bytes resident_threshold_bytes = 1 * MiB);

[[nodiscard]] json::Value to_json(const Manifest& m);

/// S_stream と実測帯域から導かれる T_max の表（§2.2 / §2.3）。
/// bandwidths_bytes_per_s は測定値または想定値。単位は byte/s。
[[nodiscard]] json::Value bandwidth_ceiling_table(
    const Manifest& m, const std::vector<double>& bandwidths_bytes_per_s);

/// VRAM 常駐率を振ったときの実効 S_stream と T_max（設計書 §2.4「大きな重み常駐率」）。
///
/// 重要: Batch=1 の Dense モデルでは、各重みは 1 トークンにつきちょうど 1 回しか
/// 使われない。したがって「どの重みを常駐させるか」は総転送量に影響しない。
/// 効くのは「何バイト常駐させるか」だけである。選択方針の議論は Batch > 1、
/// 投機デコード、MoE で初めて意味を持つ。
///
/// resident_bytes_list の各値について:
///   S_stream_effective = max(0, per_token_bytes_upper - resident)
///   T_max              = B / S_stream_effective
[[nodiscard]] json::Value residency_table(
    const Manifest& m, const std::vector<Bytes>& resident_bytes_list,
    const std::vector<double>& bandwidths_bytes_per_s);

}  // namespace alsr
