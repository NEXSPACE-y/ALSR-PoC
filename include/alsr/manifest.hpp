// ALSR PoC — 実行順序マニフェスト (FR-02)。
//
//   - テンソルを論理レイヤーへ関連付ける
//   - 実行順序、常駐候補を manifest へ記録する
//   - Embedding / output / norm / attention / FFN を区別する
//   - 解析できない名前規則は unknown として保存し、人間の確認を要求する
//
// 用語（設計書 §2.1 の定義に厳密に従う）:
//
//   per_token_demand  … 1 トークンの Forward Pass で「読む必要がある」重み量。
//                       常駐しているかどうかは問わない。
//   S_stream          … そのうち「RAM→VRAM 転送が必要な」量。
//                       すなわち  S_stream = per_token_demand - resident。
//
//   この 2 つを混同してはならない。設計書 §2.3 は「VRAM へ保持できない部分を
//   毎トークン転送する場合」と書いており、S_stream は最初から常駐分を
//   差し引いた量である。常駐 0 は S_stream の特殊ケースにすぎない。
//   §2.4 は「大きな重み常駐率」を 405B 級の成立条件として明示的に挙げている。
//
// 上限・下限（推測を混ぜないための方針）:
//   per_token_demand は、埋め込み行列を lm_head として全読みするかどうかで
//   倍近く変わる。GGUF だけからは断定できないため下限と上限の両方を出し、
//   どちらを採用するかは実行経路の実測（Phase 5）で確定させる。
#pragma once

#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/gguf.hpp"
#include "alsr/json.hpp"

namespace alsr {

/// テンソルが属するサブシステム。
/// マルチモーダル GGUF では言語モデル本体・視覚エンコーダ・投影器が
/// 1 ファイルに同居する。**視覚側はテキスト生成の毎トークンでは読まれない**
/// ので、S_stream に含めてはならない。
enum class Subsystem {
  Text,       ///< 言語モデル本体。毎トークン走る
  Vision,     ///< 視覚エンコーダ (v.*)。画像 1 枚につき 1 回
  Projector,  ///< 視覚→言語の投影器 (mm.*)。画像 1 枚につき 1 回
};

std::string_view to_string(Subsystem s) noexcept;

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
  PatchEmbed,  ///< 視覚側のパッチ埋め込み
  PosEmbed,    ///< 視覚側の位置埋め込み
  Projector,   ///< 視覚→言語の投影
  Unknown,     ///< 人間の確認が必要
};

std::string_view to_string(TensorRole r) noexcept;

struct ManifestEntry {
  std::size_t tensor_index = 0;   ///< GgufFile::tensors の添字
  std::string name;
  std::string role_detail;        ///< "attn_q" など、接頭辞を除いた識別子
  TensorRole  role = TensorRole::Unknown;
  Subsystem   subsystem = Subsystem::Text;
  int         layer = -1;         ///< ブロック番号。非ブロックは -1
  u64         exec_order = 0;     ///< 全体を通した実行順（0 始まり）
  Bytes       bytes = 0;
  u32         type_id = 0;

  /// Batch=1 の 1 トークンで全量を読む必要があるか（Dense 実行を仮定）。
  bool per_token_full_read = false;

  // --- FR-02 依存関係 ---------------------------------------------------
  /// このテンソルを使う前に完了していなければならない実行順の集合。
  /// 線形な「直前」ではなく、計算グラフ上の先行関係を入れる。
  std::vector<u64> depends_on;
  /// 同じ値を持つ entry どうしは順序制約が無い（並列に先読みしてよい）。
  /// 例: attn_q / attn_k / attn_v は attn_norm さえ済んでいれば任意順。
  u64 parallel_group = 0;

  // --- FR-02 常駐候補 ---------------------------------------------------
  /// VRAM 常駐候補か（閾値ヒューリスティック）。
  bool resident_candidate = false;
  /// 常駐させる優先順位。小さいほど優先。S_stream の計算に使う。
  u64 resident_priority = 0;
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
  /// 閾値ヒューリスティックで常駐候補とした分の合計（FR-02）
  Bytes resident_candidate_bytes = 0;

  // --- サブシステム別（マルチモーダル GGUF の分離） ------------------
  Bytes text_bytes = 0;
  Bytes vision_bytes = 0;
  Bytes projector_bytes = 0;
  int   vision_block_count = 0;
  /// 視覚側を含むか。含む場合、per_token_demand は視覚側を**含まない**。
  bool  has_vision_tower = false;

  /// 1 トークンで読む必要がある重み量。**これは S_stream ではない。**
  Bytes per_token_demand_lower = 0;
  Bytes per_token_demand_upper = 0;
  bool  has_separate_output_tensor = false;

  /// 設計書 §2.1 の S_stream。
  ///   S_stream = per_token_demand - resident_bytes
  /// resident_bytes が demand 以上なら 0（全部常駐 = 転送不要）。
  [[nodiscard]] Bytes stream_bytes(Bytes resident_bytes,
                                   bool upper_case = true) const noexcept {
    const Bytes demand = upper_case ? per_token_demand_upper
                                    : per_token_demand_lower;
    return resident_bytes >= demand ? Bytes{0} : demand - resident_bytes;
  }
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
///   S_stream = max(0, per_token_demand_upper - resident)
///   T_max              = B / S_stream_effective
[[nodiscard]] json::Value residency_table(
    const Manifest& m, const std::vector<Bytes>& resident_bytes_list,
    const std::vector<double>& bandwidths_bytes_per_s);

/// 規模外挿（設計書 §4.1-6 / §2.4 / §11 Phase 6）。
/// 実測した基準モデルの「パラメータあたりバイト数」を使い、31B/70B/120B/405B
/// 級で必要になる帯域と成立条件を数理的に外挿する。実ファイルは取得しない。
struct ExtrapolationTarget {
  std::string name;
  double      params_billion = 0.0;
  /// 1 パラメータあたりのバイト数。0 なら基準モデルの実測値を使う。
  double      bytes_per_param = 0.0;
  /// 重みに割ける VRAM [byte]
  Bytes       resident_bytes = 0;
};

/// base_params_billion は基準モデルのパラメータ数。GGUF から正確に取れない
/// ことがあるので外から与える（推測しない）。
[[nodiscard]] json::Value extrapolate_table(
    const Manifest& base, double base_params_billion,
    const std::vector<ExtrapolationTarget>& targets,
    const std::vector<double>& bandwidths_bytes_per_s,
    double target_token_per_s);

}  // namespace alsr
