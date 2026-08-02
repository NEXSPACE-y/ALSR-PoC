#include "alsr/manifest.hpp"

#include <algorithm>
#include <numeric>
#include <array>
#include <sstream>
#include <unordered_map>

#include "alsr/ggml_types.hpp"

namespace alsr {
namespace {

/// テンソル名を (subsystem, layer, detail) へ分解する。
///
///   "blk.12.attn_q.weight"   -> Text,      layer=12, "attn_q.weight"
///   "v.blk.3.ln1.weight"     -> Vision,    layer=3,  "ln1.weight"
///   "v.patch_embd.weight"    -> Vision,    layer=-1, "patch_embd.weight"
///   "mm.input_projection.weight" -> Projector, layer=-1, "input_projection.weight"
///   "token_embd.weight"      -> Text,      layer=-1, "token_embd.weight"
struct NameParts {
  Subsystem   subsystem = Subsystem::Text;
  int         layer = -1;
  std::string detail;
};

/// "blk.<N>." で始まるなら (N, 残り) を返す。そうでなければ layer=-1。
bool strip_block_prefix(const std::string& in, int& layer_out,
                        std::string& rest_out) {
  constexpr std::string_view kPrefix = "blk.";
  if (in.rfind(kPrefix, 0) != 0) return false;
  std::size_t i = kPrefix.size();
  const std::size_t digits_begin = i;
  while (i < in.size() && in[i] >= '0' && in[i] <= '9') ++i;
  if (i == digits_begin || i >= in.size() || in[i] != '.') return false;
  try {
    layer_out = std::stoi(in.substr(digits_begin, i - digits_begin));
  } catch (...) {
    return false;  // 桁溢れ等は unknown へ倒す（推測しない）
  }
  rest_out = in.substr(i + 1);
  return true;
}

NameParts split_name(const std::string& name) {
  NameParts p;
  std::string body = name;

  if (name.rfind("v.", 0) == 0) {
    p.subsystem = Subsystem::Vision;
    body = name.substr(2);
  } else if (name.rfind("mm.", 0) == 0) {
    p.subsystem = Subsystem::Projector;
    body = name.substr(3);
  }

  int layer = -1;
  std::string rest;
  if (strip_block_prefix(body, layer, rest)) {
    p.layer = layer;
    p.detail = rest;
  } else {
    p.detail = body;
  }
  return p;
}

/// 視覚エンコーダ側の名前規則（llama.cpp の ViT 命名に合わせる）。
TensorRole classify_vision(const std::string& detail, int layer) {
  static const std::unordered_map<std::string, TensorRole> kBlockRoles = {
      {"attn_q.weight", TensorRole::Attention},
      {"attn_k.weight", TensorRole::Attention},
      {"attn_v.weight", TensorRole::Attention},
      {"attn_out.weight", TensorRole::Attention},
      {"attn_output.weight", TensorRole::Attention},
      {"attn_q_norm.weight", TensorRole::Norm},
      {"attn_k_norm.weight", TensorRole::Norm},
      {"attn_post_norm.weight", TensorRole::Norm},
      {"ffn_post_norm.weight", TensorRole::Norm},
      {"ln1.weight", TensorRole::Norm},
      {"ln2.weight", TensorRole::Norm},
      {"ln1.bias", TensorRole::Norm},
      {"ln2.bias", TensorRole::Norm},
      {"ffn_gate.weight", TensorRole::Ffn},
      {"ffn_up.weight", TensorRole::Ffn},
      {"ffn_down.weight", TensorRole::Ffn},
  };
  static const std::unordered_map<std::string, TensorRole> kGlobalRoles = {
      {"patch_embd.weight", TensorRole::PatchEmbed},
      {"patch_embd.bias", TensorRole::PatchEmbed},
      {"position_embd.weight", TensorRole::PosEmbed},
      {"std_scale", TensorRole::Scale},
      {"std_bias", TensorRole::Scale},
      {"post_ln.weight", TensorRole::Norm},
      {"post_ln.bias", TensorRole::Norm},
  };
  const auto& tbl = (layer >= 0) ? kBlockRoles : kGlobalRoles;
  const auto it = tbl.find(detail);
  return it == tbl.end() ? TensorRole::Unknown : it->second;
}

/// 視覚→言語の投影器。
TensorRole classify_projector(const std::string& detail) {
  static const std::unordered_map<std::string, TensorRole> kRoles = {
      {"input_projection.weight", TensorRole::Projector},
      {"input_projection.bias", TensorRole::Projector},
      {"soft_emb_norm.weight", TensorRole::Norm},
      {"mm_input_projection.weight", TensorRole::Projector},
  };
  const auto it = kRoles.find(detail);
  return it == kRoles.end() ? TensorRole::Unknown : it->second;
}

/// role_detail から役割を決める。表に無い名前は Unknown。
TensorRole classify(const std::string& detail, int layer) {
  static const std::unordered_map<std::string, TensorRole> kBlockRoles = {
      // attention
      {"attn_q.weight", TensorRole::Attention},
      {"attn_k.weight", TensorRole::Attention},
      {"attn_v.weight", TensorRole::Attention},
      {"attn_output.weight", TensorRole::Attention},
      {"attn_qkv.weight", TensorRole::Attention},
      // norm
      {"attn_norm.weight", TensorRole::Norm},
      {"attn_q_norm.weight", TensorRole::Norm},
      {"attn_k_norm.weight", TensorRole::Norm},
      {"post_attention_norm.weight", TensorRole::Norm},
      {"post_ffw_norm.weight", TensorRole::Norm},
      {"ffn_norm.weight", TensorRole::Norm},
      {"attn_norm_2.weight", TensorRole::Norm},
      // ffn
      {"ffn_gate.weight", TensorRole::Ffn},
      {"ffn_up.weight", TensorRole::Ffn},
      {"ffn_down.weight", TensorRole::Ffn},
      // MoE（Dense PoC では出ない想定。出たら別系統として扱う）
      {"ffn_gate_inp.weight", TensorRole::MoeRouter},
      {"ffn_gate_exps.weight", TensorRole::MoeExpert},
      {"ffn_up_exps.weight", TensorRole::MoeExpert},
      {"ffn_down_exps.weight", TensorRole::MoeExpert},
      // scale
      {"layer_output_scale.weight", TensorRole::Scale},
  };
  static const std::unordered_map<std::string, TensorRole> kGlobalRoles = {
      {"token_embd.weight", TensorRole::Embedding},
      {"output.weight", TensorRole::Output},
      {"output_norm.weight", TensorRole::Norm},
      {"rope_freqs.weight", TensorRole::Rope},
      {"token_embd_norm.weight", TensorRole::Norm},
  };

  const auto& tbl = (layer >= 0) ? kBlockRoles : kGlobalRoles;
  const auto it = tbl.find(detail);
  return it == tbl.end() ? TensorRole::Unknown : it->second;
}

/// ブロック内の標準的な実行順。表に無いものは末尾へ回し、ファイル順を保つ。
int intra_block_rank(const std::string& detail) {
  static const std::array<const char*, 14> kOrder = {
      "attn_norm.weight",        "attn_q.weight",
      "attn_q_norm.weight",      "attn_k.weight",
      "attn_k_norm.weight",      "attn_v.weight",
      "attn_output.weight",      "post_attention_norm.weight",
      "ffn_norm.weight",         "ffn_gate.weight",
      "ffn_up.weight",           "ffn_down.weight",
      "post_ffw_norm.weight",    "layer_output_scale.weight",
  };
  for (std::size_t i = 0; i < kOrder.size(); ++i) {
    if (detail == kOrder[i]) return static_cast<int>(i);
  }
  return static_cast<int>(kOrder.size());  // 未知は末尾
}

/// ブロック内の依存関係（FR-02「依存関係を manifest へ記録する」）。
/// key を使う前に value のテンソルが揃っている必要がある。
/// ここに無い名前は「同ブロック内の直前」に依存するものとして扱う。
const std::unordered_map<std::string, std::vector<std::string>>& block_deps() {
  static const std::unordered_map<std::string, std::vector<std::string>> kDeps = {
      {"attn_q.weight",              {"attn_norm.weight"}},
      {"attn_k.weight",              {"attn_norm.weight"}},
      {"attn_v.weight",              {"attn_norm.weight"}},
      {"attn_qkv.weight",            {"attn_norm.weight"}},
      {"attn_q_norm.weight",         {"attn_q.weight"}},
      {"attn_k_norm.weight",         {"attn_k.weight"}},
      {"attn_output.weight",         {"attn_q.weight", "attn_k.weight",
                                      "attn_v.weight"}},
      {"post_attention_norm.weight", {"attn_output.weight"}},
      {"ffn_norm.weight",            {"attn_output.weight"}},
      {"ffn_gate.weight",            {"ffn_norm.weight"}},
      {"ffn_up.weight",              {"ffn_norm.weight"}},
      {"ffn_down.weight",            {"ffn_gate.weight", "ffn_up.weight"}},
      {"post_ffw_norm.weight",       {"ffn_down.weight"}},
      {"layer_output_scale.weight",  {"ffn_down.weight"}},
  };
  return kDeps;
}

/// 並列グループ。同じ番号どうしは順序制約が無い（任意順で先読みしてよい）。
/// 0 は「単独」。
int parallel_group_of(const std::string& detail) {
  if (detail == "attn_q.weight" || detail == "attn_k.weight" ||
      detail == "attn_v.weight") {
    return 1;
  }
  if (detail == "ffn_gate.weight" || detail == "ffn_up.weight") return 2;
  return 0;
}

int pre_stage_rank(const std::string& detail) {
  if (detail == "token_embd.weight") return 0;
  if (detail == "token_embd_norm.weight") return 1;
  if (detail == "rope_freqs.weight") return 2;
  return 3;
}

int post_stage_rank(const std::string& detail) {
  if (detail == "output_norm.weight") return 0;
  if (detail == "output.weight") return 1;
  return 2;
}

}  // namespace

std::string_view to_string(Subsystem s) noexcept {
  switch (s) {
    case Subsystem::Text:      return "text";
    case Subsystem::Vision:    return "vision";
    case Subsystem::Projector: return "projector";
  }
  return "unknown";
}

std::string_view to_string(TensorRole r) noexcept {
  switch (r) {
    case TensorRole::Embedding: return "embedding";
    case TensorRole::Output:    return "output";
    case TensorRole::Norm:      return "norm";
    case TensorRole::Attention: return "attention";
    case TensorRole::Ffn:       return "ffn";
    case TensorRole::MoeRouter: return "moe_router";
    case TensorRole::MoeExpert: return "moe_expert";
    case TensorRole::Rope:       return "rope";
    case TensorRole::Scale:      return "scale";
    case TensorRole::PatchEmbed: return "patch_embed";
    case TensorRole::PosEmbed:   return "pos_embed";
    case TensorRole::Projector:  return "projector";
    case TensorRole::Unknown:    return "unknown";
  }
  return "unknown";
}

Manifest build_manifest(const GgufFile& f, Bytes resident_threshold_bytes) {
  Manifest m;
  m.model_path   = f.path;
  m.architecture = f.architecture;

  // block_count はメタデータ優先。無ければテンソル名から復元する。
  const std::string bc_key = f.architecture.empty()
                                 ? std::string("block_count")
                                 : f.architecture + ".block_count";
  int block_count_meta = -1;
  if (const json::Value* v = f.metadata.find(bc_key)) {
    if (v->is_int()) block_count_meta = static_cast<int>(v->as_int());
  }

  struct Pending {
    std::size_t tensor_index;
    NameParts   parts;
    TensorRole  role;
    int         rank;
  };
  std::vector<Pending> pre, post, aux;
  std::vector<std::vector<Pending>> blocks;

  int max_layer = -1;
  int max_vision_layer = -1;
  for (std::size_t i = 0; i < f.tensors.size(); ++i) {
    const auto& t = f.tensors[i];
    NameParts   p = split_name(t.name);
    TensorRole role = TensorRole::Unknown;
    switch (p.subsystem) {
      case Subsystem::Text:      role = classify(p.detail, p.layer); break;
      case Subsystem::Vision:    role = classify_vision(p.detail, p.layer); break;
      case Subsystem::Projector: role = classify_projector(p.detail); break;
    }

    // 視覚側・投影器はテキスト生成の実行順に載せない。別の段へ集める。
    if (p.subsystem != Subsystem::Text) {
      aux.push_back({i, p, role, 0});
      if (p.subsystem == Subsystem::Vision && p.layer >= 0) {
        max_vision_layer = std::max(max_vision_layer, p.layer);
      }
      continue;
    }

    if (p.layer >= 0) {
      max_layer = std::max(max_layer, p.layer);
      if (static_cast<std::size_t>(p.layer) >= blocks.size()) {
        blocks.resize(static_cast<std::size_t>(p.layer) + 1);
      }
      blocks[static_cast<std::size_t>(p.layer)].push_back(
          {i, p, role, intra_block_rank(p.detail)});
    } else if (role == TensorRole::Output || p.detail == "output_norm.weight") {
      post.push_back({i, p, role, post_stage_rank(p.detail)});
    } else {
      pre.push_back({i, p, role, pre_stage_rank(p.detail)});
    }
  }

  m.block_count = block_count_meta >= 0 ? block_count_meta : max_layer + 1;
  if (block_count_meta >= 0 && block_count_meta != max_layer + 1) {
    std::ostringstream os;
    os << "metadata の " << bc_key << "=" << block_count_meta
       << " とテンソル名から得た層数 " << (max_layer + 1) << " が一致しない";
    m.review_notes.push_back(os.str());
    m.requires_human_review = true;
  }

  m.has_separate_output_tensor = (f.find_tensor("output.weight") != nullptr);

  // --- 実行順に並べる -------------------------------------------------
  const auto stable_sort_by_rank = [](std::vector<Pending>& v) {
    std::stable_sort(v.begin(), v.end(),
                     [](const Pending& a, const Pending& b) { return a.rank < b.rank; });
  };
  stable_sort_by_rank(pre);
  for (auto& b : blocks) stable_sort_by_rank(b);
  stable_sort_by_rank(post);

  u64 order = 0;
  const auto emit_stage = [&](const std::string& stage_name, int layer,
                              const std::vector<Pending>& src) {
    ExecStage stage;
    stage.name  = stage_name;
    stage.layer = layer;
    for (const Pending& p : src) {
      const auto& t = f.tensors[p.tensor_index];
      ManifestEntry e;
      e.tensor_index = p.tensor_index;
      e.name         = t.name;
      e.role_detail  = p.parts.detail;
      e.role         = p.role;
      e.layer        = p.parts.layer;
      e.exec_order   = order++;
      e.bytes        = t.nbytes;
      e.type_id      = t.type_id;

      // Batch=1 Dense の毎トークン全読み判定。
      //   embedding は行 lookup のみなので false（tied lm_head の扱いは
      //   per_token_demand_upper 側で別途加算する）。
      // Batch=1 Dense のテキスト生成で毎トークン全量を読むか。
      //   - 視覚エンコーダ／投影器: 画像 1 枚につき 1 回。毎トークンではない
      //   - embedding: 行 lookup のみ（tied lm_head は upper 側で別途加算）
      e.subsystem = p.parts.subsystem;
      // 埋め込みの扱い:
      //   output.weight がある → 埋め込みは行 lookup のみ。毎トークン全読みしない
      //   output.weight が無い → tied lm_head として毎トークン全読みする
      // 後者は M5 実測（15.35 token/s × 全重み = 113 GB/s）が支持する
      //   （DR-0005）。別仮説は per_token_demand_lower 側に残してある。
      const bool embedding_streamed =
          (p.role == TensorRole::Embedding) && !m.has_separate_output_tensor;
      e.per_token_full_read = (p.parts.subsystem == Subsystem::Text) &&
                              (p.role != TensorRole::Embedding ||
                               embedding_streamed);
      e.resident_candidate  = (t.nbytes < resident_threshold_bytes);
      e.parallel_group =
          static_cast<u64>(parallel_group_of(p.parts.detail));

      if (p.role == TensorRole::Unknown) {
        ++m.unknown_count;
        m.requires_human_review = true;
      }
      stage.entries.push_back(m.entries.size());
      m.entries.push_back(std::move(e));
    }
    m.stages.push_back(std::move(stage));
  };

  emit_stage("pre", -1, pre);
  for (std::size_t l = 0; l < blocks.size(); ++l) {
    emit_stage("blk." + std::to_string(l), static_cast<int>(l), blocks[l]);
  }
  emit_stage("post", -1, post);
  // 視覚エンコーダ・投影器は「画像 1 枚につき 1 回」の別系統。
  // テキスト生成の実行順の後ろに置き、per_token 判定では除外する。
  // テキスト専用モデルでは空の段を作らない。
  if (!aux.empty()) emit_stage("aux_vision_projector", -1, aux);

  // --- FR-02 依存関係 -------------------------------------------------
  // 段ごとに (role_detail -> exec_order) を引けるようにしてから解決する。
  {
    const auto& deps = block_deps();
    for (const ExecStage& stage : m.stages) {
      std::unordered_map<std::string, u64> in_stage;
      for (const std::size_t idx : stage.entries) {
        in_stage.emplace(m.entries[idx].role_detail, m.entries[idx].exec_order);
      }
      // 段の先頭は「前の段の最後」に依存する
      u64 prev_stage_last = 0;
      bool has_prev = false;
      for (const auto& e : m.entries) {
        if (!stage.entries.empty() && e.exec_order < m.entries[stage.entries.front()].exec_order) {
          prev_stage_last = e.exec_order;
          has_prev = true;
        }
      }
      for (const std::size_t idx : stage.entries) {
        ManifestEntry& e = m.entries[idx];
        const auto it = deps.find(e.role_detail);
        if (it != deps.end()) {
          for (const std::string& need : it->second) {
            const auto found = in_stage.find(need);
            if (found != in_stage.end()) {
              e.depends_on.push_back(found->second);
            }
          }
        }
        if (e.depends_on.empty() && has_prev && e.exec_order > 0) {
          // 表に無い、または段内に先行が居ない → 直前の段の最後に依存する
          e.depends_on.push_back(prev_stage_last);
        }
      }
    }
  }

  // --- FR-02 常駐優先度 -------------------------------------------------
  // 小さいものから常駐させる。Batch=1 Dense では総転送量に影響しないが、
  // 細切れの転送単位を減らす意味はある（§2.4 / residency_table のコメント）。
  {
    std::vector<std::size_t> by_size(m.entries.size());
    std::iota(by_size.begin(), by_size.end(), std::size_t{0});
    std::stable_sort(by_size.begin(), by_size.end(),
                     [&](std::size_t a, std::size_t b) {
                       return m.entries[a].bytes < m.entries[b].bytes;
                     });
    for (std::size_t rank = 0; rank < by_size.size(); ++rank) {
      m.entries[by_size[rank]].resident_priority = rank;
    }
  }

  // --- 物理量 ---------------------------------------------------------
  Bytes embedding_bytes = 0;
  for (const auto& e : m.entries) {
    m.total_tensor_bytes += e.bytes;
    switch (e.subsystem) {
      case Subsystem::Text:      m.text_bytes += e.bytes; break;
      case Subsystem::Vision:    m.vision_bytes += e.bytes; break;
      case Subsystem::Projector: m.projector_bytes += e.bytes; break;
    }
    if (e.resident_candidate) m.resident_candidate_bytes += e.bytes;
    if (e.subsystem == Subsystem::Text && e.role == TensorRole::Embedding) {
      embedding_bytes += e.bytes;
    }
    if (e.per_token_full_read) m.per_token_demand_upper += e.bytes;
  }
  m.vision_block_count = max_vision_layer + 1;
  m.has_vision_tower = (m.vision_bytes > 0 || m.projector_bytes > 0);
  if (m.has_vision_tower) {
    std::ostringstream os;
    os << "視覚エンコーダ " << m.vision_bytes << " B + 投影器 "
       << m.projector_bytes << " B がモデル本体に同梱されている。"
       "テキスト生成の毎トークンでは読まれないため per_token_demand から"
       "除外した。画像入力を扱う経路では別途必要になる";
    m.review_notes.push_back(os.str());
  }
  // 下限は「埋め込み行列を lm_head として全読みしない」仮説。
  m.per_token_demand_lower =
      m.per_token_demand_upper > embedding_bytes
          ? m.per_token_demand_upper - embedding_bytes
          : m.per_token_demand_upper;
  if (!m.has_separate_output_tensor) {
    // output.weight が無い = 埋め込み重み共有(tied)で logits を計算する構成が
    // 一般的。その場合、毎トークン埋め込み行列を全読みする。断定はしない。
    if (embedding_bytes > 0) {
      std::ostringstream os;
      os << "output.weight が無い。tied embedding を lm_head として全読みする"
            "実装なら毎トークン +" << embedding_bytes
         << " bytes。実行経路(Phase 5)で要確認";
      m.review_notes.push_back(os.str());
    }
  }

  if (m.unknown_count > 0) {
    std::ostringstream os;
    os << "未分類テンソルが " << m.unknown_count
       << " 件ある。名前規則を確認し classify() を更新すること";
    m.review_notes.push_back(os.str());
  }
  return m;
}

json::Value to_json(const Manifest& m) {
  json::Value root = json::Value::make_object();
  root.set("model_path", m.model_path);
  root.set("architecture", m.architecture);
  root.set("model_sha256", m.model_sha256);
  root.set("block_count", m.block_count);
  root.set("tensor_count", m.entries.size());
  root.set("unknown_count", m.unknown_count);
  root.set("requires_human_review", m.requires_human_review);

  json::Value notes = json::Value::make_array();
  for (const auto& n : m.review_notes) notes.push_back(json::Value(n));
  root.set("review_notes", std::move(notes));

  json::Value phys = json::Value::make_object();
  phys.set("total_tensor_bytes", m.total_tensor_bytes);
  phys.set("text_bytes", m.text_bytes);
  phys.set("vision_bytes", m.vision_bytes);
  phys.set("projector_bytes", m.projector_bytes);
  phys.set("vision_block_count", m.vision_block_count);
  phys.set("has_vision_tower", m.has_vision_tower);
  phys.set("resident_candidate_bytes", m.resident_candidate_bytes);
  phys.set("per_token_demand_lower", m.per_token_demand_lower);
  phys.set("per_token_demand_upper", m.per_token_demand_upper);
  phys.set("s_stream_at_zero_residency_upper", m.stream_bytes(0, true));
  phys.set("s_stream_at_candidate_residency_upper",
           m.stream_bytes(m.resident_candidate_bytes, true));
  phys.set("has_separate_output_tensor", m.has_separate_output_tensor);
  phys.set("definition",
           "設計書 §2.1: S_stream = 1トークンで RAM→VRAM 転送が必要な重み量 = "
           "per_token_demand - resident。per_token_demand は常駐を考慮しない"
           "『読む必要がある量』であり S_stream ではない。");
  phys.set("note",
           "per_token_demand の lower/upper は Batch=1 Dense 前提。lower は"
           "埋め込み行列を lm_head として全読みしない場合、upper は全読みする場合。");
  root.set("physics", std::move(phys));

  // 役割別集計
  json::Value by_role = json::Value::make_object();
  for (int r = 0; r <= static_cast<int>(TensorRole::Unknown); ++r) {
    const auto role = static_cast<TensorRole>(r);
    std::size_t count = 0;
    Bytes bytes = 0;
    for (const auto& e : m.entries) {
      if (e.role == role) { ++count; bytes += e.bytes; }
    }
    if (count == 0) continue;
    json::Value o = json::Value::make_object();
    o.set("count", count);
    o.set("bytes", bytes);
    by_role.set(std::string(to_string(role)), std::move(o));
  }
  root.set("by_role", std::move(by_role));

  json::Value entries = json::Value::make_array();
  for (const auto& e : m.entries) {
    json::Value j = json::Value::make_object();
    j.set("exec_order", e.exec_order);
    j.set("tensor_index", e.tensor_index);
    j.set("name", e.name);
    j.set("layer", e.layer);
    j.set("role", std::string(to_string(e.role)));
    j.set("subsystem", std::string(to_string(e.subsystem)));
    j.set("role_detail", e.role_detail);
    j.set("type", ggml_type_name(e.type_id));
    j.set("bytes", e.bytes);
    j.set("per_token_full_read", e.per_token_full_read);
    j.set("resident_candidate", e.resident_candidate);
    j.set("resident_priority", e.resident_priority);
    j.set("parallel_group", e.parallel_group);
    json::Value deps = json::Value::make_array();
    for (const u64 d : e.depends_on) deps.push_back(json::Value(d));
    j.set("depends_on", std::move(deps));
    entries.push_back(std::move(j));
  }
  root.set("entries", std::move(entries));

  json::Value stages = json::Value::make_array();
  for (const auto& s : m.stages) {
    json::Value j = json::Value::make_object();
    j.set("name", s.name);
    j.set("layer", s.layer);
    j.set("entry_count", s.entries.size());
    Bytes b = 0;
    for (const std::size_t idx : s.entries) b += m.entries[idx].bytes;
    j.set("bytes", b);
    stages.push_back(std::move(j));
  }
  root.set("stages", std::move(stages));

  return root;
}

json::Value bandwidth_ceiling_table(const Manifest& m,
                                    const std::vector<double>& bandwidths) {
  json::Value rows = json::Value::make_array();
  for (const double b : bandwidths) {
    if (!(b > 0.0)) {
      fail(ErrorKind::Invalid, "bandwidth must be > 0 byte/s");
    }
    json::Value r = json::Value::make_object();
    r.set("bandwidth_bytes_per_s", b);
    r.set("bandwidth_gb_per_s", b / 1e9);
    r.set("t_max_token_per_s_upper_bound_case",
          b / static_cast<double>(m.per_token_demand_upper));
    r.set("t_max_token_per_s_lower_bound_case",
          b / static_cast<double>(m.per_token_demand_lower));
    rows.push_back(std::move(r));
  }

  json::Value out = json::Value::make_object();
  out.set("formula", "T_max <= B_h2d / S_stream  (設計書 §2.3)");
  out.set("residency_assumed_bytes", 0);
  out.set("s_stream_lower_bytes", m.per_token_demand_lower);
  out.set("s_stream_upper_bytes", m.per_token_demand_upper);
  out.set("residency_note",
          "この表は常駐 0 の場合。常駐率を振った表は residency_table() を見ること"
          "（設計書 §2.1 の S_stream は常駐分を差し引いた量である）。");
  out.set("rows", std::move(rows));
  out.set("caveat",
          "転送のみの理想上限。カーネル実行・同期・KVキャッシュ・再配置は"
          "含まない。実効値は必ずこれより低い。");
  return out;
}

json::Value residency_table(const Manifest& m,
                            const std::vector<Bytes>& resident_bytes_list,
                            const std::vector<double>& bandwidths) {
  json::Value rows = json::Value::make_array();
  for (const Bytes resident : resident_bytes_list) {
    // 等号を含める。resident == S_stream なら流すものが無く、T_max は
    // 「転送に律速されない」であって 0 ではない。0 を返すと誤読される。
    if (resident >= m.per_token_demand_upper) {
      // 常駐量が S_stream を超える = 全部載る = ALSR 不要。黙って丸めない。
      json::Value r = json::Value::make_object();
      r.set("resident_bytes", resident);
      r.set("fully_resident", true);
      r.set("note", "重み全体が常駐する。この構成では ALSR は不要である。");
      rows.push_back(std::move(r));
      continue;
    }
    const Bytes streamed = m.per_token_demand_upper - resident;

    json::Value r = json::Value::make_object();
    r.set("resident_bytes", resident);
    r.set("resident_fraction",
          static_cast<double>(resident) /
              static_cast<double>(m.per_token_demand_upper));
    r.set("s_stream_bytes", streamed);
    r.set("streamed_bytes_per_token", streamed);  // 別名（後方互換）
    r.set("fully_resident", false);

    json::Value tmax = json::Value::make_array();
    for (const double b : bandwidths) {
      if (!(b > 0.0)) fail(ErrorKind::Invalid, "bandwidth must be > 0 byte/s");
      json::Value t = json::Value::make_object();
      t.set("bandwidth_gb_per_s", b / 1e9);
      // streamed == 0 は上の分岐で除いてあるので、ここは必ず正。
      t.set("t_max_token_per_s", b / static_cast<double>(streamed));
      tmax.push_back(std::move(t));
    }
    r.set("t_max", std::move(tmax));
    rows.push_back(std::move(r));
  }

  json::Value out = json::Value::make_object();
  out.set("formula",
          "S_stream = per_token_demand_upper - resident_bytes  (設計書 §2.1)");
  out.set("per_token_demand_upper_bytes", m.per_token_demand_upper);
  out.set("rows", std::move(rows));
  out.set("note",
          "Batch=1 の Dense では各重みが 1 トークンにつき 1 回しか使われないため、"
          "どの重みを常駐させるかは総転送量に影響しない。効くのはバイト数だけ。"
          "選択方針が意味を持つのは Batch>1・投機デコード・MoE から。");
  out.set("caveat",
          "resident_bytes は『重みに割ける VRAM』であり VRAM 総容量ではない。"
          "KV キャッシュ・アクティベーション・CUDA コンテキストを引くこと。");
  return out;
}

json::Value extrapolate_table(const Manifest& base, double base_params_billion,
                              const std::vector<ExtrapolationTarget>& targets,
                              const std::vector<double>& bandwidths,
                              double target_token_per_s) {
  if (!(base_params_billion > 0.0)) {
    fail(ErrorKind::Invalid,
         "base_params_billion は正の値が必要（GGUF から推測しない）");
  }
  if (!(target_token_per_s > 0.0)) {
    fail(ErrorKind::Invalid, "target_token_per_s must be > 0");
  }

  const double base_bytes_per_param =
      static_cast<double>(base.per_token_demand_upper) /
      (base_params_billion * 1e9);

  json::Value rows = json::Value::make_array();
  for (const ExtrapolationTarget& t : targets) {
    if (!(t.params_billion > 0.0)) {
      fail(ErrorKind::Invalid, "params_billion must be > 0 for " + t.name);
    }
    const double bpp =
        t.bytes_per_param > 0.0 ? t.bytes_per_param : base_bytes_per_param;
    const double demand = t.params_billion * 1e9 * bpp;
    const double resident = static_cast<double>(t.resident_bytes);
    const double s_stream = demand > resident ? demand - resident : 0.0;

    json::Value r = json::Value::make_object();
    r.set("name", t.name);
    r.set("params_billion", t.params_billion);
    r.set("bytes_per_param", bpp);
    r.set("bytes_per_param_source",
          t.bytes_per_param > 0.0 ? "explicit" : "extrapolated_from_base");
    r.set("per_token_demand_bytes", demand);
    r.set("resident_bytes", t.resident_bytes);
    r.set("s_stream_bytes", s_stream);
    r.set("fully_resident", s_stream == 0.0);

    // §2.2: min(B) >= S_stream × T_target。目標速度に必要な帯域。
    r.set("required_bandwidth_bytes_per_s_for_target",
          s_stream * target_token_per_s);
    r.set("required_bandwidth_gb_per_s_for_target",
          s_stream * target_token_per_s / 1e9);

    json::Value tmax = json::Value::make_array();
    for (const double b : bandwidths) {
      if (!(b > 0.0)) fail(ErrorKind::Invalid, "bandwidth must be > 0");
      json::Value e = json::Value::make_object();
      e.set("bandwidth_gb_per_s", b / 1e9);
      if (s_stream == 0.0) {
        e.set("t_max_token_per_s", nullptr);
        e.set("note", "全重みが常駐する。転送に律速されない");
      } else {
        e.set("t_max_token_per_s", b / s_stream);
      }
      tmax.push_back(std::move(e));
    }
    r.set("t_max", std::move(tmax));
    rows.push_back(std::move(r));
  }

  json::Value out = json::Value::make_object();
  out.set("base_model", base.model_path);
  out.set("base_architecture", base.architecture);
  out.set("base_params_billion", base_params_billion);
  out.set("base_bytes_per_param", base_bytes_per_param);
  out.set("base_per_token_demand_bytes", base.per_token_demand_upper);
  out.set("target_token_per_s", target_token_per_s);
  out.set("rows", std::move(rows));
  out.set("method",
          "基準モデルの実測 bytes/param を規模へ線形外挿する。"
          "設計書 §4.1-6 / §11 Phase 6。実ファイルは取得しない（§5.2 / §14）。");
  out.set("caveat",
          "線形外挿は同一量子形式・同一アーキテクチャ族を前提とする。"
          "MoE はアクティブ重みが総パラメータと一致しないため、この表を"
          "そのまま適用してはならない（設計書 §2.5）。"
          "また転送のみの上限であり、カーネル・同期・KVキャッシュを含まない。");
  return out;
}

}  // namespace alsr
