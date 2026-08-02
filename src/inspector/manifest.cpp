#include "alsr/manifest.hpp"

#include <algorithm>
#include <array>
#include <sstream>
#include <unordered_map>

#include "alsr/ggml_types.hpp"

namespace alsr {
namespace {

/// "blk.<N>.<rest>" を (N, rest) へ分解する。ブロック外なら layer=-1。
struct NameParts {
  int         layer = -1;
  std::string detail;  ///< blk.N. を除いた残り。ブロック外は名前そのもの。
};

NameParts split_name(const std::string& name) {
  NameParts p;
  constexpr std::string_view kPrefix = "blk.";
  if (name.rfind(kPrefix, 0) != 0) {
    p.detail = name;
    return p;
  }
  std::size_t i = kPrefix.size();
  std::size_t digits_begin = i;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
  if (i == digits_begin || i >= name.size() || name[i] != '.') {
    p.detail = name;  // "blk." で始まるが想定形でない → unknown 扱いへ
    return p;
  }
  const std::string num = name.substr(digits_begin, i - digits_begin);
  // 桁溢れは unknown として扱う（推測しない）
  try {
    p.layer = std::stoi(num);
  } catch (...) {
    p.detail = name;
    return p;
  }
  p.detail = name.substr(i + 1);
  return p;
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

std::string_view to_string(TensorRole r) noexcept {
  switch (r) {
    case TensorRole::Embedding: return "embedding";
    case TensorRole::Output:    return "output";
    case TensorRole::Norm:      return "norm";
    case TensorRole::Attention: return "attention";
    case TensorRole::Ffn:       return "ffn";
    case TensorRole::MoeRouter: return "moe_router";
    case TensorRole::MoeExpert: return "moe_expert";
    case TensorRole::Rope:      return "rope";
    case TensorRole::Scale:     return "scale";
    case TensorRole::Unknown:   return "unknown";
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
  std::vector<Pending> pre, post;
  std::vector<std::vector<Pending>> blocks;

  int max_layer = -1;
  for (std::size_t i = 0; i < f.tensors.size(); ++i) {
    const auto& t = f.tensors[i];
    NameParts   p = split_name(t.name);
    const TensorRole role = classify(p.detail, p.layer);

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
      //   per_token_bytes_upper 側で別途加算する）。
      e.per_token_full_read = (p.role != TensorRole::Embedding);
      e.resident_candidate  = (t.nbytes < resident_threshold_bytes);

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

  // --- 物理量 ---------------------------------------------------------
  Bytes embedding_bytes = 0;
  for (const auto& e : m.entries) {
    m.total_tensor_bytes += e.bytes;
    if (e.resident_candidate) m.resident_candidate_bytes += e.bytes;
    if (e.role == TensorRole::Embedding) embedding_bytes += e.bytes;
    if (e.per_token_full_read) m.per_token_bytes_lower += e.bytes;
  }
  m.per_token_bytes_upper = m.per_token_bytes_lower;
  if (!m.has_separate_output_tensor) {
    // output.weight が無い = 埋め込み重み共有(tied)で logits を計算する構成が
    // 一般的。その場合、毎トークン埋め込み行列を全読みする。断定はしない。
    m.per_token_bytes_upper += embedding_bytes;
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
  phys.set("resident_candidate_bytes", m.resident_candidate_bytes);
  phys.set("per_token_bytes_lower", m.per_token_bytes_lower);
  phys.set("per_token_bytes_upper", m.per_token_bytes_upper);
  phys.set("has_separate_output_tensor", m.has_separate_output_tensor);
  phys.set("note",
           "per_token_bytes_lower/upper は Batch=1 Dense 前提。lower は埋め込み"
           "行列を lm_head として全読みしない場合、upper は全読みする場合。");
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
    j.set("role_detail", e.role_detail);
    j.set("type", ggml_type_name(e.type_id));
    j.set("bytes", e.bytes);
    j.set("per_token_full_read", e.per_token_full_read);
    j.set("resident_candidate", e.resident_candidate);
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
          b / static_cast<double>(m.per_token_bytes_upper));
    r.set("t_max_token_per_s_lower_bound_case",
          b / static_cast<double>(m.per_token_bytes_lower));
    rows.push_back(std::move(r));
  }

  json::Value out = json::Value::make_object();
  out.set("formula", "T_max <= B_h2d / S_stream  (設計書 §2.3)");
  out.set("s_stream_lower_bytes", m.per_token_bytes_lower);
  out.set("s_stream_upper_bytes", m.per_token_bytes_upper);
  out.set("rows", std::move(rows));
  out.set("caveat",
          "転送のみの理想上限。カーネル実行・同期・KVキャッシュ・再配置は"
          "含まない。実効値は必ずこれより低い。");
  return out;
}

}  // namespace alsr
