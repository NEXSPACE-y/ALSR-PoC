#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <vector>

#include "alsr/ggml_types.hpp"
#include "alsr/gguf.hpp"
#include "alsr/manifest.hpp"
#include "test_helpers.hpp"

using namespace alsr;
using namespace alsr::test;

namespace {

std::vector<TensorSpec> gemma_like_specs() {
  std::vector<TensorSpec> t;
  t.push_back({"output_norm.weight", 0, {3840}});
  t.push_back({"rope_freqs.weight", 0, {256}});
  t.push_back({"token_embd.weight", 14, {3840, 1024}});
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    t.push_back({p + "attn_norm.weight", 0, {3840}});
    t.push_back({p + "attn_q.weight", 12, {3840, 512}});
    t.push_back({p + "attn_k.weight", 12, {3840, 256}});
    t.push_back({p + "attn_v.weight", 12, {3840, 256}});
    t.push_back({p + "attn_output.weight", 12, {3840, 512}});
    t.push_back({p + "ffn_norm.weight", 0, {3840}});
    t.push_back({p + "ffn_gate.weight", 12, {3840, 1024}});
    t.push_back({p + "ffn_up.weight", 12, {3840, 1024}});
    t.push_back({p + "ffn_down.weight", 14, {3840, 1024}});
    t.push_back({p + "layer_output_scale.weight", 0, {1}});
  }
  return t;
}

}  // namespace

TEST_CASE("GGUF: 合成ファイルを解析でき、被覆が完全に説明できる",
          "[unit][gguf]") {
  TempDir tmp("gguf-basic");
  const std::string p = tmp.file("model.gguf");
  const auto built = write_test_gguf(p, gemma_like_specs());

  const GgufFile f = read_gguf(p);
  REQUIRE(f.version == 3);
  REQUIRE(f.architecture == "testarch");
  REQUIRE(f.alignment == 32);
  REQUIRE(f.tensor_count == gemma_like_specs().size());
  REQUIRE(f.file_size == built.file_size);
  REQUIRE(f.data_start == built.data_start);

  REQUIRE(f.layout.ok());
  REQUIRE(f.layout.unexplained_bytes == 0);
  REQUIRE(f.layout.issues.empty());
  REQUIRE(f.layout.tensor_bytes_total + f.layout.padding_bytes ==
          f.layout.data_bytes);
}

TEST_CASE("GGUF: 決定論 — 同じファイルから同じ manifest", "[unit][gguf]") {
  TempDir tmp("gguf-det");
  const std::string p = tmp.file("model.gguf");
  write_test_gguf(p, gemma_like_specs());

  const GgufFile a = read_gguf(p);
  const GgufFile b = read_gguf(p);
  REQUIRE(to_json(a).dump() == to_json(b).dump());

  const Manifest ma = build_manifest(a);
  const Manifest mb = build_manifest(b);
  REQUIRE(to_json(ma).dump() == to_json(mb).dump());
}

TEST_CASE("GGUF: magic 不一致は Format エラー", "[unit][gguf]") {
  TempDir tmp("gguf-magic");
  const std::string p = tmp.file("bad.gguf");
  {
    std::ofstream out(p, std::ios::binary);
    out << "NOPE";
    for (int i = 0; i < 64; ++i) out.put('\0');
  }
  try {
    (void)read_gguf(p);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Format);
  }
}

TEST_CASE("GGUF: 未対応バージョンは推測せず停止する", "[unit][gguf]") {
  TempDir tmp("gguf-ver");
  const std::string p = tmp.file("v2.gguf");
  write_test_gguf(p, {{"a.weight", 0, {32}}}, 32, /*version=*/2);
  try {
    (void)read_gguf(p);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Unsupported);
  }
}

TEST_CASE("GGUF: 切り詰められたファイルは short read で落ちる", "[unit][gguf]") {
  TempDir tmp("gguf-trunc");
  const std::string p = tmp.file("model.gguf");
  const auto built = write_test_gguf(p, gemma_like_specs());

  // ヘッダの途中で切る
  std::vector<char> head;
  {
    std::ifstream in(p, std::ios::binary);
    head.resize(static_cast<std::size_t>(built.data_start / 2));
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
  }
  const std::string cut = tmp.file("cut.gguf");
  {
    std::ofstream out(cut, std::ios::binary | std::ios::trunc);
    out.write(head.data(), static_cast<std::streamsize>(head.size()));
  }
  REQUIRE_THROWS_AS(read_gguf(cut), Error);
}

TEST_CASE("GGUF: データ領域が短いと out_of_range として報告される",
          "[unit][gguf]") {
  TempDir tmp("gguf-short-data");
  const std::string p = tmp.file("model.gguf");
  const auto built = write_test_gguf(p, gemma_like_specs());

  // データ領域を途中で切る（ヘッダは無傷）
  std::vector<char> all;
  {
    std::ifstream in(p, std::ios::binary);
    all.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const std::string cut = tmp.file("shortdata.gguf");
  {
    std::ofstream out(cut, std::ios::binary | std::ios::trunc);
    const auto keep = static_cast<std::streamsize>(built.data_start + 1024);
    out.write(all.data(), keep);
  }
  const GgufFile f = read_gguf(cut);
  REQUIRE_FALSE(f.layout.ok());
  bool has_oor = false;
  for (const auto& i : f.layout.issues) {
    if (i.kind == "out_of_range") has_oor = true;
  }
  REQUIRE(has_oor);
}

TEST_CASE("GGUF: alignment 非 2 冪は拒否する", "[unit][gguf]") {
  TempDir tmp("gguf-align");
  const std::string p = tmp.file("align.gguf");
  write_test_gguf(p, {{"a.weight", 0, {32}}}, /*alignment=*/48);
  try {
    (void)read_gguf(p);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Format);
  }
}

TEST_CASE("Manifest: 役割分類と実行順序", "[unit][manifest]") {
  TempDir tmp("manifest");
  const std::string p = tmp.file("model.gguf");
  write_test_gguf(p, gemma_like_specs());

  const GgufFile f = read_gguf(p);
  const Manifest m = build_manifest(f);

  REQUIRE(m.block_count == 2);
  REQUIRE(m.unknown_count == 0);
  REQUIRE(m.entries.size() == f.tensors.size());

  // exec_order は 0..n-1 の全順序
  for (std::size_t i = 0; i < m.entries.size(); ++i) {
    REQUIRE(m.entries[i].exec_order == i);
  }

  // pre → blk.0 → blk.1 → post
  REQUIRE(m.stages.size() == 4);
  REQUIRE(m.stages.front().name == "pre");
  REQUIRE(m.stages.back().name == "post");

  // token_embd が最初、output_norm が最後の段
  REQUIRE(m.entries.front().name == "token_embd.weight");
  REQUIRE(m.entries.front().role == TensorRole::Embedding);
  REQUIRE(m.entries.back().name == "output_norm.weight");

  // ブロック内順序: attn_norm が attn_q より先
  u64 order_attn_norm = 0;
  u64 order_attn_q = 0;
  for (const auto& e : m.entries) {
    if (e.name == "blk.0.attn_norm.weight") order_attn_norm = e.exec_order;
    if (e.name == "blk.0.attn_q.weight") order_attn_q = e.exec_order;
  }
  REQUIRE(order_attn_norm < order_attn_q);
}

TEST_CASE("Manifest: 未知のテンソル名は人間の確認を要求する", "[unit][manifest]") {
  TempDir tmp("manifest-unknown");
  const std::string p = tmp.file("model.gguf");
  auto specs = gemma_like_specs();
  specs.push_back({"blk.0.mystery_thing.weight", 0, {32}});
  write_test_gguf(p, specs);

  const GgufFile f = read_gguf(p);
  const Manifest m = build_manifest(f);
  REQUIRE(m.unknown_count == 1);
  REQUIRE(m.requires_human_review);
  REQUIRE_FALSE(m.review_notes.empty());
}

TEST_CASE("Manifest: S_stream の上下限と埋め込み共有の扱い", "[unit][manifest]") {
  TempDir tmp("manifest-stream");
  const std::string p = tmp.file("model.gguf");
  write_test_gguf(p, gemma_like_specs());

  const GgufFile f = read_gguf(p);
  const Manifest m = build_manifest(f);

  const Bytes embd = ggml_tensor_bytes(14, {3840, 1024});
  REQUIRE_FALSE(m.has_separate_output_tensor);
  REQUIRE(m.per_token_bytes_lower == m.total_tensor_bytes - embd);
  REQUIRE(m.per_token_bytes_upper == m.total_tensor_bytes);

  // 上限は下限より大きい（tied embedding を全読みする場合）
  REQUIRE(m.per_token_bytes_upper > m.per_token_bytes_lower);

  const json::Value tbl = bandwidth_ceiling_table(m, {25e9});
  const json::Value& row = tbl.at("rows").as_array().at(0);
  REQUIRE(row.at("t_max_token_per_s_upper_bound_case").as_real() ==
          25e9 / static_cast<double>(m.per_token_bytes_upper));
  REQUIRE_THROWS_AS(bandwidth_ceiling_table(m, {0.0}), Error);
}

TEST_CASE("Residency: 常駐バイト数だけが実効 S_stream を決める", "[unit][manifest]") {
  TempDir tmp("residency");
  const std::string p = tmp.file("model.gguf");
  write_test_gguf(p, gemma_like_specs());

  const GgufFile f = read_gguf(p);
  const Manifest m = build_manifest(f);
  const Bytes total = m.per_token_bytes_upper;

  const json::Value t =
      residency_table(m, {Bytes{0}, total / 4, total / 2, total}, {25e9});
  const auto& rows = t.at("rows").as_array();
  REQUIRE(rows.size() == 4);

  // 常駐 0 なら S_stream そのまま
  REQUIRE(rows[0].at("streamed_bytes_per_token").as_int() ==
          static_cast<i64>(total));
  REQUIRE(rows[0].at("t_max").as_array()[0].at("t_max_token_per_s").as_real() ==
          25e9 / static_cast<double>(total));

  // 常駐が増えるほど毎トークン転送量は単調に減る
  for (std::size_t i = 0; i + 1 < 3; ++i) {
    REQUIRE(rows[i].at("streamed_bytes_per_token").as_int() >
            rows[i + 1].at("streamed_bytes_per_token").as_int());
  }

  // 全部常駐なら「ALSR 不要」と明示する（黙って 0 除算しない）
  REQUIRE(rows[3].get_bool("fully_resident", false));
  REQUIRE_FALSE(rows[3].get_string("note", "").empty());

  // 常駐が S_stream を超えても落ちない
  REQUIRE_NOTHROW(residency_table(m, {total * 2}, {25e9}));
  REQUIRE_THROWS_AS(residency_table(m, {Bytes{0}}, {0.0}), Error);
}
