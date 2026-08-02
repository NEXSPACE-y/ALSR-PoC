#include <catch2/catch_test_macros.hpp>

#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/ggml_types.hpp"
#include "alsr/gguf.hpp"
#include "alsr/host_store.hpp"
#include "alsr/manifest.hpp"
#include "alsr/planner.hpp"
#include "test_helpers.hpp"

using namespace alsr;
using namespace alsr::test;

namespace {

std::vector<TensorSpec> specs() {
  std::vector<TensorSpec> t;
  t.push_back({"output_norm.weight", 0, {3840}});
  t.push_back({"token_embd.weight", 14, {3840, 4096}});  // 12.9 MiB 級
  for (int layer = 0; layer < 4; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    t.push_back({p + "attn_norm.weight", 0, {3840}});
    t.push_back({p + "attn_q.weight", 12, {3840, 512}});
    t.push_back({p + "attn_k.weight", 12, {3840, 256}});
    t.push_back({p + "attn_v.weight", 12, {3840, 256}});
    t.push_back({p + "attn_output.weight", 12, {3840, 512}});
    t.push_back({p + "ffn_gate.weight", 12, {3840, 1024}});
    t.push_back({p + "ffn_up.weight", 12, {3840, 1024}});
    t.push_back({p + "ffn_down.weight", 14, {3840, 1024}});
  }
  return t;
}

struct Fixture {
  TempDir     tmp;
  std::string path;
  GgufFile    gguf;
  Manifest    manifest;

  explicit Fixture(const std::string& tag) : tmp(tag) {
    path = tmp.file("model.gguf");
    write_test_gguf(path, specs());
    gguf = read_gguf(path);
    manifest = build_manifest(gguf);
  }
};

/// 全テンソルの全バイトがちょうど 1 回覆われていることを直接数える。
void assert_exact_coverage(const GgufFile& f, const ChunkPlan& plan) {
  std::map<std::size_t, std::vector<char>> cover;
  for (const auto& t : f.tensors) {
    cover[t.file_index].assign(static_cast<std::size_t>(t.nbytes), 0);
  }
  std::size_t out_of_range = 0;
  std::size_t duplicated = 0;
  for (const Chunk& c : plan.chunks) {
    for (const auto& r : c.tensors) {
      auto& v = cover.at(r.tensor_index);
      for (Bytes k = 0; k < r.bytes; ++k) {
        const std::size_t idx = static_cast<std::size_t>(r.offset_in_tensor + k);
        if (idx >= v.size()) { ++out_of_range; continue; }
        if (v[idx] != 0) ++duplicated;
        v[idx] = 1;
      }
    }
  }
  std::size_t missing = 0;
  for (const auto& [idx, v] : cover) {
    (void)idx;
    for (const char b : v) {
      if (b != 1) ++missing;
    }
  }
  // 集計して 1 回だけ表明する（バイト単位の REQUIRE はアサーション数が爆発する）
  REQUIRE(out_of_range == 0);
  REQUIRE(duplicated == 0);
  REQUIRE(missing == 0);
}

}  // namespace

TEST_CASE("Planner: 完全被覆・欠落なし・重複なし（固定 Chunk）",
          "[unit][planner]") {
  Fixture fx("plan-fixed");
  for (const Bytes mib : {Bytes{1}, Bytes{4}, Bytes{16}, Bytes{64}, Bytes{256}}) {
    PlanOptions opt;
    opt.mode = ChunkMode::Fixed;
    opt.target_bytes = mib * MiB;
    opt.compute_checksums = false;

    INFO("chunk = " << mib << " MiB");
    const ChunkPlan plan = build_plan(fx.gguf, fx.manifest, opt);
    REQUIRE_NOTHROW(verify_plan(fx.gguf, plan));

    // 敷き詰め
    REQUIRE(plan.covered_bytes == fx.gguf.file_size - fx.gguf.data_start);
    REQUIRE(plan.tensor_bytes_total == fx.gguf.layout.tensor_bytes_total);
    assert_exact_coverage(fx.gguf, plan);
  }
}

TEST_CASE("Planner: 可変 Chunk でも不変条件は同じ", "[unit][planner]") {
  Fixture fx("plan-var");
  PlanOptions opt;
  opt.mode = ChunkMode::Variable;
  opt.min_bytes = 2 * MiB;
  opt.max_bytes = 8 * MiB;
  opt.compute_checksums = false;

  const ChunkPlan plan = build_plan(fx.gguf, fx.manifest, opt);
  REQUIRE_NOTHROW(verify_plan(fx.gguf, plan));
  assert_exact_coverage(fx.gguf, plan);
  REQUIRE(plan.covered_bytes == fx.gguf.file_size - fx.gguf.data_start);
}

TEST_CASE("Planner: テンソル内部の切れ目は量子ブロックと alignment を守る",
          "[unit][planner]") {
  Fixture fx("plan-granule");
  PlanOptions opt;
  opt.mode = ChunkMode::Fixed;
  opt.target_bytes = 1 * MiB;  // token_embd(Q6_K) を必ず分割させる
  opt.compute_checksums = false;

  const ChunkPlan plan = build_plan(fx.gguf, fx.manifest, opt);
  REQUIRE(plan.split_tensor_count > 0);

  for (const Chunk& c : plan.chunks) {
    for (const auto& r : c.tensors) {
      if (r.offset_in_tensor == 0) continue;
      const auto& t = fx.gguf.tensors[r.tensor_index];
      const Bytes block = ggml_block_bytes(t.type_id);
      INFO(r.name << " offset_in_tensor=" << r.offset_in_tensor);
      REQUIRE(r.offset_in_tensor % block == 0);          // 量子ブロック境界
      REQUIRE(r.offset_in_tensor % fx.gguf.alignment == 0);  // alignment
    }
  }
}

TEST_CASE("Planner: 決定論 — 同じ入力と設定から同じ計画", "[unit][planner]") {
  Fixture fx("plan-det");
  PlanOptions opt;
  opt.mode = ChunkMode::Fixed;
  opt.target_bytes = 4 * MiB;
  opt.compute_checksums = true;

  const ChunkPlan a = build_plan(fx.gguf, fx.manifest, opt);
  const ChunkPlan b = build_plan(fx.gguf, fx.manifest, opt);
  REQUIRE(a.plan_structure_sha256 == b.plan_structure_sha256);
  REQUIRE(to_json(a).dump() == to_json(b).dump());

  // 設定が変われば構造ハッシュも変わる
  PlanOptions opt2 = opt;
  opt2.target_bytes = 8 * MiB;
  const ChunkPlan c = build_plan(fx.gguf, fx.manifest, opt2);
  REQUIRE(c.plan_structure_sha256 != a.plan_structure_sha256);
}

TEST_CASE("Planner: checksum は実データと一致する", "[unit][planner]") {
  Fixture fx("plan-crc");
  PlanOptions opt;
  opt.mode = ChunkMode::Fixed;
  opt.target_bytes = 2 * MiB;
  opt.compute_checksums = true;

  const ChunkPlan plan = build_plan(fx.gguf, fx.manifest, opt);
  HostStore store(fx.path, HostStoreMode::Read);

  std::vector<u8> buf;
  for (const Chunk& c : plan.chunks) {
    REQUIRE(c.checksum_computed);
    buf.resize(static_cast<std::size_t>(c.bytes));
    store.read_into(c.source_offset, c.bytes, buf.data());
    REQUIRE(crc32c(buf.data(), buf.size()) == c.crc32c);
  }
}

TEST_CASE("Planner: JSON 往復で計画が保たれる", "[unit][planner]") {
  Fixture fx("plan-json");
  PlanOptions opt;
  opt.target_bytes = 4 * MiB;
  opt.compute_checksums = true;

  const ChunkPlan a = build_plan(fx.gguf, fx.manifest, opt);
  const ChunkPlan b = plan_from_json(to_json(a));
  REQUIRE(b.chunks.size() == a.chunks.size());
  REQUIRE(b.exec_sequence == a.exec_sequence);
  for (std::size_t i = 0; i < a.chunks.size(); ++i) {
    REQUIRE(b.chunks[i].source_offset == a.chunks[i].source_offset);
    REQUIRE(b.chunks[i].bytes == a.chunks[i].bytes);
    REQUIRE(b.chunks[i].crc32c == a.chunks[i].crc32c);
  }
  REQUIRE_NOTHROW(verify_plan(fx.gguf, b));
}

TEST_CASE("Planner: 不正な設定は明示的に失敗する", "[unit][planner]") {
  Fixture fx("plan-invalid");
  PlanOptions opt;
  opt.mode = ChunkMode::Fixed;
  opt.target_bytes = 0;
  REQUIRE_THROWS_AS(build_plan(fx.gguf, fx.manifest, opt), Error);

  opt.mode = ChunkMode::Variable;
  opt.min_bytes = 16 * MiB;
  opt.max_bytes = 1 * MiB;
  REQUIRE_THROWS_AS(build_plan(fx.gguf, fx.manifest, opt), Error);

  REQUIRE_THROWS_AS(chunk_mode_from_string("nope"), Error);
}

TEST_CASE("Planner: verify_plan は壊れた計画を検出する", "[unit][planner]") {
  Fixture fx("plan-verify");
  PlanOptions opt;
  opt.target_bytes = 4 * MiB;
  opt.compute_checksums = false;
  ChunkPlan plan = build_plan(fx.gguf, fx.manifest, opt);
  REQUIRE(plan.chunks.size() >= 3);

  SECTION("隙間") {
    ChunkPlan broken = plan;
    broken.chunks[1].source_offset += 32;
    REQUIRE_THROWS_AS(verify_plan(fx.gguf, broken), Error);
  }
  SECTION("テンソル欠落") {
    ChunkPlan broken = plan;
    broken.chunks[1].tensors.clear();
    REQUIRE_THROWS_AS(verify_plan(fx.gguf, broken), Error);
  }
  SECTION("バイト会計不一致") {
    ChunkPlan broken = plan;
    broken.chunks[0].padding_bytes += 1;
    REQUIRE_THROWS_AS(verify_plan(fx.gguf, broken), Error);
  }
  SECTION("末尾まで届かない") {
    ChunkPlan broken = plan;
    broken.chunks.pop_back();
    REQUIRE_THROWS_AS(verify_plan(fx.gguf, broken), Error);
  }
}

TEST_CASE("Planner: exec_sequence は実行順の全順序", "[unit][planner]") {
  Fixture fx("plan-exec");
  PlanOptions opt;
  opt.target_bytes = 4 * MiB;
  opt.compute_checksums = false;
  const ChunkPlan plan = build_plan(fx.gguf, fx.manifest, opt);

  REQUIRE(plan.exec_sequence.size() == plan.chunks.size());
  std::set<std::size_t> seen(plan.exec_sequence.begin(), plan.exec_sequence.end());
  REQUIRE(seen.size() == plan.chunks.size());

  for (std::size_t k = 0; k + 1 < plan.exec_sequence.size(); ++k) {
    const Chunk& a = plan.chunks[plan.exec_sequence[k]];
    const Chunk& b = plan.chunks[plan.exec_sequence[k + 1]];
    REQUIRE(a.first_exec_order <= b.first_exec_order);
  }
  // output_norm がファイル先頭にあるため、実行順は必ずファイル順とずれる
  REQUIRE(plan.exec_order_inversions > 0);
}
