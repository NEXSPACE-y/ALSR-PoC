#include <catch2/catch_test_macros.hpp>

#include "alsr/gguf.hpp"
#include "alsr/json.hpp"
#include "alsr/real_graph_manifest.hpp"
#include "test_helpers.hpp"

using namespace alsr;

namespace {

json::Value make_trace(bool include_unmapped_use = false,
                       bool include_duplicate_instance = false,
                       u64 runtime_bytes = 256) {
  json::Value root = json::Value::make_object();
  root.set("schema", "alsr.llama.graph-trace.v2");
  json::Value runtime = json::Value::make_object();
  runtime.set("name", "llama.cpp");
  runtime.set("tag", "test");
  runtime.set("commit", "0123456789abcdef");
  runtime.set("patch_sha256", "runtime-patch-hash");
  root.set("runtime", std::move(runtime));
  json::Value model = json::Value::make_object();
  model.set("sha256", "model-hash");
  root.set("model", std::move(model));

  json::Value tensors = json::Value::make_array();
  for (const char* name : {"blk.0.ffn_down.weight", "blk.0.attn_q.weight"}) {
    json::Value tensor = json::Value::make_object();
    tensor.set("name", name);
    tensor.set("runtime_instance", 0);
    tensor.set("bytes", runtime_bytes);
    tensor.set("aliases", json::Value::make_array());
    tensors.push_back(std::move(tensor));
  }
  if (include_unmapped_use) {
    json::Value tensor = json::Value::make_object();
    tensor.set("name", "runtime.only.weight");
    tensor.set("runtime_instance", 0);
    tensor.set("bytes", runtime_bytes);
    tensor.set("aliases", json::Value::make_array());
    tensors.push_back(std::move(tensor));
  }
  if (include_duplicate_instance) {
    json::Value tensor = json::Value::make_object();
    tensor.set("name", "blk.0.attn_q.weight");
    tensor.set("runtime_instance", 1);
    tensor.set("bytes", runtime_bytes);
    tensor.set("aliases", json::Value::make_array());
    tensors.push_back(std::move(tensor));
  }
  root.set("runtime_tensors", std::move(tensors));

  json::Value phase = json::Value::make_object();
  phase.set("name", "decode");
  phase.set("input_token_count", 1);
  phase.set("graph_signature_sha256", "graph-hash");
  json::Value nodes = json::Value::make_array();
  {
    json::Value node = json::Value::make_object();
    node.set("order", 0);
    json::Value weights = json::Value::make_array();
    json::Value weight = json::Value::make_object();
    weight.set("name", "blk.0.attn_q.weight");
    weight.set("runtime_instance", 0);
    weights.push_back(std::move(weight));
    node.set("weight_inputs", std::move(weights));
    nodes.push_back(std::move(node));
  }
  if (include_duplicate_instance) {
    json::Value node = json::Value::make_object();
    node.set("order", 2);
    json::Value weights = json::Value::make_array();
    json::Value weight = json::Value::make_object();
    weight.set("name", "blk.0.attn_q.weight");
    weight.set("runtime_instance", 1);
    weights.push_back(std::move(weight));
    node.set("weight_inputs", std::move(weights));
    nodes.push_back(std::move(node));
  }
  {
    json::Value node = json::Value::make_object();
    node.set("order", 1);
    json::Value weights = json::Value::make_array();
    json::Value weight = json::Value::make_object();
    weight.set("name", "blk.0.ffn_down.weight");
    weight.set("runtime_instance", 0);
    weights.push_back(std::move(weight));
    if (include_unmapped_use) {
      json::Value missing = json::Value::make_object();
      missing.set("name", "runtime.only.weight");
      missing.set("runtime_instance", 0);
      weights.push_back(std::move(missing));
    }
    node.set("weight_inputs", std::move(weights));
    nodes.push_back(std::move(node));
  }
  phase.set("nodes", std::move(nodes));
  json::Value phases = json::Value::make_array();
  phases.push_back(std::move(phase));
  root.set("phases", std::move(phases));
  return root;
}

}  // namespace

TEST_CASE("RealGraphManifest: 実graph first-use順をGGUF rangeへ結合する",
          "[unit][real-graph]") {
  test::TempDir tmp("real-graph");
  const std::string path = tmp.file("model.gguf");
  test::write_test_gguf(path,
                        {{"blk.0.ffn_down.weight", 0, {64}},
                         {"blk.0.attn_q.weight", 0, {64}}});
  const GgufFile gguf = read_gguf(path);
  RealGraphManifestOptions options;
  options.target_chunk_bytes = 32;
  const json::Value manifest =
      build_real_graph_manifest(gguf, make_trace(), options);
  REQUIRE(manifest.at("integrity").at("complete").as_bool());
  REQUIRE(manifest.at("identity")
              .at("runtime_patch_sha256")
              .as_string() == "runtime-patch-hash");
  REQUIRE_NOTHROW(verify_real_graph_manifest(gguf, manifest));

  const auto& sequence = manifest.at("demand_plans").as_array().at(0)
                             .at("first_use_demand_sequence").as_array();
  REQUIRE(sequence.size() == 2);
  REQUIRE(sequence.at(0).at("runtime_name").as_string() ==
          "blk.0.attn_q.weight");
  REQUIRE(sequence.at(1).at("runtime_name").as_string() ==
          "blk.0.ffn_down.weight");
}

TEST_CASE("RealGraphManifest: 使用weightがGGUFへ写像できなければ停止する",
          "[unit][real-graph]") {
  test::TempDir tmp("real-graph-unmapped");
  const std::string path = tmp.file("model.gguf");
  test::write_test_gguf(path,
                        {{"blk.0.ffn_down.weight", 0, {64}},
                         {"blk.0.attn_q.weight", 0, {64}}});
  const GgufFile gguf = read_gguf(path);
  RealGraphManifestOptions options;
  options.target_chunk_bytes = 32;
  REQUIRE_THROWS_AS(
      build_real_graph_manifest(gguf, make_trace(true), options), Error);
}

TEST_CASE("RealGraphManifest: 同名の別Runtime実体を需要順で潰さない",
          "[unit][real-graph]") {
  test::TempDir tmp("real-graph-duplicate-instance");
  const std::string path = tmp.file("model.gguf");
  test::write_test_gguf(path,
                        {{"blk.0.ffn_down.weight", 0, {64}},
                         {"blk.0.attn_q.weight", 0, {64}}});
  const GgufFile gguf = read_gguf(path);
  RealGraphManifestOptions options;
  options.target_chunk_bytes = 32;
  const json::Value manifest = build_real_graph_manifest(
      gguf, make_trace(false, true), options);

  const auto& integrity = manifest.at("integrity");
  REQUIRE(integrity.at("runtime_tensor_instance_count").as_int() == 3);
  REQUIRE(integrity.at("used_runtime_tensor_identity_count").as_int() == 3);
  REQUIRE(integrity.at("used_gguf_tensor_count").as_int() == 2);

  const auto& sequence = manifest.at("demand_plans").as_array().at(0)
                             .at("first_use_demand_sequence").as_array();
  REQUIRE(sequence.size() == 3);
  REQUIRE(sequence.at(0).at("runtime_name").as_string() ==
          "blk.0.attn_q.weight");
  REQUIRE(sequence.at(0).at("runtime_instance").as_int() == 0);
  REQUIRE(sequence.at(2).at("runtime_name").as_string() ==
          "blk.0.attn_q.weight");
  REQUIRE(sequence.at(2).at("runtime_instance").as_int() == 1);
  REQUIRE(sequence.at(0).at("gguf_tensor_index").as_int() ==
          sequence.at(2).at("gguf_tensor_index").as_int());
}

TEST_CASE("RealGraphManifest: RuntimeとGGUFのbyte不一致を拒否する",
          "[unit][real-graph]") {
  test::TempDir tmp("real-graph-size-mismatch");
  const std::string path = tmp.file("model.gguf");
  test::write_test_gguf(path,
                        {{"blk.0.ffn_down.weight", 0, {64}},
                         {"blk.0.attn_q.weight", 0, {64}}});
  const GgufFile gguf = read_gguf(path);
  RealGraphManifestOptions options;
  options.target_chunk_bytes = 32;
  REQUIRE_THROWS_AS(
      build_real_graph_manifest(gguf, make_trace(false, false, 32), options),
      Error);
}
