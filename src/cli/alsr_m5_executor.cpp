// alsr-m5-executor — M2 fixed-path M5 Real Executor.
//
// Direct mode is the exact-pin correctness control. Capture mode records the
// original Runtime graph before GGML scheduler mutation. ALSR mode requires the
// graph-derived manifest and authorizes every model-weight use through the
// version-pinned scheduler supply hook. CPU weight repacking is build-disabled.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/resource.h>

#include "alsr/checksum.hpp"
#include "alsr/json.hpp"
#include "alsr/real_supply.hpp"
#include "cli_common.hpp"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama.h"

#ifndef ALSR_LLAMA_CPP_TAG
#define ALSR_LLAMA_CPP_TAG "UNPINNED"
#endif
#ifndef ALSR_LLAMA_CPP_COMMIT
#define ALSR_LLAMA_CPP_COMMIT "UNPINNED"
#endif
#ifndef ALSR_LLAMA_CPP_PATCH_SHA256
#define ALSR_LLAMA_CPP_PATCH_SHA256 "UNPINNED"
#endif
#ifndef ALSR_M2_CPU_REPACK_DISABLED
#error "M2 must be built with CPU weight repacking disabled"
#endif

using namespace alsr;

namespace {

using Clock = std::chrono::steady_clock;

struct LlamaBackendGuard {
  ~LlamaBackendGuard() { llama_backend_free(); }
};

using ModelHandle = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextHandle = std::unique_ptr<llama_context, decltype(&llama_free)>;

void m2_log_callback(enum ggml_log_level level, const char* message, void*) {
  if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
    std::cerr << message;
  }
}

struct WeightIdentity {
  std::string name;
  u64 instance = 0;

  bool operator<(const WeightIdentity& other) const noexcept {
    if (name != other.name) return name < other.name;
    return instance < other.instance;
  }
};

std::string identity_key(const WeightIdentity& identity) {
  return identity.name + "\n" + std::to_string(identity.instance);
}

struct RuntimeTensorGroup {
  const ggml_tensor* tensor = nullptr;
  std::vector<std::string> names;
  std::size_t first_runtime_index = 0;
  std::string canonical;
  u64 instance = 0;
};

std::vector<RuntimeTensorGroup> runtime_tensor_groups(llama_model* model) {
  std::vector<RuntimeTensorGroup> groups;
  std::unordered_map<const ggml_tensor*, std::size_t> by_pointer;
  std::size_t runtime_index = 0;
  for (const auto& [name, tensor] : llama_internal_get_tensor_map(model)) {
    const auto found = by_pointer.find(tensor);
    if (found == by_pointer.end()) {
      by_pointer.emplace(tensor, groups.size());
      groups.push_back({tensor, {name}, runtime_index, {}, 0});
    } else {
      groups[found->second].names.push_back(name);
    }
    ++runtime_index;
  }
  for (auto& group : groups) {
    std::sort(group.names.begin(), group.names.end());
    const std::string runtime_name = ggml_get_name(group.tensor);
    const auto exact =
        std::find(group.names.begin(), group.names.end(), runtime_name);
    group.canonical = exact != group.names.end() ? *exact : group.names.front();
  }
  std::stable_sort(groups.begin(), groups.end(), [](const auto& a,
                                                     const auto& b) {
    if (a.canonical != b.canonical) return a.canonical < b.canonical;
    return a.first_runtime_index < b.first_runtime_index;
  });
  std::map<std::string, u64> next_instance;
  for (auto& group : groups) {
    group.instance = next_instance[group.canonical]++;
  }
  return groups;
}

bool is_weight_passthrough(enum ggml_op op) {
  switch (op) {
    case GGML_OP_NONE:
    case GGML_OP_DUP:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
      return true;
    default:
      return false;
  }
}

void resolve_weight_sources(
    const ggml_tensor* tensor,
    const std::unordered_map<const ggml_tensor*, WeightIdentity>& weights,
    std::set<WeightIdentity>& out, std::set<const ggml_tensor*>& visited,
    int depth = 0) {
  if (tensor == nullptr || depth > 16 || !visited.insert(tensor).second) return;
  const auto exact = weights.find(tensor);
  if (exact != weights.end()) {
    out.insert(exact->second);
    return;
  }
  if (tensor->view_src != nullptr) {
    resolve_weight_sources(tensor->view_src, weights, out, visited, depth + 1);
  }
  if (!is_weight_passthrough(tensor->op)) return;
  for (const auto* source : tensor->src) {
    resolve_weight_sources(source, weights, out, visited, depth + 1);
  }
}

std::set<WeightIdentity> node_weights(
    const ggml_tensor* node,
    const std::unordered_map<const ggml_tensor*, WeightIdentity>& weights) {
  std::set<WeightIdentity> out;
  for (const auto* source : node->src) {
    std::set<const ggml_tensor*> visited;
    resolve_weight_sources(source, weights, out, visited);
  }
  return out;
}

json::Value graph_node_json(
    const ggml_tensor* node, u64 order,
    const std::unordered_map<const ggml_tensor*, WeightIdentity>& weights) {
  json::Value value = json::Value::make_object();
  value.set("order", order);
  value.set("name", std::string(ggml_get_name(node)));
  value.set("op", std::string(ggml_op_name(node->op)));
  value.set("type", std::string(ggml_type_name(node->type)));
  json::Value shape = json::Value::make_array();
  const int dimensions = ggml_n_dims(node);
  for (int i = 0; i < dimensions; ++i) shape.push_back(node->ne[i]);
  value.set("shape", std::move(shape));
  json::Value inputs = json::Value::make_array();
  for (const auto& identity : node_weights(node, weights)) {
    json::Value item = json::Value::make_object();
    item.set("name", identity.name);
    item.set("runtime_instance", identity.instance);
    inputs.push_back(std::move(item));
  }
  value.set("weight_inputs", std::move(inputs));
  return value;
}

struct ManifestChunk {
  u64 id = 0;
  Bytes source_offset = 0;
  Bytes offset_in_tensor = 0;
  Bytes bytes = 0;
  u32 crc32c = 0;
  std::string sha256;
};

struct ManifestMapping {
  std::size_t gguf_tensor_index = 0;
  Bytes source_bytes = 0;
};

struct ExpectedPhase {
  std::string signature;
  std::vector<std::string> demand_keys;
  std::size_t graph_node_count = 0;
  std::size_t input_token_count = 0;
};

struct ObservedPhase {
  std::string signature;
  std::vector<std::string> demand_keys;
  std::size_t graph_node_count = 0;
};

class SupplyController {
 public:
  SupplyController(const std::vector<RuntimeTensorGroup>& groups,
                   const std::string& event_path,
                   const json::Value* manifest, bool enforce,
                   const std::string& model_path,
                   HostStoreMode store_mode, std::size_t ring_depth,
                   std::size_t prefetch_depth)
      : event_out_(event_path, std::ios::trunc), enforce_(enforce),
        start_(Clock::now()) {
    if (!event_out_) fail(ErrorKind::Io, "cannot create M2 supply trace");
    for (const auto& group : groups) {
      const WeightIdentity identity{group.canonical, group.instance};
      runtime_weights_.emplace(group.tensor, identity);
      runtime_tensor_by_key_.emplace(identity_key(identity), group.tensor);
    }
    if (manifest != nullptr) load_manifest(*manifest);
    if (enforce_) {
      Bytes slot_capacity = 0;
      for (const auto& [index, chunks] : chunks_by_gguf_) {
        (void)index;
        for (const auto& chunk : chunks) {
          slot_capacity = std::max(slot_capacity, chunk.bytes);
        }
      }
      if (slot_capacity == 0) {
        fail(ErrorKind::Consistency, "M2 manifest has no tensor supply Chunk");
      }
      pipeline_ = std::make_unique<RealSupplyPipeline>(
          model_path, store_mode, ring_depth, slot_capacity, prefetch_depth,
          [this](const RealSupplyEvent& event) { on_pipeline_event(event); });
    }
  }

  void set_phase(std::string name, std::vector<llama_token> input_tokens) {
    if (phase_active_) fail(ErrorKind::State, "M2 phase already active");
    phase_name_ = std::move(name);
    phase_input_tokens_ = std::move(input_tokens);
    phase_active_ = true;
    graph_complete_seen_ = false;
    if (pipeline_) pipeline_->start_phase(phase_name_, phase_plan(phase_name_));
  }

  void finish_phase() {
    if (!phase_active_ || !graph_complete_seen_) {
      fail(ErrorKind::Consistency, "M2 graph did not reach synchronized completion");
    }
    if (pipeline_) pipeline_->finish_phase();
    phase_active_ = false;
  }

  bool on_event(ggml_backend_sched_alsr_event_data& event) {
    try {
      switch (event.event) {
        case GGML_BACKEND_SCHED_ALSR_GRAPH_BEGIN:
          graph_begin(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_GRAPH_NODE:
          graph_node(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_GRAPH_END:
          graph_end(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_COPY_BEGIN:
          copy_begin(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_RANGE_REQUEST:
          range_request(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_RANGE_COMMITTED:
          range_committed(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_COPY_SUBMITTED:
          copy_submitted(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_SPLIT_NODE:
          split_node(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_SPLIT_SUBMITTED:
          split_submitted(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_GRAPH_REUSE:
          graph_reuse(event);
          break;
        case GGML_BACKEND_SCHED_ALSR_GRAPH_COMPLETE:
          graph_complete(event);
          break;
      }
      return true;
    } catch (const std::exception& error) {
      failed_ = true;
      error_ = error.what();
      std::cerr << "M2_SUPPLY_CALLBACK_ERROR: " << error_ << '\n';
      return false;
    }
  }

  [[nodiscard]] bool failed() const noexcept { return failed_; }
  [[nodiscard]] const std::string& error() const noexcept { return error_; }

  [[nodiscard]] json::Value captured_trace(
      const std::vector<RuntimeTensorGroup>& groups, const std::string& model_path,
      const std::string& model_sha256, llama_model* model,
      const std::string& prompt_sha256, std::size_t prompt_bytes,
      llama_token first_token,
      const std::string& first_piece, u64 n_ctx, u64 threads,
      int32_t n_gpu_layers, bool nonresident_weight_storage_forced_host) const {
    json::Value root = json::Value::make_object();
    root.set("schema", "alsr.llama.graph-trace.v2");
    root.set("mission", "ALSR-M2-M5-REAL-EXECUTOR-01");
    json::Value runtime = json::Value::make_object();
    runtime.set("name", "llama.cpp");
    runtime.set("tag", ALSR_LLAMA_CPP_TAG);
    runtime.set("commit", ALSR_LLAMA_CPP_COMMIT);
    runtime.set("patch_sha256", ALSR_LLAMA_CPP_PATCH_SHA256);
    runtime.set("callback", "GGML_BACKEND_SCHED_ALSR_GRAPH_NODE pre-mutation");
    runtime.set("callback_can_request_prepared_ranges", true);
    runtime.set("callback_requests_tensor_data", enforce_);
    root.set("runtime", std::move(runtime));
    json::Value model_json = json::Value::make_object();
    model_json.set("path", model_path);
    model_json.set("sha256", model_sha256);
    model_json.set("runtime_size_bytes", llama_model_size(model));
    model_json.set("runtime_parameter_count", llama_model_n_params(model));
    model_json.set("runtime_tensor_count", groups.size());
    root.set("model", std::move(model_json));
    json::Value request = json::Value::make_object();
    request.set("prompt_sha256", prompt_sha256);
    request.set("prompt_bytes", prompt_bytes);
    request.set("n_ctx", n_ctx);
    request.set("n_batch", 512);
    request.set("n_ubatch", 512);
    request.set("n_gpu_layers", n_gpu_layers);
    request.set("nonresident_weight_storage_forced_host",
                nonresident_weight_storage_forced_host);
    request.set("threads", threads);
    request.set("sampled_decode_token_id", first_token);
    request.set("sampled_decode_token_piece", first_piece);
    root.set("request", std::move(request));
    json::Value tensors = json::Value::make_array();
    for (const auto& group : groups) {
      json::Value item = json::Value::make_object();
      item.set("name", group.canonical);
      item.set("runtime_instance", group.instance);
      json::Value aliases = json::Value::make_array();
      for (const auto& name : group.names) {
        if (name != group.canonical) aliases.push_back(name);
      }
      item.set("aliases", std::move(aliases));
      item.set("bytes", ggml_nbytes(group.tensor));
      item.set("type", std::string(ggml_type_name(group.tensor->type)));
      json::Value shape = json::Value::make_array();
      const int dimensions = ggml_n_dims(group.tensor);
      for (int i = 0; i < dimensions; ++i) {
        shape.push_back(group.tensor->ne[i]);
      }
      item.set("shape", std::move(shape));
      tensors.push_back(std::move(item));
    }
    root.set("runtime_tensors", std::move(tensors));
    json::Value phases = json::Value::make_array();
    for (const auto& phase : captured_phases_) phases.push_back(phase);
    root.set("phases", std::move(phases));
    return root;
  }

  [[nodiscard]] json::Value summary() const {
    json::Value value = json::Value::make_object();
    value.set("enforcement_enabled", enforce_);
    value.set("graph_count", graph_count_.load());
    value.set("graph_signature_match_count", graph_signature_matches_);
    value.set("graph_demand_match_count", graph_demand_matches_);
    value.set("lease_acquire_count", lease_acquire_count_);
    value.set("lease_retire_count", lease_retire_count_);
    value.set("weight_copy_count", weight_copy_count_);
    value.set("weight_copy_bytes", weight_copy_bytes_);
    value.set("event_protected_split_count", event_protected_split_count_);
    value.set("peak_split_weight_leases", peak_split_weight_leases_);
    value.set("distinct_destination_slots", destination_slots_.size());
    value.set("verified_gguf_tensor_count", verified_gguf_.size());
    value.set("verified_chunk_count", verified_chunk_count_);
    value.set("verified_source_bytes", verified_source_bytes_);
    value.set("prepared_range_count", prepared_range_count_);
    value.set("prepared_range_bytes", prepared_range_bytes_);
    value.set("runtime_source_fallback_count", runtime_source_fallback_count_);
    if (pipeline_) {
      const auto& stats = pipeline_->stats();
      const auto& ring = pipeline_->ring_stats();
      value.set("model_store_mode", std::string(to_string(pipeline_->store_mode())));
      value.set("model_store_bytes_read", pipeline_->store_bytes_read());
      value.set("pipeline_phases_started", stats.phases_started);
      value.set("pipeline_phases_completed", stats.phases_completed);
      value.set("scheduler_decisions", stats.scheduler_decisions);
      value.set("chunks_prepared", stats.chunks_prepared);
      value.set("chunks_retired", stats.chunks_retired);
      value.set("prefetch_hits", stats.prefetch_hits);
      value.set("consumer_waits", stats.consumer_waits);
      value.set("consumer_wait_seconds", stats.consumer_wait_seconds);
      value.set("pipeline_bytes_prepared", stats.bytes_prepared);
      value.set("pipeline_bytes_supplied", stats.bytes_supplied);
      value.set("peak_ready_queue", stats.peak_ready_queue);
      value.set("ring_depth", ring.depth);
      value.set("ring_slot_capacity", ring.slot_capacity);
      value.set("ring_peak_in_flight", ring.peak_in_flight);
      value.set("ring_wrap_arounds", ring.wrap_arounds);
      value.set("ring_illegal_transition_attempts",
                ring.illegal_transition_attempts);
      value.set("pipeline_failed", pipeline_->failed());
      value.set("pipeline_error", pipeline_->error());
    }
    value.set("failed", failed_);
    value.set("error", error_);
    return value;
  }

 private:
  void load_manifest(const json::Value& manifest) {
    if (manifest.get_string("schema", "") != "alsr.real-graph-manifest.v1" ||
        !manifest.at("integrity").at("complete").as_bool()) {
      fail(ErrorKind::Invalid, "M2 requires a complete real graph manifest");
    }
    for (const auto& item : manifest.at("runtime_tensor_mappings").as_array()) {
      if (!item.at("mapped").as_bool() ||
          !item.at("runtime_source_bytes_match").as_bool()) {
        fail(ErrorKind::Consistency, "M2 manifest has incomplete Runtime mapping");
      }
      WeightIdentity identity{item.at("runtime_name").as_string(),
                              static_cast<u64>(item.at("runtime_instance").as_int())};
      ManifestMapping mapping;
      mapping.gguf_tensor_index = static_cast<std::size_t>(
          item.at("gguf_tensor_index").as_int());
      mapping.source_bytes = static_cast<Bytes>(item.at("source_bytes").as_int());
      mappings_.emplace(identity_key(identity), mapping);
    }
    for (const auto& chunk : manifest.at("storage_plan").at("chunks").as_array()) {
      if (chunk.at("kind").as_string() != "tensor") continue;
      if (!chunk.at("checksum_computed").as_bool()) {
        fail(ErrorKind::Consistency, "M2 manifest tensor Chunk lacks checksum");
      }
      ManifestChunk parsed;
      parsed.id = static_cast<u64>(chunk.at("chunk_id").as_int());
      parsed.source_offset = static_cast<Bytes>(
          chunk.at("source_offset").as_int());
      parsed.offset_in_tensor = static_cast<Bytes>(
          chunk.at("offset_in_tensor").as_int());
      parsed.bytes = static_cast<Bytes>(chunk.at("bytes").as_int());
      parsed.crc32c = static_cast<u32>(chunk.at("crc32c").as_int());
      parsed.sha256 = chunk.at("sha256").as_string();
      const auto index = static_cast<std::size_t>(
          chunk.at("tensor_index").as_int());
      chunks_by_gguf_[index].push_back(std::move(parsed));
    }
    for (const auto& phase : manifest.at("demand_plans").as_array()) {
      ExpectedPhase expected;
      expected.signature = phase.at("graph_signature_sha256").as_string();
      expected.graph_node_count = static_cast<std::size_t>(
          phase.at("graph_node_count").as_int());
      expected.input_token_count = static_cast<std::size_t>(
          phase.at("input_token_count").as_int());
      for (const auto& item :
           phase.at("first_use_demand_sequence").as_array()) {
        WeightIdentity identity{
            item.at("runtime_name").as_string(),
            static_cast<u64>(item.at("runtime_instance").as_int())};
        expected.demand_keys.push_back(identity_key(identity));
      }
      expected_phases_.emplace(phase.at("phase").as_string(),
                               std::move(expected));
    }
  }

  void write_event(json::Value event) {
    std::lock_guard lock(event_mutex_);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start_).count();
    event.set("elapsed_us", elapsed);
    event.set("phase", phase_name_);
    event.set("graph_index", graph_count_.load());
    event_out_ << event.dump() << '\n';
    if (!event_out_) fail(ErrorKind::Io, "cannot write M2 supply trace");
  }

  std::vector<RealSupplyChunk> phase_plan(const std::string& phase) const {
    const auto expected = expected_phases_.find(phase);
    if (expected == expected_phases_.end()) {
      fail(ErrorKind::Consistency, "M2 manifest lacks supply phase " + phase);
    }
    std::vector<RealSupplyChunk> plan;
    for (const auto& key : expected->second.demand_keys) {
      const auto tensor = runtime_tensor_by_key_.find(key);
      const auto mapping = mappings_.find(key);
      if (tensor == runtime_tensor_by_key_.end() || mapping == mappings_.end()) {
        fail(ErrorKind::Consistency, "M2 supply phase has unmapped Runtime weight");
      }
      if (!ggml_backend_buffer_is_host(tensor->second->buffer)) continue;
      const auto chunks = chunks_by_gguf_.find(mapping->second.gguf_tensor_index);
      if (chunks == chunks_by_gguf_.end()) {
        fail(ErrorKind::Consistency, "M2 supply phase has no storage Chunks");
      }
      for (const auto& chunk : chunks->second) {
        plan.push_back(RealSupplyChunk{
            .chunk_id = chunk.id,
            .tensor_key = key,
            .source_offset = chunk.source_offset,
            .offset_in_tensor = chunk.offset_in_tensor,
            .bytes = chunk.bytes,
            .crc32c = chunk.crc32c,
            .sha256 = chunk.sha256});
      }
    }
    return plan;
  }

  void on_pipeline_event(const RealSupplyEvent& source) {
    json::Value event = json::Value::make_object();
    event.set("event", source.event);
    event.set("supply_phase", source.phase);
    if (!source.tensor_key.empty()) event.set("tensor_key", source.tensor_key);
    if (source.chunk_id != 0) event.set("chunk_id", source.chunk_id);
    event.set("slot_id", source.slot_id);
    event.set("slot_generation", source.generation);
    event.set("source_offset", source.source_offset);
    event.set("offset_in_tensor", source.offset_in_tensor);
    event.set("bytes", source.bytes);
    if (!source.from_state.empty()) event.set("from_state", source.from_state);
    if (!source.to_state.empty()) event.set("to_state", source.to_state);
    event.set("queue_depth", source.queue_depth);
    event.set("prefetch_depth", source.prefetch_depth);
    event.set("prepare_ms", source.prepare_ms);
    if (source.transition_sequence != 0) {
      event.set("transition_sequence", source.transition_sequence);
    }
    if (source.event == "SLOT_STATE" && source.to_state == "READY") {
      if (verified_chunk_ids_.insert(source.chunk_id).second) {
        ++verified_chunk_count_;
        verified_source_bytes_ += source.bytes;
      }
    }
    write_event(std::move(event));
  }

  void graph_begin(const ggml_backend_sched_alsr_event_data& event) {
    if (!phase_active_) fail(ErrorKind::State, "M2 graph began outside a phase");
    if (graph_open_) fail(ErrorKind::State, "nested M2 graph");
    graph_open_ = true;
    graph_uid_ = event.graph_uid;
    graph_nodes_ = json::Value::make_array();
    graph_weight_keys_.clear();
    copy_weights_.clear();
    graph_lease_keys_.clear();
    current_split_keys_.clear();
    current_split_has_copied_weight_ = false;
    current_split_id_ = -1;
    ++graph_count_;
    json::Value record = json::Value::make_object();
    record.set("event", "GRAPH_BEGIN");
    record.set("graph_uid", graph_uid_);
    write_event(std::move(record));
  }

  void graph_node(const ggml_backend_sched_alsr_event_data& event) {
    if (!graph_open_ || event.node == nullptr) {
      fail(ErrorKind::State, "invalid M2 graph node event");
    }
    json::Value node = graph_node_json(
        event.node, static_cast<u64>(event.node_index), runtime_weights_);
    for (const auto& input : node.at("weight_inputs").as_array()) {
      WeightIdentity identity{
          input.at("name").as_string(),
          static_cast<u64>(input.at("runtime_instance").as_int())};
      graph_weight_keys_.insert(identity_key(identity));
    }
    graph_nodes_.push_back(std::move(node));
  }

  void graph_end(const ggml_backend_sched_alsr_event_data&) {
    if (!graph_open_) fail(ErrorKind::State, "M2 graph end without begin");
    json::Value stable = json::Value::make_object();
    stable.set("name", phase_name_);
    stable.set("input_token_count", phase_input_tokens_.size());
    stable.set("nodes", graph_nodes_);
    const std::string serialized = stable.dump();
    const std::string signature =
        sha256_hex(serialized.data(), serialized.size());

    std::vector<std::string> demand_keys;
    std::set<std::string> seen;
    for (const auto& node : graph_nodes_.as_array()) {
      for (const auto& input : node.at("weight_inputs").as_array()) {
        WeightIdentity identity{
            input.at("name").as_string(),
            static_cast<u64>(input.at("runtime_instance").as_int())};
        const std::string key = identity_key(identity);
        if (seen.insert(key).second) demand_keys.push_back(key);
      }
    }

    if (enforce_) {
      const auto found = expected_phases_.find(phase_name_);
      if (found == expected_phases_.end()) {
        fail(ErrorKind::Consistency, "M2 manifest lacks phase " + phase_name_);
      }
      const auto& expected = found->second;
      if (signature != expected.signature ||
          graph_nodes_.as_array().size() != expected.graph_node_count ||
          phase_input_tokens_.size() != expected.input_token_count) {
        fail(ErrorKind::Consistency, "M2 graph identity differs from manifest");
      }
      ++graph_signature_matches_;
      if (demand_keys != expected.demand_keys) {
        fail(ErrorKind::Consistency, "M2 graph demand order differs from manifest");
      }
      ++graph_demand_matches_;
    }

    const ObservedPhase observed{signature, demand_keys,
                                 graph_nodes_.as_array().size()};
    const auto [observed_it, observed_inserted] =
        observed_phases_.emplace(phase_name_, observed);
    if (!observed_inserted &&
        (observed_it->second.signature != observed.signature ||
         observed_it->second.demand_keys != observed.demand_keys ||
         observed_it->second.graph_node_count != observed.graph_node_count)) {
      fail(ErrorKind::Consistency,
           "M2 rebuilt graph differs from earlier graph in the same phase");
    }

    if (captured_phase_names_.insert(phase_name_).second) {
      json::Value phase = json::Value::make_object();
      phase.set("name", phase_name_);
      phase.set("input_token_count", phase_input_tokens_.size());
      json::Value tokens = json::Value::make_array();
      for (const auto token : phase_input_tokens_) tokens.push_back(token);
      phase.set("input_token_ids", std::move(tokens));
      phase.set("graph_signature_sha256", signature);
      phase.set("nodes", graph_nodes_);
      captured_phases_.push_back(std::move(phase));
    }

    json::Value record = json::Value::make_object();
    record.set("event", "GRAPH_IDENTITY_VERIFIED");
    record.set("graph_uid", graph_uid_);
    record.set("signature_sha256", signature);
    record.set("node_count", graph_nodes_.as_array().size());
    record.set("weight_identity_count", demand_keys.size());
    record.set("manifest_enforced", enforce_);
    write_event(std::move(record));
  }

  void graph_reuse(const ggml_backend_sched_alsr_event_data& event) {
    if (!phase_active_) fail(ErrorKind::State, "M2 graph reuse outside a phase");
    if (graph_open_) fail(ErrorKind::State, "M2 graph reuse while graph is open");
    const auto observed = observed_phases_.find(phase_name_);
    if (observed == observed_phases_.end()) {
      fail(ErrorKind::Consistency,
           "M2 scheduler reused a graph before its identity was observed");
    }
    graph_open_ = true;
    graph_uid_ = event.graph_uid;
    graph_nodes_ = json::Value::make_array();
    graph_weight_keys_.clear();
    graph_weight_keys_.insert(observed->second.demand_keys.begin(),
                              observed->second.demand_keys.end());
    copy_weights_.clear();
    graph_lease_keys_.clear();
    current_split_keys_.clear();
    current_split_has_copied_weight_ = false;
    current_split_id_ = -1;
    ++graph_count_;

    if (enforce_) {
      const auto expected = expected_phases_.find(phase_name_);
      if (expected == expected_phases_.end() ||
          expected->second.signature != observed->second.signature ||
          expected->second.demand_keys != observed->second.demand_keys ||
          expected->second.graph_node_count != observed->second.graph_node_count ||
          expected->second.input_token_count != phase_input_tokens_.size()) {
        fail(ErrorKind::Consistency,
             "M2 cached graph identity differs from manifest");
      }
      ++graph_signature_matches_;
      ++graph_demand_matches_;
    }

    json::Value record = json::Value::make_object();
    record.set("event", "GRAPH_REUSE_IDENTITY_VERIFIED");
    record.set("graph_uid", graph_uid_);
    record.set("signature_sha256", observed->second.signature);
    record.set("node_count", observed->second.graph_node_count);
    record.set("weight_identity_count", observed->second.demand_keys.size());
    record.set("manifest_enforced", enforce_);
    record.set("scheduler_reused_allocated_graph", true);
    write_event(std::move(record));
  }

  void begin_split(int split_id) {
    if (current_split_id_ == -1) {
      current_split_id_ = split_id;
    } else if (current_split_id_ != split_id) {
      fail(ErrorKind::State, "M2 split advanced before submission");
    }
  }

  std::set<WeightIdentity> resolve_with_copies(const ggml_tensor* tensor) const {
    std::unordered_map<const ggml_tensor*, WeightIdentity> combined =
        runtime_weights_;
    for (const auto& [pointer, identity] : copy_weights_) {
      combined.insert_or_assign(pointer, identity);
    }
    return node_weights(tensor, combined);
  }

  void acquire(const WeightIdentity& identity, const std::string& supply_mode,
               int split_id, u64 slot_id = 0) {
    begin_split(split_id);
    const std::string key = identity_key(identity);
    if (!graph_weight_keys_.contains(key)) {
      fail(ErrorKind::Consistency,
           "M2 supplied weight absent from original graph: " + identity.name +
               "#" + std::to_string(identity.instance));
    }
    if (!current_split_keys_.insert(key).second) return;
    graph_lease_keys_.insert(key);
    ++lease_acquire_count_;
    peak_split_weight_leases_ =
        std::max(peak_split_weight_leases_, current_split_keys_.size());
    json::Value record = json::Value::make_object();
    record.set("event", "LEASE_ACQUIRE");
    record.set("split_id", split_id);
    record.set("runtime_name", identity.name);
    record.set("runtime_instance", identity.instance);
    record.set("supply_mode", supply_mode);
    if (slot_id != 0) record.set("destination_slot_id", slot_id);
    write_event(std::move(record));
  }

  WeightIdentity single_source_identity(const ggml_tensor* tensor) const {
    std::set<WeightIdentity> identities;
    std::set<const ggml_tensor*> visited;
    resolve_weight_sources(tensor, runtime_weights_, identities, visited);
    if (identities.size() != 1) {
      fail(ErrorKind::Consistency,
           "M2 weight copy does not resolve to exactly one Runtime identity");
    }
    return *identities.begin();
  }

  void copy_begin(const ggml_backend_sched_alsr_event_data& event) {
    if (!event.source_is_weight) return;
    if (event.source == nullptr || event.destination == nullptr) {
      fail(ErrorKind::State, "M2 weight copy lacks source/destination");
    }
    const WeightIdentity identity = single_source_identity(event.source);
    copy_weights_.insert_or_assign(event.destination, identity);
    const void* slot_pointer = event.destination->data;
    auto [slot_it, inserted] = destination_slots_.emplace(
        slot_pointer, static_cast<u64>(destination_slots_.size() + 1));
    (void)inserted;
    acquire(identity,
            enforce_ ? "EXPLICIT_RANGE_RING_TO_METAL"
                     : "RUNTIME_MMAP_TO_METAL_SPLIT_COPY",
            event.split_id,
            slot_it->second);
    current_split_has_copied_weight_ = true;
    ++weight_copy_count_;
    weight_copy_bytes_ += ggml_nbytes(event.source);
    json::Value record = json::Value::make_object();
    record.set("event", "WEIGHT_COPY_BEGIN");
    record.set("split_id", event.split_id);
    record.set("input_index", event.input_index);
    record.set("runtime_name", identity.name);
    record.set("runtime_instance", identity.instance);
    record.set("bytes", ggml_nbytes(event.source));
    record.set("destination_slot_id", slot_it->second);
    record.set("source_backend", std::string(ggml_backend_name(event.source_backend)));
    record.set("destination_backend",
               std::string(ggml_backend_name(event.destination_backend)));
    write_event(std::move(record));
  }

  void range_request(ggml_backend_sched_alsr_event_data& event) {
    if (!event.source_is_weight || !enforce_) return;
    if (!pipeline_ || event.source == nullptr) {
      fail(ErrorKind::State, "M2 prepared range request lacks pipeline/source");
    }
    const WeightIdentity identity = single_source_identity(event.source);
    const std::string key = identity_key(identity);
    const auto mapping = mappings_.find(key);
    if (mapping == mappings_.end()) {
      fail(ErrorKind::Consistency, "M2 range request has no manifest mapping");
    }
    const PreparedRange range = pipeline_->acquire(
        key, static_cast<Bytes>(event.requested_source_offset),
        static_cast<Bytes>(event.requested_source_bytes));
    event.prepared_data = range.data;
    event.prepared_bytes = static_cast<std::size_t>(range.bytes);
    event.prepared_slot_id = static_cast<u64>(range.slot.slot_id);
    event.prepared_slot_generation = range.slot.generation;
    event.prepared_data_valid = true;
    verified_gguf_.insert(mapping->second.gguf_tensor_index);

    json::Value record = json::Value::make_object();
    record.set("event", "PREPARED_RANGE_BOUND");
    record.set("split_id", event.split_id);
    record.set("runtime_name", identity.name);
    record.set("runtime_instance", identity.instance);
    record.set("chunk_id", range.chunk_id);
    record.set("slot_id", range.slot.slot_id);
    record.set("slot_generation", range.slot.generation);
    record.set("offset_in_tensor", range.offset_in_tensor);
    record.set("bytes", range.bytes);
    write_event(std::move(record));
  }

  void range_committed(const ggml_backend_sched_alsr_event_data& event) {
    if (!event.source_is_weight || !enforce_) return;
    if (!event.prepared_data_valid || !pipeline_) {
      ++runtime_source_fallback_count_;
      fail(ErrorKind::Consistency,
           "M2 weight copy committed without ALSR prepared data");
    }
    const SlotRef slot{static_cast<std::size_t>(event.prepared_slot_id),
                       event.prepared_slot_generation};
    pipeline_->retire(slot);
    ++prepared_range_count_;
    prepared_range_bytes_ +=
        static_cast<Bytes>(event.requested_source_bytes);

    json::Value record = json::Value::make_object();
    record.set("event", "PREPARED_RANGE_COMMITTED");
    record.set("split_id", event.split_id);
    record.set("slot_id", slot.slot_id);
    record.set("slot_generation", slot.generation);
    record.set("offset_in_tensor", event.requested_source_offset);
    record.set("bytes", event.requested_source_bytes);
    record.set("prepared_source_consumed_before_retire", true);
    record.set("copy_submission", "SYNCHRONOUS_TENSOR_SET");
    write_event(std::move(record));
  }

  void copy_submitted(const ggml_backend_sched_alsr_event_data& event) {
    if (!event.source_is_weight) return;
    const WeightIdentity identity = single_source_identity(event.source);
    json::Value record = json::Value::make_object();
    record.set("event", "WEIGHT_COPY_SUBMITTED");
    record.set("split_id", event.split_id);
    record.set("runtime_name", identity.name);
    record.set("runtime_instance", identity.instance);
    write_event(std::move(record));
  }

  void split_node(const ggml_backend_sched_alsr_event_data& event) {
    if (event.node == nullptr) return;
    begin_split(event.split_id);
    for (const auto& identity : resolve_with_copies(event.node)) {
      acquire(identity, "MMAP_DIRECT_OR_PREPARED_COPY", event.split_id);
    }
  }

  void split_submitted(const ggml_backend_sched_alsr_event_data& event) {
    begin_split(event.split_id);
    if (current_split_has_copied_weight_ &&
        event.completion_event == nullptr && enforce_) {
      fail(ErrorKind::Consistency,
           "M2 copied-weight lease lacks completion event");
    }
    if (!current_split_keys_.empty() && event.completion_event != nullptr) {
      ++event_protected_split_count_;
    }
    json::Value record = json::Value::make_object();
    record.set("event", "SPLIT_SUBMITTED");
    record.set("split_id", event.split_id);
    record.set("weight_lease_count", current_split_keys_.size());
    record.set("copied_weight_present", current_split_has_copied_weight_);
    record.set("completion_event_backed", event.completion_event != nullptr);
    record.set("lifetime_guard",
               current_split_has_copied_weight_
                   ? "BACKEND_COMPLETION_EVENT"
                   : "IMMUTABLE_MODEL_MMAP_UNTIL_GRAPH_SYNC");
    record.set("destination_backend",
               std::string(ggml_backend_name(event.destination_backend)));
    write_event(std::move(record));
    current_split_keys_.clear();
    current_split_has_copied_weight_ = false;
    current_split_id_ = -1;
  }

  void graph_complete(const ggml_backend_sched_alsr_event_data&) {
    if (!graph_open_) fail(ErrorKind::State, "M2 completion without open graph");
    if (!current_split_keys_.empty()) {
      fail(ErrorKind::State, "M2 graph completed with open split lease");
    }
    if (enforce_) {
      const auto found = expected_phases_.find(phase_name_);
      if (found == expected_phases_.end() ||
          graph_lease_keys_.size() != found->second.demand_keys.size()) {
        fail(ErrorKind::Consistency, "M2 did not lease every graph weight");
      }
    }
    lease_retire_count_ += graph_lease_keys_.size();
    json::Value record = json::Value::make_object();
    record.set("event", "GRAPH_COMPLETE_AND_LEASES_RETIRED");
    record.set("graph_uid", graph_uid_);
    record.set("retired_weight_lease_count", graph_lease_keys_.size());
    record.set("backend_synchronized", true);
    write_event(std::move(record));
    graph_lease_keys_.clear();
    graph_open_ = false;
    graph_complete_seen_ = true;
  }

  std::ofstream event_out_;
  mutable std::mutex event_mutex_;
  bool enforce_ = false;
  Clock::time_point start_;
  std::unordered_map<const ggml_tensor*, WeightIdentity> runtime_weights_;
  std::map<std::string, const ggml_tensor*> runtime_tensor_by_key_;
  std::unordered_map<const ggml_tensor*, WeightIdentity> copy_weights_;
  std::map<std::string, ManifestMapping> mappings_;
  std::map<std::size_t, std::vector<ManifestChunk>> chunks_by_gguf_;
  std::map<std::string, ExpectedPhase> expected_phases_;
  std::map<std::string, ObservedPhase> observed_phases_;
  std::map<const void*, u64> destination_slots_;
  std::set<std::size_t> verified_gguf_;
  std::set<u64> verified_chunk_ids_;
  std::set<std::string> captured_phase_names_;
  std::vector<json::Value> captured_phases_;
  json::Value graph_nodes_ = json::Value::make_array();
  std::set<std::string> graph_weight_keys_;
  std::set<std::string> graph_lease_keys_;
  std::set<std::string> current_split_keys_;
  std::string phase_name_;
  std::vector<llama_token> phase_input_tokens_;
  u64 graph_uid_ = 0;
  int current_split_id_ = -1;
  bool phase_active_ = false;
  bool graph_open_ = false;
  bool graph_complete_seen_ = false;
  bool current_split_has_copied_weight_ = false;
  bool failed_ = false;
  std::string error_;
  std::atomic<u64> graph_count_{0};
  u64 graph_signature_matches_ = 0;
  u64 graph_demand_matches_ = 0;
  u64 lease_acquire_count_ = 0;
  u64 lease_retire_count_ = 0;
  u64 weight_copy_count_ = 0;
  Bytes weight_copy_bytes_ = 0;
  u64 event_protected_split_count_ = 0;
  std::size_t peak_split_weight_leases_ = 0;
  u64 verified_chunk_count_ = 0;
  Bytes verified_source_bytes_ = 0;
  u64 prepared_range_count_ = 0;
  Bytes prepared_range_bytes_ = 0;
  u64 runtime_source_fallback_count_ = 0;
  std::unique_ptr<RealSupplyPipeline> pipeline_;
};

bool supply_callback(ggml_backend_sched_alsr_event_data* event,
                     void* user_data) {
  if (event == nullptr || user_data == nullptr) return false;
  return static_cast<SupplyController*>(user_data)->on_event(*event);
}

std::string token_piece(const llama_vocab* vocab, llama_token token) {
  std::vector<char> buffer(256);
  int32_t size = llama_token_to_piece(vocab, token, buffer.data(),
                                      static_cast<int32_t>(buffer.size()), 0,
                                      true);
  if (size < 0) {
    buffer.resize(static_cast<std::size_t>(-size));
    size = llama_token_to_piece(vocab, token, buffer.data(),
                                static_cast<int32_t>(buffer.size()), 0, true);
  }
  if (size < 0) fail(ErrorKind::State, "cannot decode generated token piece");
  return std::string(buffer.data(), static_cast<std::size_t>(size));
}

struct SampleResult {
  llama_token token = 0;
  json::Value top = json::Value::make_array();
  double selected_logprob = 0.0;
};

SampleResult greedy_top_logits(llama_context* context, const llama_vocab* vocab,
                               int top_count) {
  const int n_vocab = llama_vocab_n_tokens(vocab);
  const float* logits = llama_get_logits_ith(context, -1);
  if (logits == nullptr || n_vocab <= 0) {
    fail(ErrorKind::State, "M2 logits unavailable");
  }
  std::vector<llama_token> ids(static_cast<std::size_t>(n_vocab));
  std::iota(ids.begin(), ids.end(), 0);
  const auto better = [&](llama_token a, llama_token b) {
    if (logits[a] != logits[b]) return logits[a] > logits[b];
    return a < b;
  };
  const std::size_t take = std::min<std::size_t>(
      static_cast<std::size_t>(top_count), ids.size());
  std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(take),
                    ids.end(), better);
  const llama_token selected = ids.front();
  if (!std::isfinite(logits[selected])) {
    fail(ErrorKind::Consistency, "M2 selected logit is not finite");
  }
  double max_logit = -INFINITY;
  for (int i = 0; i < n_vocab; ++i) {
    max_logit = std::max(max_logit, static_cast<double>(logits[i]));
  }
  double sum = 0.0;
  for (int i = 0; i < n_vocab; ++i) {
    sum += std::exp(static_cast<double>(logits[i]) - max_logit);
  }
  const double log_z = max_logit + std::log(sum);
  SampleResult result;
  result.token = selected;
  result.selected_logprob = static_cast<double>(logits[selected]) - log_z;
  for (std::size_t rank = 0; rank < take; ++rank) {
    const llama_token id = ids[rank];
    json::Value item = json::Value::make_object();
    item.set("rank", rank + 1);
    item.set("token_id", id);
    item.set("piece", token_piece(vocab, id));
    item.set("logit", static_cast<double>(logits[id]));
    item.set("logprob", static_cast<double>(logits[id]) - log_z);
    result.top.push_back(std::move(item));
  }
  return result;
}

json::Value model_buffer_inventory(
    const std::vector<RuntimeTensorGroup>& groups, Bytes model_runtime_bytes,
    bool enforce_residency_limit) {
  std::map<ggml_backend_buffer_t, std::size_t> tensor_counts;
  for (const auto& group : groups) {
    ggml_backend_buffer_t buffer = group.tensor->view_src
                                       ? group.tensor->view_src->buffer
                                       : group.tensor->buffer;
    if (buffer == nullptr) fail(ErrorKind::Consistency, "model tensor has no buffer");
    ++tensor_counts[buffer];
  }
  json::Value inventory = json::Value::make_object();
  json::Value buffers = json::Value::make_array();
  Bytes total = 0;
  bool all_host = true;
  bool repack_seen = false;
  Bytes resident_model_bytes = 0;
  for (const auto& [buffer, count] : tensor_counts) {
    const std::string name = ggml_backend_buffer_name(buffer);
    const Bytes bytes = ggml_backend_buffer_get_size(buffer);
    const bool host = ggml_backend_buffer_is_host(buffer);
    all_host = all_host && host;
    if (!host) resident_model_bytes += bytes;
    repack_seen = repack_seen || name.find("REPACK") != std::string::npos;
    total += bytes;
    json::Value item = json::Value::make_object();
    item.set("name", name);
    item.set("bytes", bytes);
    item.set("is_host", host);
    item.set("runtime_tensor_instances", count);
    buffers.push_back(std::move(item));
  }
  constexpr Bytes kM2ResidentModelLimit = 1 * GiB;
  const Bytes bytes_beyond_source_mapping =
      total > model_runtime_bytes ? total - model_runtime_bytes : 0;
  const bool hidden_second_full_load =
      repack_seen ||
      bytes_beyond_source_mapping > kM2ResidentModelLimit + 64 * MiB;
  if (enforce_residency_limit &&
      (hidden_second_full_load ||
       resident_model_bytes > kM2ResidentModelLimit)) {
    fail(ErrorKind::Consistency,
         "M2 hidden-load/residency guard rejected model buffers");
  }
  inventory.set("buffers", std::move(buffers));
  inventory.set("unique_buffer_count", tensor_counts.size());
  inventory.set("unique_buffer_bytes", total);
  inventory.set("all_model_buffers_host", all_host);
  inventory.set("resident_model_bytes", resident_model_bytes);
  inventory.set("resident_model_limit_bytes", kM2ResidentModelLimit);
  inventory.set("bytes_beyond_source_mapping", bytes_beyond_source_mapping);
  inventory.set("cpu_repack_seen", repack_seen);
  inventory.set("hidden_second_full_load_detected", hidden_second_full_load);
  return inventory;
}

json::Value scheduler_buffers(llama_context* context) {
  ggml_backend_sched_t scheduler = context->get_sched();
  json::Value buffers = json::Value::make_array();
  const int count = ggml_backend_sched_get_n_backends(scheduler);
  for (int i = 0; i < count; ++i) {
    ggml_backend_t backend = ggml_backend_sched_get_backend(scheduler, i);
    json::Value item = json::Value::make_object();
    item.set("backend", std::string(ggml_backend_name(backend)));
    item.set("bytes", ggml_backend_sched_get_buffer_size(scheduler, backend));
    buffers.push_back(std::move(item));
  }
  return buffers;
}

std::vector<llama_token> int_array_tokens(const json::Value& value,
                                          std::string_view field) {
  std::vector<llama_token> out;
  for (const auto& item : value.at(field).as_array()) {
    out.push_back(static_cast<llama_token>(item.as_int()));
  }
  return out;
}

void write_json_file(const std::string& path, const json::Value& value) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) fail(ErrorKind::Io, "cannot write " + path);
  out << value.dump(2) << '\n';
}

void usage() {
  std::cout << R"(alsr-m5-executor — M2 fixed-path Real Executor

Usage:
  alsr-m5-executor --mode <capture|direct|alsr> --model <GGUF>
    --model-sha256 <hex> --baseline <M0_TOKEN_ID_BASELINE.json>
    --out-dir <directory> [--manifest <real-manifest.json>]
    [--tokens <n>] [--n-ctx <n>] [--threads <n>] [--top-logits <n>]
    [--host-store <read|mmap>] [--ring-depth <n>] [--prefetch-depth <n>]

capture observes Runtime mmap -> Metal split copies. alsr requires explicit
Model Store range reads into a finite prepared Ring and makes those prepared
bytes the actual Metal copy source. It also enforces the exact graph manifest.
direct is the exact-pin full-Metal correctness control.
)";
}

}  // namespace

int main(int argc, char** argv) {
  llama_log_set(m2_log_callback, nullptr);
  return cli::run_main([&]() -> int {
    cli::Args args(argc, argv);
    if (args.flag("help") || argc == 1) {
      usage();
      return 0;
    }
    args.reject_unknown({"mode", "model", "model-sha256", "baseline",
                         "manifest", "out-dir", "tokens", "n-ctx",
                         "threads", "top-logits", "host-store",
                         "ring-depth", "prefetch-depth", "help"});
    for (const char* required : {"mode", "model", "model-sha256", "baseline",
                                 "out-dir"}) {
      if (!args.has(required)) {
        fail(ErrorKind::Invalid, std::string("--") + required + " is required");
      }
    }
    const std::string mode = args.get("mode");
    if (mode != "capture" && mode != "direct" && mode != "alsr") {
      fail(ErrorKind::Invalid, "--mode must be capture, direct, or alsr");
    }
    if (mode == "alsr" && !args.has("manifest")) {
      fail(ErrorKind::Invalid, "ALSR mode requires --manifest");
    }
    if (std::string(ALSR_LLAMA_CPP_COMMIT) == "UNPINNED") {
      fail(ErrorKind::Consistency, "M2 Runtime is not commit-pinned");
    }
    const bool hooked = mode != "direct";
    const bool enforce = mode == "alsr";
    if (hooked) {
      const char* threshold = std::getenv("GGML_OP_OFFLOAD_MIN_BATCH");
      if (threshold == nullptr || std::string(threshold) != "0") {
        fail(ErrorKind::Consistency,
             "M2 capture/ALSR requires GGML_OP_OFFLOAD_MIN_BATCH=0");
      }
    }

    const u64 generated_count = args.get_u64("tokens", 128);
    const u64 n_ctx = args.get_u64("n-ctx", 4096);
    const u64 threads = args.get_u64("threads", 8);
    const int top_count = static_cast<int>(args.get_u64("top-logits", 20));
    const HostStoreMode store_mode = host_store_mode_from_string(
        args.get("host-store", "read"));
    const std::size_t ring_depth = static_cast<std::size_t>(
        args.get_u64("ring-depth", 2));
    const std::size_t prefetch_depth = static_cast<std::size_t>(
        args.get_u64("prefetch-depth", 2));
    if (generated_count == 0 || generated_count > 128 || top_count <= 0) {
      fail(ErrorKind::Invalid, "invalid M2 generation or top-logit count");
    }
    if (enforce && store_mode == HostStoreMode::Synthetic) {
      fail(ErrorKind::Invalid, "M2 real inference forbids synthetic Model Store");
    }
    if (ring_depth == 0 || prefetch_depth == 0 ||
        prefetch_depth > ring_depth) {
      fail(ErrorKind::Invalid, "invalid M2 Ring/prefetch depth");
    }
    const std::string out_dir = args.get("out-dir");
    std::filesystem::create_directories(out_dir);
    const json::Value baseline = json::Value::parse_file(args.get("baseline"));
    const std::vector<llama_token> prompt_tokens =
        int_array_tokens(baseline, "prompt_token_ids");
    const std::vector<llama_token> expected_tokens =
        int_array_tokens(baseline, "generated_token_ids");
    if (prompt_tokens.empty() || expected_tokens.size() < generated_count) {
      fail(ErrorKind::Invalid, "M2 token baseline is incomplete");
    }

    json::Value manifest;
    const json::Value* manifest_ptr = nullptr;
    if (args.has("manifest")) {
      manifest = json::Value::parse_file(args.get("manifest"));
      manifest_ptr = &manifest;
      if (manifest.at("identity").at("model_sha256").as_string() !=
              args.get("model-sha256") ||
          manifest.at("identity").at("runtime_commit").as_string() !=
              ALSR_LLAMA_CPP_COMMIT ||
          manifest.at("identity").get_string("runtime_patch_sha256", "") !=
              ALSR_LLAMA_CPP_PATCH_SHA256) {
        fail(ErrorKind::Consistency, "M2 manifest identity mismatch");
      }
    }

    const auto program_start = Clock::now();
    llama_backend_init();
    LlamaBackendGuard backend_guard;
    llama_model_params model_params = llama_model_default_params();
    // Keep the tied output embedding resident (the ordinary direct path does
    // the same) while every block/norm/rope weight remains host-mapped and is
    // supplied through finite scheduler slots.
    llama_model_tensor_buft_override host_weight_overrides[] = {
        {"^(blk\\.|output_norm\\.|rope_freqs\\.).*",
         ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr}};
    model_params.n_gpu_layers = 999;
    model_params.tensor_buft_overrides =
        hooked ? host_weight_overrides : nullptr;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    ModelHandle model(
        llama_model_load_from_file(args.get("model").c_str(), model_params),
        &llama_model_free);
    if (!model) fail(ErrorKind::State, "M2 model load failed");
    const auto model_loaded = Clock::now();
    const auto groups = runtime_tensor_groups(model.get());
    if (groups.size() != 668) {
      fail(ErrorKind::Consistency, "M2 Runtime tensor instance count changed");
    }
    json::Value buffer_inventory =
        model_buffer_inventory(groups, llama_model_size(model.get()), enforce);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(n_ctx);
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_threads = static_cast<int32_t>(threads);
    context_params.n_threads_batch = context_params.n_threads;
    context_params.op_offload = true;
    context_params.no_perf = false;
    ContextHandle context(llama_init_from_model(model.get(), context_params),
                          &llama_free);
    if (!context) fail(ErrorKind::State, "M2 context init failed");
    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    SupplyController controller(
        groups, out_dir + "/supply_trace.jsonl", manifest_ptr, enforce,
        args.get("model"), store_mode, ring_depth, prefetch_depth);
    if (hooked) {
      ggml_backend_sched_set_alsr_callback(context->get_sched(),
                                           supply_callback, &controller);
    }

    const auto inference_start = Clock::now();
    if (hooked) controller.set_phase("prefill", prompt_tokens);
    llama_batch prefill = llama_batch_get_one(
        const_cast<llama_token*>(prompt_tokens.data()),
        static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(context.get(), prefill) != 0) {
      fail(ErrorKind::State, "M2 prefill failed");
    }

    json::Value generated = json::Value::make_array();
    json::Value steps = json::Value::make_array();
    std::vector<llama_token> generated_ids;
    std::string generated_text;
    llama_token first_generated = 0;
    std::string first_piece;
    for (u64 step = 0; step < generated_count; ++step) {
      const SampleResult sampled = greedy_top_logits(context.get(), vocab, top_count);
      if (hooked) controller.finish_phase();
      const std::string piece = token_piece(vocab, sampled.token);
      if (step == 0) {
        first_generated = sampled.token;
        first_piece = piece;
      }
      generated_ids.push_back(sampled.token);
      generated_text += piece;
      generated.push_back(sampled.token);
      json::Value step_json = json::Value::make_object();
      step_json.set("step", step);
      step_json.set("token_id", sampled.token);
      step_json.set("piece", piece);
      step_json.set("selected_logprob", sampled.selected_logprob);
      step_json.set("top_logits", sampled.top);
      steps.push_back(std::move(step_json));

      if (step + 1 < generated_count) {
        std::vector<llama_token> decode_input{sampled.token};
        if (hooked) controller.set_phase("decode", decode_input);
        llama_batch decode = llama_batch_get_one(decode_input.data(), 1);
        if (llama_decode(context.get(), decode) != 0) {
          fail(ErrorKind::State, "M2 decode failed");
        }
      }
    }
    const auto inference_end = Clock::now();
    if (controller.failed()) {
      fail(ErrorKind::Consistency, "M2 supply controller: " + controller.error());
    }

    bool token_match = true;
    for (std::size_t i = 0; i < generated_ids.size(); ++i) {
      token_match = token_match && generated_ids[i] == expected_tokens[i];
    }
    const bool text_match = generated_count == expected_tokens.size()
                                ? generated_text == baseline.at("response_text").as_string()
                                : true;
    const llama_perf_context_data perf = llama_perf_context(context.get());

    json::Value result = json::Value::make_object();
    result.set("schema", "alsr.m2.executor-result.v1");
    result.set("mission", "ALSR-M2-M5-REAL-EXECUTOR-01");
    result.set("mode", mode);
    result.set("runtime_tag", ALSR_LLAMA_CPP_TAG);
    result.set("runtime_commit", ALSR_LLAMA_CPP_COMMIT);
    result.set("runtime_patch_sha256", ALSR_LLAMA_CPP_PATCH_SHA256);
    result.set("model_path", args.get("model"));
    result.set("model_sha256", args.get("model-sha256"));
    result.set("model_runtime_bytes", llama_model_size(model.get()));
    result.set("runtime_tensor_instances", groups.size());
    result.set("n_gpu_layers", model_params.n_gpu_layers);
    result.set("nonresident_weight_storage_forced_host", hooked);
    result.set("cpu_repack_build_disabled", true);
    result.set("op_offload_min_batch",
               std::getenv("GGML_OP_OFFLOAD_MIN_BATCH") != nullptr
                   ? std::getenv("GGML_OP_OFFLOAD_MIN_BATCH")
                   : "UNSET");
    result.set("prompt_token_count", prompt_tokens.size());
    result.set("generated_token_count", generated_ids.size());
    result.set("generated_token_ids", std::move(generated));
    result.set("generated_text", generated_text);
    result.set("steps", std::move(steps));
    json::Value correctness = json::Value::make_object();
    correctness.set("reference", "M0 response.context generated suffix");
    correctness.set("token_ids_exact_match", token_match);
    correctness.set("text_exact_match_when_full_length", text_match);
    correctness.set("accepted", token_match && text_match);
    result.set("correctness", std::move(correctness));
    result.set("model_buffer_inventory", std::move(buffer_inventory));
    result.set("scheduler_buffers", scheduler_buffers(context.get()));
    result.set("supply_controller", controller.summary());
    json::Value timing = json::Value::make_object();
    timing.set("model_load_seconds",
               std::chrono::duration<double>(model_loaded - program_start).count());
    timing.set("inference_seconds",
               std::chrono::duration<double>(inference_end - inference_start).count());
    timing.set("prompt_eval_ms", perf.t_p_eval_ms);
    timing.set("decode_eval_ms", perf.t_eval_ms);
    timing.set("prompt_eval_tokens", perf.n_p_eval);
    timing.set("decode_eval_tokens", perf.n_eval);
    result.set("timing", std::move(timing));
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
      fail(ErrorKind::State, "M2 could not read process resource usage");
    }
    json::Value resources = json::Value::make_object();
#if defined(__APPLE__)
    resources.set("peak_resident_set_bytes",
                  static_cast<u64>(usage.ru_maxrss));
#else
    resources.set("peak_resident_set_bytes",
                  static_cast<u64>(usage.ru_maxrss) * 1024ULL);
#endif
    resources.set("minor_page_faults", static_cast<u64>(usage.ru_minflt));
    resources.set("major_page_faults", static_cast<u64>(usage.ru_majflt));
    resources.set("swap_operations", static_cast<u64>(usage.ru_nswap));
    resources.set("voluntary_context_switches",
                  static_cast<u64>(usage.ru_nvcsw));
    resources.set("involuntary_context_switches",
                  static_cast<u64>(usage.ru_nivcsw));
    resources.set("process_elapsed_seconds",
                  std::chrono::duration<double>(inference_end - program_start)
                      .count());
    result.set("process_resources", std::move(resources));
    write_json_file(out_dir + "/RESULT.json", result);

    if (hooked) {
      const std::string prompt = baseline.at("prompt").as_string();
      const json::Value trace = controller.captured_trace(
          groups, args.get("model"), args.get("model-sha256"), model.get(),
          sha256_hex(prompt.data(), prompt.size()), prompt.size(),
          first_generated, first_piece, n_ctx, threads,
          model_params.n_gpu_layers, hooked);
      write_json_file(out_dir + "/GRAPH_TRACE.json", trace);
      ggml_backend_sched_set_alsr_callback(context->get_sched(), nullptr,
                                           nullptr);
    }

    std::cout << "mode=" << mode << " tokens=" << generated_ids.size()
              << " token_match=" << (token_match ? "true" : "false")
              << " text_match=" << (text_match ? "true" : "false") << '\n';
    std::cout << "wrote " << out_dir << "/RESULT.json\n";

    return token_match && text_match ? 0 : 3;
  });
}
