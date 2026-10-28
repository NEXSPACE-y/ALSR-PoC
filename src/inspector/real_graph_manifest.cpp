#include "alsr/real_graph_manifest.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/ggml_types.hpp"

namespace alsr {
namespace {

struct StorageChunk {
  u64 id = 0;
  std::string kind;
  Bytes source_offset = 0;
  Bytes bytes = 0;
  std::size_t tensor_index = 0;
  std::string tensor_name;
  Bytes offset_in_tensor = 0;
  u32 crc32c = 0;
  std::string sha256;
};

struct RuntimeTensorMapping {
  std::string runtime_name;
  u64 runtime_instance = 0;
  std::vector<std::string> aliases;
  const GgufTensorInfo* gguf = nullptr;
};

std::string runtime_key(const std::string& name, u64 instance) {
  return name + "\n" + std::to_string(instance);
}

std::vector<std::string> string_array(const json::Value& value,
                                      std::string_view field) {
  const json::Value* array = value.find(field);
  if (array == nullptr) return {};
  std::vector<std::string> out;
  for (const auto& item : array->as_array()) out.push_back(item.as_string());
  return out;
}

const GgufTensorInfo* find_source_tensor(
    const GgufFile& gguf, const std::string& runtime_name,
    const std::vector<std::string>& aliases) {
  if (const auto* exact = gguf.find_tensor(runtime_name)) return exact;
  for (const auto& alias : aliases) {
    if (const auto* match = gguf.find_tensor(alias)) return match;
  }
  return nullptr;
}

std::vector<StorageChunk> make_storage_chunks(
    const GgufFile& gguf, const RealGraphManifestOptions& options) {
  if (options.target_chunk_bytes == 0) {
    fail(ErrorKind::Invalid, "real graph target_chunk_bytes must be > 0");
  }
  std::vector<const GgufTensorInfo*> sorted;
  sorted.reserve(gguf.tensors.size());
  for (const auto& tensor : gguf.tensors) sorted.push_back(&tensor);
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
    if (a->abs_offset != b->abs_offset) return a->abs_offset < b->abs_offset;
    return a->file_index < b->file_index;
  });

  std::vector<StorageChunk> chunks;
  Bytes cursor = gguf.data_start;
  u64 next_id = 0;
  const auto add_padding = [&](Bytes begin, Bytes end,
                               std::vector<StorageChunk>& dst) {
    if (end <= begin) return;
    StorageChunk chunk;
    chunk.id = next_id++;
    chunk.kind = "padding";
    chunk.source_offset = begin;
    chunk.bytes = end - begin;
    dst.push_back(std::move(chunk));
  };

  for (const auto* tensor : sorted) {
    if (tensor->abs_offset < cursor) {
      fail(ErrorKind::Consistency,
           "GGUF tensor overlap before real graph chunk construction");
    }
    add_padding(cursor, tensor->abs_offset, chunks);

    const Bytes granule = std::lcm(
        ggml_block_bytes(tensor->type_id),
        gguf.alignment == 0 ? Bytes{1} : gguf.alignment);
    Bytes regular = align_down(options.target_chunk_bytes, granule);
    if (regular == 0) regular = granule;
    Bytes offset = 0;
    while (offset < tensor->nbytes) {
      const Bytes remaining = tensor->nbytes - offset;
      const Bytes take = remaining > regular ? regular : remaining;
      StorageChunk chunk;
      chunk.id = next_id++;
      chunk.kind = "tensor";
      chunk.source_offset = tensor->abs_offset + offset;
      chunk.bytes = take;
      chunk.tensor_index = tensor->file_index;
      chunk.tensor_name = tensor->name;
      chunk.offset_in_tensor = offset;
      chunks.push_back(std::move(chunk));
      offset += take;
    }
    cursor = tensor->abs_offset + tensor->nbytes;
  }
  add_padding(cursor, gguf.file_size, chunks);
  return chunks;
}

void checksum_chunks(const std::string& path, Bytes data_start,
                     std::vector<StorageChunk>& chunks, Bytes buffer_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(ErrorKind::Io, "cannot open GGUF for M1 checksums: " + path);
  in.seekg(static_cast<std::streamoff>(data_start), std::ios::beg);
  if (!in) fail(ErrorKind::Io, "cannot seek to GGUF data_start");

  const std::size_t size = static_cast<std::size_t>(
      std::max<Bytes>(buffer_bytes, 64 * KiB));
  std::vector<char> buffer(size);
  Bytes expected = data_start;
  for (auto& chunk : chunks) {
    if (chunk.source_offset != expected) {
      fail(ErrorKind::Consistency, "M1 chunks are not file-order contiguous");
    }
    Crc32c crc;
    Sha256 sha;
    Bytes remaining = chunk.bytes;
    while (remaining > 0) {
      const std::size_t want = static_cast<std::size_t>(
          std::min<Bytes>(remaining, buffer.size()));
      in.read(buffer.data(), static_cast<std::streamsize>(want));
      if (in.gcount() != static_cast<std::streamsize>(want)) {
        fail(ErrorKind::Io, "short read while hashing M1 storage chunk");
      }
      crc.update(buffer.data(), want);
      sha.update(buffer.data(), want);
      remaining -= want;
    }
    chunk.crc32c = crc.value();
    chunk.sha256 = sha.hex();
    expected += chunk.bytes;
  }
}

json::Value chunk_to_json(const StorageChunk& chunk, bool checksums) {
  json::Value value = json::Value::make_object();
  value.set("chunk_id", chunk.id);
  value.set("kind", chunk.kind);
  value.set("source_offset", chunk.source_offset);
  value.set("bytes", chunk.bytes);
  if (chunk.kind == "tensor") {
    value.set("tensor_index", chunk.tensor_index);
    value.set("tensor_name", chunk.tensor_name);
    value.set("offset_in_tensor", chunk.offset_in_tensor);
  }
  value.set("checksum_computed", checksums);
  if (checksums) {
    value.set("crc32c", chunk.crc32c);
    value.set("sha256", chunk.sha256);
  }
  return value;
}

std::string manifest_structure_digest(const json::Value& manifest) {
  const json::Value& identity = manifest.at("identity");
  const json::Value& storage = manifest.at("storage_plan");
  const json::Value& demand = manifest.at("demand_plans");
  json::Value stable = json::Value::make_object();
  stable.set("model_sha256", identity.get_string("model_sha256", ""));
  stable.set("runtime_commit", identity.get_string("runtime_commit", ""));
  stable.set("runtime_patch_sha256",
             identity.get_string("runtime_patch_sha256", ""));
  stable.set("target_chunk_bytes",
             storage.at("target_chunk_bytes").as_int());
  stable.set("chunks", storage.at("chunks"));
  stable.set("demand_plans", demand);
  const std::string bytes = stable.dump();
  return sha256_hex(bytes.data(), bytes.size());
}

}  // namespace

json::Value build_real_graph_manifest(
    const GgufFile& gguf, const json::Value& runtime_trace,
    const RealGraphManifestOptions& options) {
  if (runtime_trace.get_string("schema", "") !=
      "alsr.llama.graph-trace.v2") {
    fail(ErrorKind::Invalid,
         "M1 requires alsr.llama.graph-trace.v2 with Runtime instance identity");
  }
  if (!gguf.layout.ok()) {
    fail(ErrorKind::Consistency,
         "GGUF layout is not fully explained; real graph manifest refused");
  }

  std::unordered_map<std::string, RuntimeTensorMapping> runtime;
  for (const auto& item : runtime_trace.at("runtime_tensors").as_array()) {
    RuntimeTensorMapping mapping;
    mapping.runtime_name = item.at("name").as_string();
    mapping.runtime_instance = static_cast<u64>(
        item.at("runtime_instance").as_int());
    mapping.aliases = string_array(item, "aliases");
    mapping.gguf = find_source_tensor(gguf, mapping.runtime_name,
                                      mapping.aliases);
    const std::string key = runtime_key(mapping.runtime_name,
                                        mapping.runtime_instance);
    const auto existing = runtime.find(key);
    if (existing != runtime.end()) {
      fail(ErrorKind::Consistency,
           "duplicate Runtime tensor identity in M1 graph trace");
    }
    if (mapping.gguf != nullptr &&
        static_cast<Bytes>(item.get_int("bytes", 0)) != mapping.gguf->nbytes) {
      fail(ErrorKind::Consistency,
           "Runtime tensor bytes differ from mapped GGUF source bytes");
    }
    runtime.emplace(key, std::move(mapping));
  }

  std::vector<StorageChunk> chunks = make_storage_chunks(gguf, options);
  if (options.compute_checksums) {
    checksum_chunks(gguf.path, gguf.data_start, chunks,
                    options.read_buffer_bytes);
  }

  std::unordered_map<std::size_t, std::vector<u64>> chunks_by_tensor;
  Bytes covered = 0;
  Bytes tensor_bytes = 0;
  Bytes padding_bytes = 0;
  for (const auto& chunk : chunks) {
    covered += chunk.bytes;
    if (chunk.kind == "tensor") {
      tensor_bytes += chunk.bytes;
      chunks_by_tensor[chunk.tensor_index].push_back(chunk.id);
    } else {
      padding_bytes += chunk.bytes;
    }
  }

  json::Value root = json::Value::make_object();
  root.set("schema", "alsr.real-graph-manifest.v1");
  root.set("mission", "ALSR-M1-REAL-GRAPH-MANIFEST-01");

  json::Value identity = json::Value::make_object();
  identity.set("model_path", gguf.path);
  identity.set("model_sha256",
               runtime_trace.at("model").get_string("sha256", ""));
  identity.set("architecture", gguf.architecture);
  identity.set("runtime", runtime_trace.at("runtime").get_string("name", ""));
  identity.set("runtime_tag", runtime_trace.at("runtime").get_string("tag", ""));
  identity.set("runtime_commit",
               runtime_trace.at("runtime").get_string("commit", ""));
  identity.set("runtime_patch_sha256",
               runtime_trace.at("runtime").get_string("patch_sha256", ""));
  identity.set("trace_authority", "REAL_RUNTIME_GRAPH_CALLBACK");
  identity.set("heuristic_tensor_name_order_authoritative", false);
  root.set("identity", std::move(identity));

  json::Value mappings = json::Value::make_array();
  std::size_t mapped_runtime_tensors = 0;
  for (const auto& item : runtime_trace.at("runtime_tensors").as_array()) {
    const std::string name = item.at("name").as_string();
    const u64 instance = static_cast<u64>(
        item.at("runtime_instance").as_int());
    const auto& mapping = runtime.at(runtime_key(name, instance));
    json::Value out = json::Value::make_object();
    out.set("runtime_name", name);
    out.set("runtime_instance", instance);
    out.set("runtime_bytes", item.get_int("bytes", 0));
    json::Value aliases = json::Value::make_array();
    for (const auto& alias : mapping.aliases) aliases.push_back(alias);
    out.set("aliases", std::move(aliases));
    out.set("mapped", mapping.gguf != nullptr);
    if (mapping.gguf != nullptr) {
      ++mapped_runtime_tensors;
      out.set("gguf_tensor_index", mapping.gguf->file_index);
      out.set("gguf_name", mapping.gguf->name);
      out.set("source_offset", mapping.gguf->abs_offset);
      out.set("source_bytes", mapping.gguf->nbytes);
      out.set("runtime_source_bytes_match",
              static_cast<Bytes>(item.get_int("bytes", 0)) ==
                  mapping.gguf->nbytes);
    }
    mappings.push_back(std::move(out));
  }
  root.set("runtime_tensor_mappings", std::move(mappings));

  json::Value demand_plans = json::Value::make_array();
  std::set<std::string> used_runtime_keys;
  std::set<std::size_t> used_gguf_indices;
  std::size_t unmapped_weight_uses = 0;
  for (const auto& phase : runtime_trace.at("phases").as_array()) {
    json::Value plan = json::Value::make_object();
    plan.set("phase", phase.at("name").as_string());
    plan.set("input_token_count", phase.at("input_token_count").as_int());
    plan.set("graph_signature_sha256",
             phase.at("graph_signature_sha256").as_string());
    plan.set("graph_node_count", phase.at("nodes").as_array().size());

    struct UseSummary {
      std::string runtime_name;
      u64 runtime_instance = 0;
      const GgufTensorInfo* tensor = nullptr;
      u64 first = 0;
      u64 last = 0;
      u64 count = 0;
    };
    std::map<std::string, UseSummary> uses;
    for (const auto& node : phase.at("nodes").as_array()) {
      const u64 order = static_cast<u64>(node.at("order").as_int());
      for (const auto& weight : node.at("weight_inputs").as_array()) {
        const std::string runtime_name = weight.at("name").as_string();
        const u64 runtime_instance = static_cast<u64>(
            weight.at("runtime_instance").as_int());
        const std::string key = runtime_key(runtime_name, runtime_instance);
        used_runtime_keys.insert(key);
        const auto found = runtime.find(key);
        if (found == runtime.end() || found->second.gguf == nullptr) {
          ++unmapped_weight_uses;
          continue;
        }
        const auto* tensor = found->second.gguf;
        used_gguf_indices.insert(tensor->file_index);
        auto [it, inserted] = uses.emplace(key, UseSummary{});
        if (inserted) {
          it->second.runtime_name = runtime_name;
          it->second.runtime_instance = runtime_instance;
          it->second.tensor = tensor;
          it->second.first = order;
        }
        it->second.last = order;
        ++it->second.count;
      }
    }

    std::vector<std::pair<std::string, UseSummary>> ordered(uses.begin(),
                                                            uses.end());
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a,
                                                        const auto& b) {
      if (a.second.first != b.second.first) {
        return a.second.first < b.second.first;
      }
      return a.first < b.first;
    });
    json::Value sequence = json::Value::make_array();
    u64 demand_order = 0;
    for (const auto& [key, use] : ordered) {
      (void)key;
      json::Value entry = json::Value::make_object();
      entry.set("demand_order", demand_order++);
      entry.set("runtime_name", use.runtime_name);
      entry.set("runtime_instance", use.runtime_instance);
      entry.set("gguf_name", use.tensor->name);
      entry.set("gguf_tensor_index", use.tensor->file_index);
      entry.set("first_graph_node_order", use.first);
      entry.set("last_graph_node_order", use.last);
      entry.set("graph_use_count", use.count);
      entry.set("source_offset", use.tensor->abs_offset);
      entry.set("source_bytes", use.tensor->nbytes);
      json::Value ids = json::Value::make_array();
      for (const u64 id : chunks_by_tensor.at(use.tensor->file_index)) {
        ids.push_back(id);
      }
      entry.set("chunk_ids_in_tensor_order", std::move(ids));
      sequence.push_back(std::move(entry));
    }
    plan.set("first_use_demand_sequence", std::move(sequence));
    demand_plans.push_back(std::move(plan));
  }
  root.set("demand_plans", std::move(demand_plans));

  json::Value storage = json::Value::make_object();
  storage.set("target_chunk_bytes", options.target_chunk_bytes);
  storage.set("chunk_count", chunks.size());
  storage.set("covered_bytes", covered);
  storage.set("tensor_bytes", tensor_bytes);
  storage.set("padding_bytes", padding_bytes);
  storage.set("checksum_algorithm_primary", "SHA-256");
  storage.set("checksum_algorithm_fast", "CRC32C");
  json::Value chunk_values = json::Value::make_array();
  for (const auto& chunk : chunks) {
    chunk_values.push_back(chunk_to_json(chunk, options.compute_checksums));
  }
  storage.set("chunks", std::move(chunk_values));
  root.set("storage_plan", std::move(storage));

  json::Value integrity = json::Value::make_object();
  integrity.set("gguf_layout_ok", gguf.layout.ok());
  integrity.set("data_bytes_expected", gguf.file_size - gguf.data_start);
  integrity.set("data_bytes_covered", covered);
  integrity.set("tensor_bytes_expected", gguf.layout.tensor_bytes_total);
  integrity.set("tensor_bytes_covered", tensor_bytes);
  integrity.set("padding_bytes_expected", gguf.layout.padding_bytes);
  integrity.set("padding_bytes_covered", padding_bytes);
  integrity.set("runtime_tensor_instance_count",
                runtime_trace.at("runtime_tensors").as_array().size());
  integrity.set("runtime_tensor_identity_count", runtime.size());
  integrity.set("mapped_runtime_tensor_instance_count",
                mapped_runtime_tensors);
  integrity.set("used_runtime_tensor_identity_count",
                used_runtime_keys.size());
  integrity.set("used_gguf_tensor_count", used_gguf_indices.size());
  integrity.set("unmapped_weight_use_count", unmapped_weight_uses);
  integrity.set("chunk_checksums_computed", options.compute_checksums);
  const bool structural_complete =
      covered == gguf.file_size - gguf.data_start &&
      tensor_bytes == gguf.layout.tensor_bytes_total &&
      padding_bytes == gguf.layout.padding_bytes;
  const bool mapping_complete = unmapped_weight_uses == 0;
  integrity.set("structural_complete", structural_complete);
  integrity.set("mapping_complete", mapping_complete);
  integrity.set("complete", structural_complete && mapping_complete &&
                                options.compute_checksums);
  root.set("integrity", std::move(integrity));

  json::Value contract = json::Value::make_object();
  contract.set("consumer", "M2_ALSR_REAL_EXECUTOR");
  contract.set("supply_order", "first_use_demand_sequence");
  contract.set("supply_unit", "storage_plan.chunks tensor entries only");
  contract.set("padding_supply_required", false);
  contract.set("identity_key",
               "model_sha256 + runtime_commit + runtime_patch_sha256 + phase + graph_signature_sha256");
  contract.set("runtime_weight_identity",
               "runtime_name + runtime_instance; duplicate names are distinct destinations");
  contract.set("retirement_requirement",
               "M2 must attach completion-event-backed lifetime to every supplied tensor range");
  contract.set("hidden_full_load_forbidden", true);
  root.set("m2_input_contract", std::move(contract));

  root.set("manifest_structure_sha256", manifest_structure_digest(root));
  verify_real_graph_manifest(gguf, root);
  return root;
}

void verify_real_graph_manifest(const GgufFile& gguf,
                                const json::Value& manifest) {
  if (manifest.get_string("schema", "") !=
      "alsr.real-graph-manifest.v1") {
    fail(ErrorKind::Consistency, "unexpected real graph manifest schema");
  }
  const auto& storage = manifest.at("storage_plan");
  const auto& chunks = storage.at("chunks").as_array();
  if (chunks.empty()) fail(ErrorKind::Consistency, "M1 storage plan is empty");
  Bytes cursor = gguf.data_start;
  std::vector<Bytes> tensor_covered(gguf.tensors.size(), 0);
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    const auto& chunk = chunks[i];
    if (static_cast<std::size_t>(chunk.at("chunk_id").as_int()) != i) {
      fail(ErrorKind::Consistency, "M1 chunk IDs are not contiguous");
    }
    const Bytes offset = static_cast<Bytes>(chunk.at("source_offset").as_int());
    const Bytes bytes = static_cast<Bytes>(chunk.at("bytes").as_int());
    if (offset != cursor || bytes == 0) {
      fail(ErrorKind::Consistency, "M1 storage range gap, overlap, or empty chunk");
    }
    if (chunk.at("kind").as_string() == "tensor") {
      const auto index = static_cast<std::size_t>(
          chunk.at("tensor_index").as_int());
      if (index >= gguf.tensors.size()) {
        fail(ErrorKind::Consistency, "M1 chunk references unknown tensor");
      }
      const auto& tensor = gguf.tensors[index];
      const Bytes in_tensor = static_cast<Bytes>(
          chunk.at("offset_in_tensor").as_int());
      if (in_tensor != tensor_covered[index] ||
          offset != tensor.abs_offset + in_tensor ||
          in_tensor + bytes > tensor.nbytes) {
        fail(ErrorKind::Consistency, "M1 tensor fragment coverage mismatch");
      }
      tensor_covered[index] += bytes;
    } else if (chunk.at("kind").as_string() != "padding") {
      fail(ErrorKind::Consistency, "M1 chunk has unknown kind");
    }
    cursor += bytes;
  }
  if (cursor != gguf.file_size) {
    fail(ErrorKind::Consistency, "M1 storage plan does not cover GGUF data tail");
  }
  for (std::size_t i = 0; i < gguf.tensors.size(); ++i) {
    if (tensor_covered[i] != gguf.tensors[i].nbytes) {
      fail(ErrorKind::Consistency, "M1 tensor is not covered exactly once");
    }
  }
  const auto& integrity = manifest.at("integrity");
  if (!integrity.at("structural_complete").as_bool()) {
    fail(ErrorKind::Consistency, "M1 structural coverage is not complete");
  }
  if (!integrity.at("mapping_complete").as_bool() ||
      integrity.at("unmapped_weight_use_count").as_int() != 0) {
    fail(ErrorKind::Consistency, "M1 graph weight mapping is not complete");
  }
  if (integrity.at("chunk_checksums_computed").as_bool() &&
      !integrity.at("complete").as_bool()) {
    fail(ErrorKind::Consistency, "M1 checksum-complete summary is inconsistent");
  }
}

}  // namespace alsr
