// alsr-llama-graph-trace — capture actual prefill/decode graph weight use.
//
// This target is built only against the exact M1-pinned llama.cpp commit. The
// callback records scheduler graph order at ask=true and returns false so it
// does not introduce per-node synchronization or copy tensor data back.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/json.hpp"
#include "cli_common.hpp"

#include "ggml.h"
#include "llama.h"
#include "llama-model.h"

#ifndef ALSR_LLAMA_CPP_TAG
#define ALSR_LLAMA_CPP_TAG "UNPINNED"
#endif
#ifndef ALSR_LLAMA_CPP_COMMIT
#define ALSR_LLAMA_CPP_COMMIT "UNPINNED"
#endif

using namespace alsr;

namespace {

struct PhaseCapture {
  std::string name;
  std::vector<llama_token> input_tokens;
  json::Value nodes = json::Value::make_array();
};

struct WeightIdentity {
  std::string name;
  u64 runtime_instance = 0;

  bool operator<(const WeightIdentity& other) const noexcept {
    if (name != other.name) return name < other.name;
    return runtime_instance < other.runtime_instance;
  }
};

struct TraceState {
  std::unordered_map<const ggml_tensor*, WeightIdentity> weights;
  PhaseCapture* phase = nullptr;
};

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

bool graph_callback(ggml_tensor* tensor, bool ask, void* user_data) {
  auto* state = static_cast<TraceState*>(user_data);
  if (!ask || state->phase == nullptr) return false;

  std::set<WeightIdentity> used;
  for (const auto* source : tensor->src) {
    std::set<const ggml_tensor*> visited;
    resolve_weight_sources(source, state->weights, used, visited);
  }

  json::Value node = json::Value::make_object();
  node.set("order", state->phase->nodes.as_array().size());
  node.set("name", std::string(ggml_get_name(tensor)));
  node.set("op", std::string(ggml_op_name(tensor->op)));
  node.set("type", std::string(ggml_type_name(tensor->type)));
  json::Value shape = json::Value::make_array();
  const int dimensions = ggml_n_dims(tensor);
  for (int i = 0; i < dimensions; ++i) shape.push_back(tensor->ne[i]);
  node.set("shape", std::move(shape));
  json::Value weight_inputs = json::Value::make_array();
  for (const auto& identity : used) {
    json::Value weight = json::Value::make_object();
    weight.set("name", identity.name);
    weight.set("runtime_instance", identity.runtime_instance);
    weight_inputs.push_back(std::move(weight));
  }
  node.set("weight_inputs", std::move(weight_inputs));
  state->phase->nodes.push_back(std::move(node));
  return false;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text) {
  int32_t count = llama_tokenize(vocab, text.data(),
                                 static_cast<int32_t>(text.size()), nullptr, 0,
                                 true, true);
  if (count == INT32_MIN) fail(ErrorKind::Invalid, "prompt token count overflow");
  if (count >= 0) {
    fail(ErrorKind::Consistency,
         "llama_tokenize size query unexpectedly returned non-negative");
  }
  std::vector<llama_token> tokens(static_cast<std::size_t>(-count));
  count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                         tokens.data(), static_cast<int32_t>(tokens.size()),
                         true, true);
  if (count < 0) fail(ErrorKind::Consistency, "llama_tokenize failed");
  tokens.resize(static_cast<std::size_t>(count));
  return tokens;
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
  if (size < 0) return {};
  return std::string(buffer.data(), static_cast<std::size_t>(size));
}

json::Value phase_to_json(PhaseCapture& phase) {
  json::Value value = json::Value::make_object();
  value.set("name", phase.name);
  value.set("input_token_count", phase.input_tokens.size());
  json::Value tokens = json::Value::make_array();
  for (const auto token : phase.input_tokens) tokens.push_back(token);
  value.set("input_token_ids", std::move(tokens));

  json::Value stable = json::Value::make_object();
  stable.set("name", phase.name);
  stable.set("input_token_count", phase.input_tokens.size());
  stable.set("nodes", phase.nodes);
  const std::string serialized = stable.dump();
  value.set("graph_signature_sha256",
            sha256_hex(serialized.data(), serialized.size()));
  value.set("nodes", std::move(phase.nodes));
  return value;
}

void usage() {
  std::cout << R"(alsr-llama-graph-trace — real llama.cpp graph trace

Usage:
  alsr-llama-graph-trace --model <GGUF> --prompt-file <text> --out <trace.json>
                         [--model-sha256 <hex>] [--n-ctx <n>]
                         [--n-gpu-layers <n>] [--threads <n>]

Runs one real prefill and one real single-token decode through the pinned
llama.cpp Runtime. This is graph instrumentation for M1, not ALSR inference.
)";
}

}  // namespace

int main(int argc, char** argv) {
  return cli::run_main([&]() -> int {
    cli::Args args(argc, argv);
    if (args.flag("help") || argc == 1) {
      usage();
      return 0;
    }
    args.reject_unknown({"model", "prompt-file", "out", "model-sha256",
                         "n-ctx", "n-gpu-layers", "threads", "help"});
    if (!args.has("model") || !args.has("prompt-file") || !args.has("out")) {
      fail(ErrorKind::Invalid,
           "--model, --prompt-file, and --out are required");
    }
    if (std::string(ALSR_LLAMA_CPP_COMMIT) == "UNPINNED") {
      fail(ErrorKind::Consistency, "M1 llama.cpp build is not commit-pinned");
    }

    std::ifstream prompt_in(args.get("prompt-file"));
    if (!prompt_in) fail(ErrorKind::Io, "cannot read prompt file");
    const std::string prompt((std::istreambuf_iterator<char>(prompt_in)),
                             std::istreambuf_iterator<char>());

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = static_cast<int32_t>(
        args.get_u64("n-gpu-layers", 999));
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    llama_model* model = llama_model_load_from_file(args.get("model").c_str(),
                                                    model_params);
    if (model == nullptr) fail(ErrorKind::State, "llama.cpp model load failed");

    struct RuntimeTensorGroup {
      const ggml_tensor* tensor = nullptr;
      std::vector<std::string> names;
      std::size_t first_runtime_index = 0;
      std::string canonical;
      u64 runtime_instance = 0;
    };
    std::vector<RuntimeTensorGroup> runtime_groups;
    std::unordered_map<const ggml_tensor*, std::size_t> group_by_pointer;
    std::size_t runtime_index = 0;
    for (const auto& [name, tensor] : llama_internal_get_tensor_map(model)) {
      const auto found = group_by_pointer.find(tensor);
      if (found == group_by_pointer.end()) {
        group_by_pointer.emplace(tensor, runtime_groups.size());
        runtime_groups.push_back({tensor, {name}, runtime_index, {}, 0});
      } else {
        runtime_groups[found->second].names.push_back(name);
      }
      ++runtime_index;
    }
    for (auto& group : runtime_groups) {
      std::sort(group.names.begin(), group.names.end());
      const std::string ggml_name = ggml_get_name(group.tensor);
      const auto exact = std::find(group.names.begin(), group.names.end(),
                                   ggml_name);
      group.canonical = exact != group.names.end() ? *exact : group.names.front();
    }
    std::stable_sort(runtime_groups.begin(), runtime_groups.end(),
                     [](const auto& a, const auto& b) {
      if (a.canonical != b.canonical) return a.canonical < b.canonical;
      return a.first_runtime_index < b.first_runtime_index;
    });
    std::map<std::string, u64> next_instance;
    TraceState state;
    for (auto& group : runtime_groups) {
      group.runtime_instance = next_instance[group.canonical]++;
      state.weights.emplace(
          group.tensor,
          WeightIdentity{group.canonical, group.runtime_instance});
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(args.get_u64("n-ctx", 4096));
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_threads = static_cast<int32_t>(args.get_u64("threads", 8));
    context_params.n_threads_batch = context_params.n_threads;
    context_params.cb_eval = graph_callback;
    context_params.cb_eval_user_data = &state;
    context_params.no_perf = false;
    llama_context* context = llama_init_from_model(model, context_params);
    if (context == nullptr) fail(ErrorKind::State, "llama.cpp context init failed");

    const llama_vocab* vocab = llama_model_get_vocab(model);
    PhaseCapture prefill;
    prefill.name = "prefill";
    prefill.input_tokens = tokenize(vocab, prompt);
    if (prefill.input_tokens.empty() || prefill.input_tokens.size() > 512) {
      fail(ErrorKind::Invalid, "prefill token count must be in 1..512");
    }
    state.phase = &prefill;
    llama_batch prefill_batch = llama_batch_get_one(
        prefill.input_tokens.data(),
        static_cast<int32_t>(prefill.input_tokens.size()));
    if (llama_decode(context, prefill_batch) != 0) {
      fail(ErrorKind::State, "llama.cpp prefill decode failed");
    }

    llama_sampler* sampler = llama_sampler_init_greedy();
    const llama_token next = llama_sampler_sample(sampler, context, -1);
    llama_sampler_free(sampler);

    PhaseCapture decode;
    decode.name = "decode";
    decode.input_tokens = {next};
    state.phase = &decode;
    llama_batch decode_batch = llama_batch_get_one(decode.input_tokens.data(), 1);
    if (llama_decode(context, decode_batch) != 0) {
      fail(ErrorKind::State, "llama.cpp single-token decode failed");
    }
    state.phase = nullptr;

    json::Value root = json::Value::make_object();
    root.set("schema", "alsr.llama.graph-trace.v2");
    root.set("mission", "ALSR-M1-REAL-GRAPH-MANIFEST-01");
    json::Value runtime = json::Value::make_object();
    runtime.set("name", "llama.cpp");
    runtime.set("tag", ALSR_LLAMA_CPP_TAG);
    runtime.set("commit", ALSR_LLAMA_CPP_COMMIT);
    runtime.set("callback", "ggml_backend_sched_eval_callback ask=true");
    runtime.set("callback_requests_tensor_data", false);
    root.set("runtime", std::move(runtime));

    json::Value model_json = json::Value::make_object();
    model_json.set("path", args.get("model"));
    model_json.set("sha256", args.get("model-sha256", ""));
    model_json.set("runtime_size_bytes", llama_model_size(model));
    model_json.set("runtime_parameter_count", llama_model_n_params(model));
    model_json.set("runtime_tensor_count", runtime_groups.size());
    root.set("model", std::move(model_json));

    json::Value request = json::Value::make_object();
    request.set("prompt_sha256", sha256_hex(prompt.data(), prompt.size()));
    request.set("prompt_bytes", prompt.size());
    request.set("n_ctx", context_params.n_ctx);
    request.set("n_batch", context_params.n_batch);
    request.set("n_ubatch", context_params.n_ubatch);
    request.set("n_gpu_layers", model_params.n_gpu_layers);
    request.set("threads", context_params.n_threads);
    request.set("sampled_decode_token_id", next);
    request.set("sampled_decode_token_piece", token_piece(vocab, next));
    root.set("request", std::move(request));

    json::Value runtime_tensors = json::Value::make_array();
    for (const auto& group : runtime_groups) {
      const auto* tensor = group.tensor;
      const auto& names = group.names;
      const std::string& canonical = group.canonical;
      json::Value item = json::Value::make_object();
      item.set("name", canonical);
      item.set("runtime_instance", group.runtime_instance);
      json::Value aliases = json::Value::make_array();
      for (const auto& name : names) {
        if (name != canonical) aliases.push_back(name);
      }
      item.set("aliases", std::move(aliases));
      item.set("bytes", ggml_nbytes(tensor));
      item.set("type", std::string(ggml_type_name(tensor->type)));
      json::Value shape = json::Value::make_array();
      const int dimensions = ggml_n_dims(tensor);
      for (int i = 0; i < dimensions; ++i) shape.push_back(tensor->ne[i]);
      item.set("shape", std::move(shape));
      runtime_tensors.push_back(std::move(item));
    }
    root.set("runtime_tensors", std::move(runtime_tensors));

    json::Value phases = json::Value::make_array();
    phases.push_back(phase_to_json(prefill));
    phases.push_back(phase_to_json(decode));
    root.set("phases", std::move(phases));

    std::ofstream out(args.get("out"), std::ios::trunc);
    if (!out) fail(ErrorKind::Io, "cannot write graph trace");
    out << root.dump(2) << '\n';

    std::cout << "prefill tokens: " << prefill.input_tokens.size()
              << ", graph nodes: "
              << root.at("phases").as_array().at(0).at("nodes").as_array().size()
              << "\n";
    std::cout << "decode token: " << next << ", graph nodes: "
              << root.at("phases").as_array().at(1).at("nodes").as_array().size()
              << "\n";
    std::cout << "wrote " << args.get("out") << "\n";

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
  });
}
