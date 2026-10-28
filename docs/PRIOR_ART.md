# ALSR P0 Prior Art Review

```text
Mission: ALSR-P0-RUNTIME-INTEGRATION-AUDIT-01
Status: SOURCE_AUDIT_COMPLETE
Method: read-only source audit
Runtime pin: Ollama v0.32.5 / llama.cpp b10091
Evidence index: docs/P0_EVIDENCE_INDEX.json
```

## 1. Scope and claim discipline

This document records existing runtime mechanisms relevant to an M5 Real LLM Executor Adapter. No code was built or executed, no model was loaded, and no performance or correctness result was produced.

The following labels are used throughout the P0 artifacts:

- `SOURCE_CONFIRMED`: directly supported by a cited local or upstream source.
- `INFERRED_FROM_SOURCE`: architectural consequence of one or more cited source facts.
- `HYPOTHESIS_REQUIRES_M2`: plausible integration route that still requires a real-inference spike.
- `USER_DECISION_REQUIRED`: an OPEN item from the approved requirements baseline.
- `NOT_VERIFIED`: available artifact or statement that was not independently reproduced in P0.

## 2. Audited source baseline

The installed M5 environment reports macOS 26.5.2, arm64, 16 GiB physical memory, Ollama client 0.32.4 and Ollama server 0.32.5. The client/server version mismatch must be removed or explicitly frozen at M0. [P0-E014]

The source baseline for the process that actually serves models is therefore Ollama `v0.32.5`, commit `eec8e0b9458b8a01be0c216a9cc53eefde24ef50`. That release pins llama.cpp tag `b10091`, commit `b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45`. [P0-E015] [P0-E016]

P0 does not use llama.cpp `master` as an implementation baseline. Future upstream changes are prior art only until their exact commits are audited and deliberately adopted.

## 3. Ollama execution boundary

### 3.1 Existing boundary

`SOURCE_CONFIRMED`: Ollama serves all GGML models through an upstream `llama-server` subprocess. It launches that subprocess with the model path and waits for the process to load the model at startup. [P0-E017] [P0-E018]

```text
Ollama scheduler / API
        |
        | process arguments + HTTP
        v
llama-server subprocess
        |
        v
llama.cpp model loader + GGML backend scheduler
```

`INFERRED_FROM_SOURCE`: the public Ollama-to-`llama-server` process boundary does not expose a per-layer or per-weight readiness callback. ALSR cannot feed Ready Queue or Ring slots to the actual graph merely by wrapping Ollama HTTP requests.

### 3.2 Ollama GGUF compatibility layer

`SOURCE_CONFIRMED`: Ollama applies a temporary, in-process compatibility layer to published Ollama GGUFs. The layer mutates metadata, skips embedded tensors, and can perform transformed tensor reads during model loading. [P0-E019] [P0-E020] [P0-E021] [P0-E022]

For Gemma 4, the layer corrects tokenizer metadata and hides audio, vision, and projector tensors from the text loader. It disables mmap only if it must transform split MoE gate/up tensors. Dense Gemma 4 files do not enter that conditional solely because they are Gemma 4. [P0-E022]

`INFERRED_FROM_SOURCE`: directly linking unpatched upstream llama.cpp to an existing Ollama-format Gemma 4 blob is not an admissible default. The Real Executor must either:

1. use the exact Ollama-pinned llama.cpp plus the exact Ollama compatibility patch set; or
2. use a separately converted, upstream-compatible GGUF authorized by a later user Gate.

Option 2 is outside P0 because model conversion is prohibited.

## 4. llama.cpp model loading and mmap

### 4.1 Load-time placement

`SOURCE_CONFIRMED`: `llama_model_params` selects devices, buffer types, GPU layers, mmap, direct I/O, mlock, and no-allocation behavior. `llama_model_init_from_user` also permits a callback that initializes tensor data during model construction. [P0-E023]

`SOURCE_CONFIRMED`: model loading assigns layers to backend buffer types, creates buffers, marks weight buffers, and then calls `load_all_data` over the model tensors. mmap tensors become views into mapped file regions where supported. The Metal path can create a backend buffer from a host pointer. [P0-E024] [P0-E025]

`SOURCE_CONFIRMED`: when mmap is disabled, `load_all_data` can use four staging buffers and asynchronous device uploads. This mechanism completes model loading; it is not a token-time layer supply scheduler. [P0-E025]

`INFERRED_FROM_SOURCE`: `llama_model_init_from_user` and the Ollama load hooks are useful compatibility/load-time boundaries, but they do not by themselves satisfy FR-05 through FR-09. The ALSR buffer must be made ready immediately before its actual graph consumer and retained until the backend has finished using it.

### 4.2 Unified-memory scope

`SOURCE_CONFIRMED`: the Metal backend advertises asynchronous operation, host-pointer mapping, and events. It does not advertise the generic host-buffer capability. It accepts its shared, private, and mapped buffer types. [P0-E029] [P0-E030]

`INFERRED_FROM_SOURCE`: on M5, a CPU-side buffer and a Metal buffer cannot be counted as independent RAM and VRAM capacities. Whether shared, private, or mapped buffers reduce or duplicate physical residency is a later measurement question. P0 does not claim a memory saving from any of those buffer types.

## 5. Actual graph and tensor execution order

`SOURCE_CONFIRMED`: Gemma 4 tensor creation and execution are encoded in the llama.cpp model implementation. The graph loops over actual layers and constructs attention, FFN or MoE operations, per-layer projections, final normalization, and the output head. [P0-E026]

`INFERRED_FROM_SOURCE`: the execution manifest must be derived from the actual built graph or an instrumented, version-pinned mapping to that graph. GGUF file order or a static tensor-name ranking is insufficient because:

- optional tensors change the graph;
- prefill and decode may have different graph shapes and batching;
- dense and MoE paths differ;
- backend optimization can split and reorder execution units.

## 6. GGML backend scheduler as prior art

### 6.1 Existing cross-backend copy and reuse

`SOURCE_CONFIRMED`: the GGML backend scheduler assigns operations using supported operations and weight locations, splits the graph by backend, creates input copies for incompatible buffers, and copies split inputs before compute. Events protect copy-buffer reuse. [P0-E027] [P0-E028]

`SOURCE_CONFIRMED`: for host-resident weights, `op_offload` can select a higher-priority backend. A new split is started when a different incompatible weight is encountered so that memory used by the previous offloaded weight can be reused. [P0-E028]

This is the closest existing mechanism to ALSR's intended consumer-side weight staging. It must be measured as prior art before claiming that ALSR adds a novel or faster path.

### 6.2 Metal decode caveat

`SOURCE_CONFIRMED`: the Metal backend offers operation offload for matrix multiplication only when the operation batch size reaches `op_offload_min_batch_size`. The default is 32 and can be changed with `GGML_OP_OFFLOAD_MIN_BATCH`. [P0-E029] [P0-E030]

`HYPOTHESIS_REQUIRES_M2`: setting the threshold to 1 may expose the existing split-copy path during single-token decode. P0 does not claim that it is correct, memory-bounded, or performant for Gemma 4 12B/31B. It is an M2 spike candidate and a required comparison point.

### 6.3 Why the public eval callback is not the ALSR hook

`SOURCE_CONFIRMED`: `ggml_backend_sched_set_eval_callback` asks whether a caller wants to observe a node, computes the selected graph view, synchronizes the backend, and then returns the result to the callback. [P0-E027] [P0-E028]

`INFERRED_FROM_SOURCE`: this callback is an observation/cancellation hook, not a pre-consumer data-ready contract. Using it for every weight would force synchronization and still would not expose split-input ownership or safe retirement. It is not the selected ALSR integration boundary.

## 7. Metal async copy and lifetime primitives

`SOURCE_CONFIRMED`: the Metal tensor-set path wraps source data in a shared Metal buffer, queues a blit, commits without waiting, and retains the command buffer for later synchronization. Graph compute is also submitted asynchronously. [P0-E031]

`INFERRED_FROM_SOURCE`: required primitives exist for an M5 Adapter, but the lifetime contract must remain event-backed:

```text
PREPARING -> READY -> COPY_QUEUED -> IN_USE -> RETIRED
```

No Ring slot may return to `PREPARING` until the copy/compute event protecting that data has completed.

## 8. Correctness comparison prior art

The in-process llama API exposes tokenization, logits, sampling, and decode state to a direct runner. Ollama's subprocess/HTTP surface is appropriate as a user-visible control path, but it is not sufficient for per-step ALSR trace correlation. [P0-E023] [P0-E017] [P0-E018]

`INFERRED_FROM_SOURCE`: the most discriminating M2 comparison is a direct, exact-pin baseline that captures token IDs and per-step logits from the same model digest, plus a normal Ollama result as an external control. Exact acceptance tolerance remains `USER_DECISION_REQUIRED` under O-04.

## 9. Prior-art conclusion

```text
Existing complete solution found: NO
Existing reusable runtime substrate found: YES
Existing dynamic weight-copy/reuse prior art found: YES
Public no-patch ALSR supply hook found: NO
In-process narrow runtime integration appears feasible: YES, unexecuted
```

The correct P0 direction is therefore neither a standalone fake Executor nor a wholesale rewrite of llama.cpp. It is a version-pinned, in-process Real Executor based on Ollama-compatible llama.cpp, with the smallest auditable extension needed to connect ALSR preparation and lifetime events to the real GGML graph.
