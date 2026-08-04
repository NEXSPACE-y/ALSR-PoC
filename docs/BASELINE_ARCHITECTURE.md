# ALSR P0 Baseline Architecture

> Historical scope note (2026-08-03 JST): this is the pre-implementation P0 architecture snapshot. Gate M2 has since completed GREEN. Current status and the next M3 boundary are authoritative in `docs/ALSR_REQ_DESIGN.md` Rev.3.4.

```text
Mission: ALSR-P0-RUNTIME-INTEGRATION-AUDIT-01
Baseline type: source-observed, not execution-validated
Requirements baseline: docs/ALSR_REQ_DESIGN.md Rev.3
Evidence index: docs/P0_EVIDENCE_INDEX.json
```

## 1. Current repository architecture

The current repository builds a static `alsr_core` and four command-line tools: `alsr-inspect`, `alsr-plan`, `alsr-sim`, and `alsr-bw`. It links no llama.cpp, Ollama, Metal, or CUDA inference runtime. The only compiled transfer backend is the simulator; `backends/cuda` contains a README only. [P0-E002]

```text
Ollama blob resolver
        |
        v
GGUF parser -> heuristic manifest -> byte-range chunk plan
                                      |
                                      v
HostStore -> finite Ring -> Scheduler -> SimulatorBackend
                                      |
                                      v
                            virtual time + CRC32C
```

The Ollama integration resolves a model layer to a local blob path and stops there. It does not invoke Ollama inference or llama.cpp. [P0-E006]

The transfer interface reports virtual preparation/copy/compute durations and exposes `consume(pointer, bytes)`. The simulator's `consume` calculates CRC32C; the main simulator retires a chunk using virtual compute time. No logits or tokens are produced. [P0-E003] [P0-E004] [P0-E005]

Therefore the current repository has no Real LLM Executor Adapter and no M5 real-inference path.

## 2. Current manifest and planner boundary

The local inspector assigns execution order through a static `intra_block_rank` table and a static dependency map keyed by tensor names. The planner constructs byte-range chunks in file-offset order and later sorts chunk references by that inferred `exec_order`. [P0-E007] [P0-E008]

This logic is useful as a structural hypothesis but does not meet Rev.3 FR-03. It has not been derived from the real llama.cpp graph. [P0-E001]

```text
Current: GGUF tensor name -> static role/order table -> chunk order
Required: real graph use -> version-pinned tensor/use mapping -> chunk demand
```

## 3. Component disposition

| Component | P0 disposition | Reason / required condition | Evidence |
|---|---|---|---|
| GGUF byte-layout parser | `REUSE_AFTER_M1_VERIFICATION` | offset/type/shape parsing is useful; compare against the exact pinned runtime loader and Ollama compatibility view | P0-E032 |
| checksum/SHA primitives | `REUSE` | general integrity primitive; runtime Evidence semantics still need extension | P0-E033 |
| Ollama blob resolver | `ADAPT` | valid read-only resolution helper, but digest/file-set capture and compat identity are incomplete | P0-E006 |
| heuristic Execution Manifest | `REPLACE_AS_AUTHORITY` | may remain diagnostic, but cannot be authoritative for graph order | P0-E007, P0-E026 |
| byte-range Chunk Planner | `ADAPT` | range integrity is reusable; demand order and lifetime must come from the real graph | P0-E008 |
| HostStore mmap/pread modes | `ADAPT` | useful backing primitives; must expose tensor/range readiness, faults, and I/O provenance | P0-E009 |
| Ring state machine | `ADAPT` | finite slots and illegal-transition detection are useful; states and event-backed retirement must match Rev.3 | P0-E010 |
| fixed/adaptive Scheduler | `ADAPT_FIXED_FIRST` | inputs/decisions are reusable; real consumer demand and event timings are absent; Adaptive remains later Gate | P0-E011 |
| simulator TransferBackend | `SIMULATOR_ONLY` | valid only for virtual-time/component testing | P0-E003, P0-E004 |
| simulator compute/consume | `DO_NOT_REUSE_AS_EXECUTOR` | virtual time plus CRC is not inference | P0-E005 |
| Telemetry/Evidence writer | `ADAPT` | useful primitives; real model/token/graph/memory/I/O evidence fields are missing | P0-E012 |
| CUDA backend | `NOT_IMPLEMENTED` | directory documentation only | P0-E002 |
| M5 Metal backend | `NOT_IMPLEMENTED` | no local source or build target | P0-E002 |
| Real LLM Executor | `NOT_IMPLEMENTED` | no tokenizer, context, KV cache, decode, sampler, logits, or token output | P0-E002 |

## 4. P0 integration baseline

The selected baseline is two distinct lanes. They must not be conflated.

### 4.1 Control lane

```text
normal Ollama v0.32.5
        -> llama-server subprocess
        -> Ollama-compatible llama.cpp b10091
        -> normal mmap/offload policy
```

Purpose:

- confirm the exact model is supported;
- preserve a user-visible ordinary-runtime baseline;
- measure ordinary mmap/paging and output behavior at later Gates;
- prevent ALSR from taking credit for existing runtime behavior.

### 4.2 ALSR experiment lane

```text
ALSR Real Executor process
  |
  +-- exact Ollama v0.32.5 compatibility layer
  +-- exact llama.cpp b10091 in process
  +-- actual graph/tensor-use collector
  +-- ALSR Model Store + fixed Scheduler + finite Ring
  +-- narrow GGML split-input supply/lifetime hook
  +-- Metal backend and events
  +-- token/logit output + ALSR trace + Evidence Pack
```

Ollama remains the control lane. It is not the per-weight data plane. The experiment lane must be in-process because the required graph and split-input lifetime state exists inside llama.cpp/GGML. [P0-E017] [P0-E018] [P0-E027] [P0-E028]

## 5. Required runtime contracts

### 5.1 `GraphDemand`

Derived from the actual graph, not a name sort.

```text
graph_id
phase: prefill | decode
node/split identity
weight tensor identity
source byte range
consumer backend
first-use order
last-use/event boundary
optional/reused flags
```

### 5.2 `PreparedWeight`

```text
chunk_id
tensor_id
source range
ready pointer/buffer
bytes
checksum state
ownership token
ready timestamp
```

### 5.3 `SupplyLease`

```text
prepared weight
destination tensor/buffer
copy event
compute/event dependency
retirement event
error/cancel state
```

`INFERRED_FROM_SOURCE`: the minimal llama.cpp extension should expose the transition between a split identifying an incompatible host weight and the scheduler copying that input, plus a completion signal after the corresponding event. The public eval callback is too late and synchronizing. [P0-E027] [P0-E028]

The exact callback ABI is a design hypothesis, not yet frozen. It must be proven by M2 before becoming the cross-platform contract.

## 6. Data lifecycle

```text
1. Resolve exact model file set and digest.
2. Apply the exact Ollama compatibility view.
3. Build the real llama.cpp graph for prefill/decode.
4. Emit version-pinned GraphDemand records.
5. Convert demand into integrity-preserving byte ranges.
6. Prepare upcoming ranges through the selected Model Store mode.
7. Mark a finite Ring slot READY.
8. At the GGML split boundary, bind/copy the READY weight to the real consumer.
9. Record Metal readiness/copy/compute events.
10. Retire the slot only after the protecting event completes.
11. Capture logits/tokens and compare with the baseline.
```

Any path in which llama.cpp separately loads a complete second copy of model weights invalidates the ALSR run under FR-09.

## 7. M5 memory model

M5 unified memory is a single physical-memory budget. P0 fixes accounting categories but not their values:

- OS and unrelated processes;
- mapped model address range;
- physically resident model pages when observable;
- active Metal weight buffers;
- KV cache;
- activations/graph work buffers;
- ALSR Ring/staging buffers;
- compressed memory;
- swap and storage I/O.

`HYPOTHESIS_REQUIRES_M2`: the existing GGML split mechanism may already reuse destination weight memory. ALSR should instrument and, where possible, reuse that mechanism instead of allocating a parallel Metal ring. [P0-E028]

`USER_DECISION_REQUIRED`: whether OS paging is permitted as an ALSR backing mechanism and whether M3 requires explicit range I/O remain O-01 and O-02. No memory-success claim is made in P0.

## 8. Gate sequence preserved

The architecture does not replace the approved Gate sequence:

1. M0: ordinary 12B baseline.
2. M1: real-graph manifest and chunk integrity.
3. M2: 12B Real Executor integration with ALSR data and trace.
4. M3: Gemma 4 31B memory-pressure real inference.
5. M4: multiple model sizes and the M5 scale envelope.
6. X0: freeze the shared contract before RTX.

P0 does not authorize implementation or any of these Runs.
