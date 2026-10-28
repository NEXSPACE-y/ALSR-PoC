# ALSR P0 Delta Hypothesis

```text
Mission: ALSR-P0-RUNTIME-INTEGRATION-AUDIT-01
Status: HYPOTHESES_FIXED_FOR_LATER_TEST
No hypothesis in this file is a measured result.
```

## 1. Source-confirmed deltas

| ID | Current state | Required state | Delta | Evidence |
|---|---|---|---|---|
| D-P0-01 | Four local tools; no inference runtime linkage | real M5 prompt-to-token Executor | add an in-process, exact-pin llama.cpp Executor lane | P0-E002 |
| D-P0-02 | Ollama support resolves a blob path only | preserve Ollama-format compatibility while executing | link the v0.32.5 compatibility layer or obtain later conversion approval | P0-E006, P0-E019–P0-E022 |
| D-P0-03 | execution order is a static tensor-name heuristic | order derived from actual prefill/decode graph | replace heuristic authority with graph-derived demand | P0-E007, P0-E026 |
| D-P0-04 | Chunk Plan has file ranges and inferred order | Chunk Plan maps graph demand to integrity-safe ranges and lifetimes | add graph/tensor/range/use mapping | P0-E008 |
| D-P0-05 | TransferBackend accepts pointer+byte count and returns CRC/timing | actual tensor identity, destination, readiness, event and retirement | replace the real-backend contract; keep simulator contract separate | P0-E003–P0-E005 |
| D-P0-06 | Ring retirement is driven by virtual compute end | retirement protected by real backend events | add event-backed ownership/lifetime | P0-E010, P0-E028–P0-E031 |
| D-P0-07 | Scheduler consumes simulator observations | Scheduler is driven by real graph demand, I/O, queue, and Metal events | connect fixed Scheduler first; defer Adaptive | P0-E011 |
| D-P0-08 | Evidence contains simulator metrics | real digest, prompt, tokens, graph/chunk trace, memory/I/O, correctness | extend Evidence schema and immutable Run capture | P0-E012, P0-E001 |
| D-P0-09 | no authoritative ordinary-runtime control | same exact model under ordinary Ollama/llama.cpp | create control lane with pinned conditions | P0-E014–P0-E018 |

## 2. Integration hypotheses

### H-P0-01 — Existing GGML op offload is the lowest-risk substrate

The GGML scheduler already detects host weights, assigns supported operations to a higher-priority backend, copies split inputs, reuses prior offloaded-weight memory, and protects reuse with events. [P0-E027–P0-E030]

Hypothesis: ALSR can extend this path rather than implementing an independent Metal graph executor.

Required M2 evidence:

- graph split and weight-copy trace for prefill and single-token decode;
- actual peak destination allocation;
- proof that weight memory is reused rather than retained for the full model;
- token/logit correctness against the exact-pin baseline;
- proof that source data came through the ALSR lease, not an untraced second path.

Falsifier: the path retains destination copies for the whole model, cannot execute decode weights on Metal, or cannot expose a safe supply boundary without broad invasive changes.

### H-P0-02 — A narrow scheduler hook is required

The public eval callback is an observation path after compute segmentation and includes synchronization. The internal split-input copy path has the needed weight identity and event ordering. [P0-E027] [P0-E028]

Hypothesis: a small, version-pinned extension at the split-input boundary can expose:

```text
before_weight_copy(weight, destination, split) -> SupplyLease
after_weight_use(lease, completion_event)
```

The names and ABI above are illustrative, not approved interface design.

Falsifier: the hook cannot preserve GGML allocation/lifetime invariants, requires architecture-specific rewrites throughout graph construction, or makes upstream rebasing unbounded.

### H-P0-03 — Manifest authority can come from actual graph instrumentation

Gemma 4's real graph construction contains the actual optional tensor branches and per-layer operations. [P0-E026]

Hypothesis: an instrumented dry graph build or first baseline execution can emit a deterministic tensor-use map for a pinned model/runtime/phase tuple.

Required identity:

```text
model digest + Ollama compat patch digest + llama.cpp commit
+ context parameters + phase + graph signature
```

Falsifier: graph optimization makes the mapping unstable under identical inputs, or the mapping cannot be related deterministically to GGUF byte ranges.

### H-P0-04 — Local structural components are salvageable

The local GGUF reader, checksum code, HostStore, finite Ring, fixed Scheduler, and Evidence writer contain useful implementation primitives. [P0-E009–P0-E012] [P0-E032] [P0-E033]

Hypothesis: these components can be adapted without preserving the simulator's false consumer abstraction.

Falsifier: comparison against the pinned runtime finds parsing/range disagreement, or the required event/lifetime changes make preservation riskier than replacement.

### H-P0-05 — The existing 31B Ollama blob is likely on the dense Gemma 4 path

A pre-existing local manifest lists 60 text blocks and dense `ffn_gate/up/down` tensors with no expert tensor names. The artifact was not regenerated in P0 and its `model_sha256` field is empty. [P0-E013]

`NOT_VERIFIED`: this is a planning clue only. M0/M1 must obtain the real digest and compare the runtime compatibility view. It is not proof that the blob will load or generate tokens.

Falsifier: the pinned runtime detects transformed MoE tensors, disables mmap, rejects the blob, or produces a tensor set inconsistent with the local parser.

### H-P0-06 — M5 can validate the structure without proving RTX behavior

Metal provides asynchronous copies, mapped host pointers, and events, while M5 has unified memory rather than a PCIe H2D boundary. [P0-E029–P0-E031]

Hypothesis: M5 can validate real graph demand, finite preparation, ordering, event-backed lifetime, correctness, and memory-pressure behavior. It cannot validate CUDA copy engines or PCIe throughput.

Falsifier: the unified-memory backend bypasses the prepared data path so completely that ALSR ownership and lifetime cannot be distinguished from ordinary paging.

## 3. M2 vertical-slice hypothesis

The first implementation slice, after user authorization, should be deliberately narrow but must remain real:

1. exact Ollama v0.32.5 / llama.cpp b10091 source pin;
2. Gemma 4 12B control result under ordinary runtime;
3. direct in-process llama.cpp prompt-to-token execution;
4. graph-derived demand for at least one complete prefill and decode path;
5. fixed Scheduler only;
6. a finite number of real PreparedWeight leases;
7. event-backed retirement;
8. token/logit comparison and immutable trace;
9. detection of any hidden full-model load.

This is not a reduced product requirement. It is Gate M2's integration proof before M3 applies the same structure to Gemma 4 31B and M4 expands the scale envelope.

## 4. Competing explanations that must be tested

| ID | Competing explanation | Required comparison |
|---|---|---|
| C-P0-01 | ordinary mmap/paging already provides equal or better behavior | ordinary Ollama and direct llama.cpp mmap baselines |
| C-P0-02 | GGML dynamic op offload already implements the useful part of ALSR | op offload enabled/disabled with identical correctness and memory evidence |
| C-P0-03 | apparent memory saving is only address mapping or compression/swap | physical-memory, compressed-memory, swap and I/O observations |
| C-P0-04 | Ring buffering duplicates unified-memory pages and worsens pressure | shared/private/mapped buffer accounting |
| C-P0-05 | heuristic chunk order happens to match one model but not the real graph | manifest-to-graph deterministic comparison |
| C-P0-06 | output stays plausible while weights are stale or corrupted | per-step logits/token comparison plus injected integrity failure at a later fault Gate |

## 5. Exit and stop conditions

Stop the current integration direction and return to a user Gate if any of the following is source-confirmed or later reproduced:

- the existing Ollama Gemma 4 blob cannot be consumed by the exact pinned compatibility runtime;
- a safe pre-consumer supply and post-consumer retirement boundary cannot be created;
- the runtime loads an untraced full second weight copy;
- the actual graph cannot be mapped deterministically to model byte ranges;
- output correctness cannot be defined or maintained;
- the M5 budget cannot contain Runtime, KV, activations, Ring, active weights, and required OS headroom under the approved paging/I/O policy;
- the existing GGML path is equivalent or better within measurement error;
- the patch surface becomes broad enough that a maintained fork is not credible.

An exit records an applicability boundary; it is not converted into a success claim.

## 6. Hypothesis status

```text
P0 source consistency: COMPLETE
M5 Adapter feasibility hypothesis: SUPPORTED_BY_SOURCE
Adapter implementation: NOT_STARTED
Real inference: NOT_EXECUTED
Correctness: NOT_MEASURED
Performance: NOT_MEASURED
Memory saving: NOT_MEASURED
```
