# M5 Real Executor Integration Feasibility

> Historical scope note (amended 2026-08-04 JST): this document is the Gate P0 feasibility snapshot. Its `PLAUSIBLE_NOT_RUN` and authorization statements describe the state at P0. The first M2 GREEN was withdrawn because its callback did not request or supply tensor bytes. M2 was later repaired with explicit range I/O and a finite Ring, then requalified on 128 tokens × 5 runs. Current status is recorded in `docs/ALSR_REQ_DESIGN.md` Rev.3.6 and `results/m2-gemma4-12b-real-supply-repair-20260804T002432Z/M2_REPORT.md`.

## 1. Formal P0 verdict

```text
Mission: ALSR-P0-RUNTIME-INTEGRATION-AUDIT-01
P0 technical audit: GREEN
P0 final closure: GREEN — O-01_TO_O-04_USER_APPROVED_2026-08-03_JST
Implementation feasibility: FEASIBLE_WITH_NARROW_PINNED_RUNTIME_EXTENSION
Unmodified Ollama HTTP integration: NOT_FEASIBLE
Current local code as Real Executor: NOT_FEASIBLE
Code/build/test/model operations performed: NONE
M0/M1/M2/M3 authorization: NOT_GRANTED_BY_THIS_REPORT
```

The three Gate P0 technical conditions are satisfied:

1. The replacement/addition boundary is identified in source: the real graph and GGML split-input copy/lifetime path. [P0-E026–P0-E031]
2. An M5 Adapter is source-feasible using the exact Ollama-compatible llama.cpp in process; Metal exposes required copy/mapping/event primitives. [P0-E019–P0-E031]
3. The delta hypotheses and exit conditions are fixed in `docs/DELTA_HYPOTHESIS.md`.

The initial source-audit result placed final closure on hold because the approved baseline assigns O-01 through O-04 to P0 and forbids the AI from silently deciding them. On 2026-08-03 JST, the user approved all four proposals on the explicit premise that the project must change `Real inference = False` to `Real inference = True`. The decisions are recorded in `docs/ALSR_REQ_DESIGN.md` Rev.3.1. [P0-E001]

## 2. What is and is not feasible

### Feasible direction

- preserve ordinary Ollama as the control lane;
- create a separate ALSR Real Executor process linked in-process with Ollama v0.32.5's exact llama.cpp b10091 and compatibility layer;
- derive demand from the actual graph;
- adapt the existing GGML dynamic split/offload path with the smallest auditable supply/lifetime extension;
- connect the local fixed Scheduler, finite Ring, Model Store, and Evidence primitives after their contracts are corrected;
- prove the path on Gemma 4 12B at M2, then use the same structure for Gemma 4 31B at M3.

### Not feasible as claimed

- treating the current CRC simulator as an Executor;
- injecting per-layer data through Ollama HTTP or command-line arguments;
- declaring tensor-name order to be actual execution order;
- calling load-time staging/prefetch a token-time ALSR scheduler;
- treating a mapped address range as a proven memory saving;
- treating a one-token or one-model result as completion of the scale research.

## 3. Selected integration point

The selected source boundary is the GGML backend scheduler's real split-input path:

```text
actual graph node references host weight
        |
        v
backend assignment / op_offload
        |
        v
split identifies incompatible weight input
        |
        +---- ALSR pre-supply / lease boundary (extension required)
        v
async/sync tensor copy to Metal destination
        |
        v
graph compute
        |
        v
event completion
        |
        +---- ALSR retirement boundary (extension required)
```

This boundary is preferred because GGML already performs backend assignment, split allocation, copy fallback, event protection, and weight-memory reuse. [P0-E027–P0-E030]

The public eval callback is explicitly rejected as the primary integration point because it is an observation/cancel interface and synchronizes around observed nodes. [P0-E027] [P0-E028]

## 4. Patch and maintenance boundary

The implementation target must be recorded as a source tuple, not merely a product version:

```text
Ollama tag: v0.32.5
Ollama commit: eec8e0b9458b8a01be0c216a9cc53eefde24ef50
llama.cpp tag: b10091
llama.cpp commit: b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45
Ollama compatibility patch SHA-256:
fd2528d78cf25a00171a904e6ae06ea5c86122f04584fa2fddb611ec56e64a4a
Ollama compatibility implementation SHA-256:
9c4e18e451c22ccf8b4d95ca2f8baaf8525ba3a06ae3223ea02a6b014f86e642
```

Any future bump requires:

- source diff of all touched integration points;
- compatibility patch identity update;
- graph/manifest determinism revalidation;
- correctness regression;
- hidden-full-load check;
- renewed P0-style evidence for changed boundaries.

## 5. M5 feasibility by requirement

| Requirement | P0 feasibility | Condition |
|---|---|---|
| real prompt-to-token path | `FEASIBLE` | in-process pinned llama.cpp Executor |
| Ollama Gemma 4 blob compatibility | `FEASIBLE_WITH_PIN` | exact Ollama compatibility layer; actual blob load remains unexecuted |
| actual execution order | `FEASIBLE` | instrument real prefill/decode graph |
| finite prepared-weight supply | `FEASIBLE_WITH_EXTENSION` | pre-copy lease hook and real backing mode |
| safe Ring reuse | `FEASIBLE_WITH_EXTENSION` | Metal/GGML event-backed retirement |
| fixed Scheduler | `FEASIBLE_AFTER_ADAPTATION` | real demand/timing inputs; no Adaptive certification |
| no hidden full load | `TESTABLE` | allocation/map/I/O trace plus model-loader instrumentation |
| correctness comparison | `TESTABLE` | exact-pin direct baseline; O-04 acceptance pending |
| 12B M2 | `PLAUSIBLE_NOT_RUN` | M0/M1 then M2 authorization |
| Gemma 4 31B M3 | `PLAUSIBLE_NOT_RUN` | same M2 structure, approved paging/I/O policy, measured budget |
| models beyond 31B | `NOT_DECIDABLE_AT_P0` | M4/S0 evidence; 31B is not the research endpoint |
| RTX portability | `ARCHITECTURALLY_PRESERVED` | no RTX claim until X0/R Gates |

## 6. M0/M1/M2 entry conditions

P0 recommends the following order after final P0 user closure:

### Before M0

- resolve Ollama client/server version mismatch;
- freeze exact 12B model digest and file set;
- approve O-01 through O-04;
- define immutable Evidence Pack storage for real runs.

### Before M1

- select the exact pinned compatibility view used to enumerate tensors;
- compare local GGUF byte ranges with the runtime's tensor view;
- specify graph-signature identity for prefill and decode.

### Before M2 implementation

- approve a bounded runtime patch surface;
- prohibit a second full model load;
- fix the initial Scheduler to non-Adaptive;
- define failure behavior for missing/late/corrupt PreparedWeight;
- define correctness acceptance and run limits.

## 7. Approved user decisions

The following P0 proposals were approved by the user on 2026-08-03 JST. They do not relax the requirement for real inference.

### O-01 — OS paging/swap

`APPROVED`: allow ordinary OS mmap/paging as a required control baseline and as observable backing behavior, but do not count it alone as proof that ALSR scheduling supplied the weights. Record page faults where available, compressed memory, swap, storage I/O, and physical pressure separately.

Reason: M3's model exceeds physical memory, but an uncontrolled OS paging success cannot prove ALSR's Queue/Ring mechanism.

### O-02 — Explicit Storage I/O

`APPROVED`: stage the implementation:

1. Ordinary mmap/paging remains an allowed comparison baseline, but does not by itself prove ALSR supply.
2. M2 12B must use explicit range I/O whose prepared bytes become the backend copy source through a finite Queue/Ring lifecycle; fallback to the ordinary Runtime source must be zero.
3. M3 Gemma 4 31B reuses that M2 structure and compares it with ordinary mmap/paging; the architecture is not deferred to M3.
4. Compare explicit ALSR I/O with ordinary mmap/paging; do not presume either is faster.

### O-03 — M5 acceptance length, repeats, and maximum time

`APPROVED CONTRACT`:

| Gate | New tokens | Repeats | Maximum wall time | Scope |
|---|---:|---:|---:|---|
| M0 ordinary 12B | 128 | 5 cold + 5 warm | 30 min/run | baseline and variance |
| M2 ALSR 12B | 128 | 5 | 30 min/run | correctness and integration |
| M3 Gemma 4 31B | 128 | 3 | 60 min/run | memory-pressure feasibility |

A timeout is a recorded failure/boundary, not permission to shorten the required output after seeing results. M4 must add sustained and multi-scale measurements; M3 does not complete the research.

### O-04 — Output correctness

`APPROVED`: use a dual criterion against the exact-pin direct baseline:

1. same model digest, tokenizer, prompt, context, seed, and deterministic greedy sampling;
2. exact generated token-ID sequence for the acceptance length;
3. per-step top-logit capture for diagnosis and a numeric tolerance fixed from baseline reproducibility before ALSR comparison;
4. any mismatch excludes the Run from performance claims;
5. perplexity is a later secondary scale/regression metric, not the only M2 criterion.

The numeric logit tolerance remains unset until the baseline's own reproducibility is measured.

## 8. Immediate next authorized action

P0 is closed. The next and only authorized Mission is `ALSR-M0-M5-BASELINE-01`: obtain the ordinary Gemma 4 12B M5 baseline under the approved O-03/O-04 contract. M1/M2 implementation remains separately gated according to the approved sequence.

## 9. Final statement

The M5-first plan is technically credible. The current code does not yet implement it, but the exact runtime contains the graph, dynamic weight-copy, buffer reuse, Metal async copy, and event primitives needed for a bounded integration. The hard part is not PCIe on M5; it is proving that the real Executor consumed the ALSR-prepared weight at the correct graph point without a hidden full-load path, while preserving output correctness under unified-memory pressure.
