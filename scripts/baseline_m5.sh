#!/usr/bin/env bash
# Phase 0 — M5 上の Ollama 基準値を「同一スクリプト」で再取得する。
#
# 設計書 §11 Phase 0 / NFR-02:
#   - 乱数 seed、モデル digest、Ollama 版、context、生成長を保存する
#   - warm-up 後に最低 5 回実施する
#   - 平均だけでなく中央値と分位点を報告する
#
# 出力: results/baseline_m5.json
#
# 注意:
#   これは「ユニファイドメモリ上で 7.38 GB 級モデルが動いた基準値」であって、
#   PCIe ストリーミングで同じ速度が出る根拠ではない（設計書 §2.3）。
set -euo pipefail

MODEL="${MODEL:-gemma4:12b}"
NUM_CTX="${NUM_CTX:-4096}"
NUM_PREDICT="${NUM_PREDICT:-256}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-1}"
SEED="${SEED:-20260802}"
OUT="${OUT:-results/baseline_m5.json}"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

PROMPT='日本の電力供給における原子力・火力・再生可能エネルギーの3つの電源について、技術面と経済面の両側面から比較・整理してください。'

mkdir -p "$(dirname "$OUT")"
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

request() {
  cat <<JSON
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": $(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
  "stream": false,
  "options": {
    "num_ctx": ${NUM_CTX},
    "num_predict": ${NUM_PREDICT},
    "temperature": 0,
    "seed": ${SEED}
  }
}
JSON
}

echo "model      : ${MODEL}"
echo "num_ctx    : ${NUM_CTX}"
echo "num_predict: ${NUM_PREDICT}"
echo "seed       : ${SEED}  (temperature=0)"
echo "repeats    : ${REPEATS} (warm-up ${WARMUP} 回は集計から除外)"
echo

TOTAL=$((WARMUP + REPEATS))
for i in $(seq 1 "$TOTAL"); do
  printf 'run %d/%d ... ' "$i" "$TOTAL"
  request | curl -sS -X POST "${HOST}/api/chat" -H 'Content-Type: application/json' -d @- \
    > "${TMPDIR_RUN}/run_${i}.json"
  python3 - "${TMPDIR_RUN}/run_${i}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ev = d.get("eval_count", 0); dur = d.get("eval_duration", 0)
print(f'{ev} tok / {dur/1e9:.3f} s = {ev/(dur/1e9):.2f} token/s' if dur else 'NO TIMING')
PY
done

# 環境証拠と digest
OLLAMA_VERSION="$(ollama --version 2>/dev/null | head -1 || echo unavailable)"
OLLAMA_PS="$(ollama ps 2>/dev/null || echo unavailable)"
MODEL_DIGEST="$(ollama list 2>/dev/null | awk -v m="$MODEL" '$1==m {print $2}' || true)"

python3 - "$TMPDIR_RUN" "$WARMUP" "$REPEATS" "$OUT" \
         "$MODEL" "$NUM_CTX" "$NUM_PREDICT" "$SEED" \
         "$OLLAMA_VERSION" "$MODEL_DIGEST" "$OLLAMA_PS" <<'PY'
import json, os, statistics, subprocess, sys

tmp, warmup, repeats, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
model, ctx, npred, seed = sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
ollama_version, digest, ollama_ps = sys.argv[9], sys.argv[10], sys.argv[11]

samples = []
for i in range(1, warmup + repeats + 1):
    d = json.load(open(os.path.join(tmp, f"run_{i}.json"), encoding="utf-8"))
    if not d.get("eval_duration"):
        continue
    s = {
        "index": i,
        "warmup": i <= warmup,
        "eval_count": d["eval_count"],
        "eval_duration_ns": d["eval_duration"],
        "prompt_eval_count": d.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": d.get("prompt_eval_duration", 0),
        "load_duration_ns": d.get("load_duration", 0),
        "total_duration_ns": d.get("total_duration", 0),
        "done_reason": d.get("done_reason", ""),
        "generation_token_per_s": d["eval_count"] / (d["eval_duration"] / 1e9),
    }
    if s["prompt_eval_duration_ns"]:
        s["prompt_token_per_s"] = s["prompt_eval_count"] / (s["prompt_eval_duration_ns"] / 1e9)
    samples.append(s)

kept = [s["generation_token_per_s"] for s in samples if not s["warmup"]]
kept.sort()

def q(v, p):
    if not v:
        return 0.0
    i = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
    return v[i]

summary = {
    "note": [
        "M5 (ユニファイドメモリ) 上の Ollama 通常推論の基準値。",
        "PCIe ストリーミングで同じ速度が出る根拠ではない (設計書 §2.3)。",
        "warm-up 分は集計から除外している。",
    ],
    "model": model,
    "model_digest": digest,
    "ollama_version": ollama_version,
    "ollama_ps_after_run": ollama_ps,
    "num_ctx": ctx,
    "num_predict": npred,
    "seed": seed,
    "temperature": 0,
    "sample_count": len(kept),
    "generation_token_per_s": {
        "median": statistics.median(kept) if kept else 0.0,
        "mean": statistics.fmean(kept) if kept else 0.0,
        "min": kept[0] if kept else 0.0,
        "max": kept[-1] if kept else 0.0,
        "p05": q(kept, 0.05),
        "p95": q(kept, 0.95),
    },
    "samples": samples,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")

print()
print(f"生成速度 中央値 : {summary['generation_token_per_s']['median']:.2f} token/s")
print(f"           範囲 : {summary['generation_token_per_s']['min']:.2f} .. "
      f"{summary['generation_token_per_s']['max']:.2f} token/s")
print(f"wrote {out}")
PY
