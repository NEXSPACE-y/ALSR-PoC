#!/usr/bin/env bash
# 設計書 §5.1「64、128、256、512、1024MiB の固定Chunk試験」+ 可変Chunk試験、
# および NFR-02「ベンチマークは warm-up 後に最低 5 回実施する」。
#
# 重要:
#   仮想時間は決定論的なので、同一設定を 5 回回しても仮想時間の結果は同一に
#   なる。反復が意味を持つのは **壁時計**（= 計測オーバーヘッド, NFR-05）だけ。
#   そのことを隠さず、両方を出す。
#
# VRAM に収まらない構成が出た場合、それは失敗ではなく物理的な測定結果である。
# 「収まらない」ことを記録して次へ進む（§12.4 RED を隠さないのと同じ理由）。
set -uo pipefail

MODEL_REF="${MODEL_REF:-gemma4:31b}"
RESULTS="${RESULTS:-results}"
OUT="${OUT:-$RESULTS/sweep_chunk_sizes.json}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-1}"
RESIDENT_BYTES="${RESIDENT_BYTES:-10000000000}"
RING_DEPTH="${RING_DEPTH:-4}"
PREFETCH="${PREFETCH:-3}"
H2D_GBPS="${H2D_GBPS:-25.0}"
DEVICE_BYTES="${DEVICE_BYTES:-12884901888}"

SIZES=(64 128 256 512 1024)

mkdir -p "$RESULTS"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "model      : $MODEL_REF"
echo "chunk 段   : ${SIZES[*]} MiB + variable"
echo "反復       : warm-up $WARMUP + 本番 $REPEATS"
echo "常駐       : $RESIDENT_BYTES bytes"
echo "ring depth : $RING_DEPTH (prefetch $PREFETCH)"
echo

echo '{"cases":[' > "$TMP/acc.json"
FIRST=1

emit() {  # $1 = JSON オブジェクト
  if [[ $FIRST -eq 0 ]]; then echo "," >> "$TMP/acc.json"; fi
  FIRST=0
  printf '%s' "$1" >> "$TMP/acc.json"
}

run_case() {  # $1 = ラベル, $2 = plan パス, $3 = chunk MiB (0 = variable)
  local label="$1" plan="$2" mib="$3"
  local cfg="$TMP/cfg_${label}.json"

  python3 - "$cfg" "$plan" "$RESIDENT_BYTES" "$RING_DEPTH" "$PREFETCH" \
             "$H2D_GBPS" "$DEVICE_BYTES" "$label" <<'PY'
import json, sys
cfg, plan, resident, depth, prefetch, h2d, dev, label = sys.argv[1:9]
plan_data = json.load(open(plan, encoding="utf-8"))
per_token = plan_data["summary"]["per_token_bytes_total"] - int(resident)
json.dump({
    "label": label,
    "seed": 20260802,
    "source": {"plan_path": plan, "model_path": "", "host_store_mode": "mmap"},
    "ring": {"ring_depth": int(depth), "prefetch_depth": int(prefetch)},
    "scheduler": {"type": "fixed", "stats_window": 8,
                  "fixed_chunk_size": plan_data["summary"]["max_chunk_bytes"],
                  "fixed_prefetch_depth": int(prefetch)},
    "backend": {
        "h2d": {"gb_per_s": float(h2d), "fixed_latency_s": 0.00002,
                "jitter_fraction": 0.05},
        "prepare": {"gb_per_s": 100.0, "fixed_latency_s": 0.000005,
                    "jitter_fraction": 0.02},
        "consume_bytes_per_s": 300000000000,
        "consume_jitter_fraction": 0.02,
        "device_memory_bytes": int(dev),
        "pinned_limit_bytes": 8589934592,
        "verify_checksums": True,
    },
    "run": {"repeat_passes": 1, "resident_bytes": int(resident),
            "exclude_non_per_token_chunks": True,
            "per_token_bytes": max(per_token, 1), "materialize_data": True},
}, open(cfg, "w"), ensure_ascii=False, indent=2)
PY

  local walls=() vsec="" toks="" wait="" ok=1 msg=""
  local total=$((WARMUP + REPEATS))
  for i in $(seq 1 "$total"); do
    if ! ./build/bin/alsr-sim --config "$cfg" --results "$RESULTS" --quiet \
         --print-summary > "$TMP/out_${label}_$i.json" 2> "$TMP/err_${label}_$i.txt"; then
      ok=0
      msg="$(head -1 "$TMP/err_${label}_$i.txt")"
      break
    fi
    if [[ $i -gt $WARMUP ]]; then
      walls+=("$(python3 -c "
import json,sys
d=json.load(open('$TMP/out_${label}_$i.json'))
print(d['throughput']['wall_seconds'])")")
    fi
  done

  if [[ $ok -eq 0 ]]; then
    echo "  $label : 測定不能 — $msg"
    emit "$(python3 - "$label" "$mib" "$msg" <<'PY'
import json, sys
print(json.dumps({"label": sys.argv[1], "chunk_mib": int(sys.argv[2]),
                  "measurable": False, "reason": sys.argv[3]},
                 ensure_ascii=False))
PY
)"
    return
  fi

  emit "$(python3 - "$label" "$mib" "$TMP/out_${label}_$total.json" "${walls[@]}" <<'PY'
import json, statistics, sys
label, mib, summary_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
walls = [float(x) for x in sys.argv[4:]]
d = json.load(open(summary_path, encoding="utf-8"))
th, ov, lat, mem = d["throughput"], d["overlap_and_starvation"], d["latency"], d["memory"]
walls.sort()
def q(v, p):
    if not v: return 0.0
    return v[min(len(v)-1, max(0, int(round(p*(len(v)-1)))))]
out = {
    "label": label, "chunk_mib": mib, "measurable": True,
    "chunk_count": th["chunk_count"],
    "streamed_bytes": th["total_bytes"],
    "resident_bytes": mem["resident_bytes"],
    "virtual_seconds": th["virtual_seconds"],
    "tokens_per_s": th["tokens_per_s"],
    "effective_h2d_gbps": th["effective_h2d_gbps"],
    "gpu_wait_ratio": ov["gpu_wait_ratio"],
    "late_wait_ratio": ov["late_wait_ratio"],
    "overlap_ratio": ov["copy_compute_overlap_ratio"],
    "starvation_events": ov["starvation_events"],
    "copy_p50_ms": lat["copy_ms"]["p50"], "copy_p95_ms": lat["copy_ms"]["p95"],
    "copy_p99_ms": lat["copy_ms"]["p99"],
    "peak_device_bytes": mem["peak_device_bytes"],
    "gate_pass": d["mandatory_gates"]["pass"],
    "wall_seconds_samples": walls,
    "wall_seconds_median": statistics.median(walls) if walls else 0.0,
    "wall_seconds_p95": q(walls, 0.95),
    "note": "仮想時間は決定論的なので反復しても不変。反復が効くのは壁時計だけ",
}
print(json.dumps(out, ensure_ascii=False))
PY
)"
  python3 -c "
import json
d=json.load(open('$TMP/out_${label}_$total.json'))
th,ov=d['throughput'],d['overlap_and_starvation']
print(f\"  {'$label':<22} chunks={th['chunk_count']:<4} tok/s={th['tokens_per_s']:.4f}  \"
      f\"wait={ov['gpu_wait_ratio']*100:5.2f}%  overlap={ov['copy_compute_overlap_ratio']*100:5.2f}%\")
"
}

for mib in "${SIZES[@]}"; do
  plan="$RESULTS/sweep_plan_${mib}mib.json"
  if [[ ! -f "$plan" ]]; then
    ./build/bin/alsr-plan --ollama "$MODEL_REF" --chunk-mib "$mib" --out "$plan" --quiet
  fi
  run_case "fixed_${mib}mib" "$plan" "$mib"
done

plan_var="$RESULTS/sweep_plan_variable.json"
if [[ ! -f "$plan_var" ]]; then
  ./build/bin/alsr-plan --ollama "$MODEL_REF" --mode variable --min-mib 16 \
      --max-mib 512 --out "$plan_var" --quiet
fi
run_case "variable_16_512mib" "$plan_var" 0

echo ']}' >> "$TMP/acc.json"
python3 -c "
import json,sys
d=json.load(open('$TMP/acc.json'))
d['config']={'model':'$MODEL_REF','repeats':$REPEATS,'warmup':$WARMUP,
             'resident_bytes':$RESIDENT_BYTES,'ring_depth':$RING_DEPTH,
             'prefetch_depth':$PREFETCH,'h2d_gb_per_s':$H2D_GBPS,
             'device_memory_bytes':$DEVICE_BYTES}
json.dump(d,open('$OUT','w'),ensure_ascii=False,indent=2)
print()
print('wrote $OUT')
"
