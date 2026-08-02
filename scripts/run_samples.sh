#!/usr/bin/env bash
# サンプル 3 Run を生成し、FR-11 の Evidence Pack を完成させる。
#
# alsr-sim 自身は stdout/stderr を書かないので、ここでリダイレクトしたものを
# Run ディレクトリへ配置する（stdout.log / stderr.log）。
set -euo pipefail

RESULTS="${RESULTS:-results}"
SIM="./build/bin/alsr-sim"
PLAN="${PLAN:-$RESULTS/plan_gemma4_256mib.json}"

if [[ ! -x "$SIM" ]]; then
  echo "error: $SIM が無い。先に cmake --build build を実行すること" >&2
  exit 1
fi

if [[ ! -f "$PLAN" ]]; then
  echo "plan がまだ無いので生成する: $PLAN"
  ./build/bin/alsr-plan --ollama "${MODEL_REF:-gemma4:12b}" --chunk-mib 256 --out "$PLAN"
fi

CONFIGS=(
  configs/run01_gemma4_fixed_pcie4.json
  configs/run02_gemma4_adaptive_pcie3.json
  configs/run03_synthetic_adaptive_healthy.json
)

TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  echo "=========================================================="
  echo "config: $cfg"
  set +e
  "$SIM" --config "$cfg" --results "$RESULTS" \
    > "$TMPDIR_RUN/stdout.log" 2> "$TMPDIR_RUN/stderr.log"
  rc=$?
  set -e
  cat "$TMPDIR_RUN/stdout.log"
  if [[ -s "$TMPDIR_RUN/stderr.log" ]]; then
    echo "--- stderr ---"
    cat "$TMPDIR_RUN/stderr.log"
  fi

  RUN_ID="$(awk '/^run_id[[:space:]]*:/ {print $3; exit}' "$TMPDIR_RUN/stdout.log")"
  if [[ -n "$RUN_ID" && -d "$RESULTS/$RUN_ID" ]]; then
    cp "$TMPDIR_RUN/stdout.log" "$RESULTS/$RUN_ID/stdout.log"
    cp "$TMPDIR_RUN/stderr.log" "$RESULTS/$RUN_ID/stderr.log"
    echo "evidence complete: $RESULTS/$RUN_ID"
  else
    echo "warning: run_id を stdout から取れなかった ($cfg)" >&2
  fi

  if [[ $rc -ne 0 ]]; then
    echo "!! この Run は必須ゲートに失敗した (exit=$rc)" >&2
    FAILED=1
  fi
done

echo "=========================================================="
echo "サマリー一覧:"
python3 tools/analyze_results.py "$RESULTS" || true

exit "$FAILED"
