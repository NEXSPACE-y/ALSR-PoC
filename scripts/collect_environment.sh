#!/usr/bin/env bash
# 設計書 §3.3「起動時に必ず収集する環境証拠」を 1 ファイルへまとめる。
# 取得できない項目は捏造せず unavailable のままにする（§20）。
set -euo pipefail

OUT="${1:-results/environment.json}"
mkdir -p "$(dirname "$OUT")"

BIN="./build/bin/alsr-inspect"
if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN が無い。先に cmake --build build を実行すること" >&2
  exit 1
fi

# alsr-inspect --env-out がそのまま §3.3 の収集器になっている。
# モデル引数が必要なので、Ollama の gemma4:12b を既定で使う。
MODEL_REF="${MODEL_REF:-gemma4:12b}"
"$BIN" --ollama "$MODEL_REF" --env-out "$OUT" --quiet >/dev/null

echo "wrote $OUT"
python3 - "$OUT" <<'PY'
import json, sys
e = json.load(open(sys.argv[1], encoding="utf-8"))
missing = []
def walk(prefix, node):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(f"{prefix}.{k}" if prefix else k, v)
    elif node == "unavailable":
        missing.append(prefix)
walk("", e)
print(f"収集項目のうち unavailable: {len(missing)} 件")
for m in missing:
    print(f"  - {m}")
print()
print("※ M5 では GPU/CUDA 系が unavailable なのは正常（設計書 §3.1）。")
print("※ RTX 検証機では nvidia_smi / cuda_version が埋まること、および")
print("   pcie.link.gen.current / width.current を必ず確認すること（§3.2）。")
PY
