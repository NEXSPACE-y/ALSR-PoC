#!/usr/bin/env python3
"""Run 群から比較レポート（Markdown）を作る。

設計書 §10 の比較表 (B0..B5 / A1 / A2) に沿った形で並べる。
Phase 0-2 では B3..B5 と A1/A2 の一部しか埋まらない。埋まらない行は
「未測定」と明示し、空欄を成果に見せない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        cfg_path = p.parent / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        out.append({"summary": s, "config": cfg, "dir": p.parent})
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "results")
    out_path = Path(argv[2] if len(argv) > 2 else root / "comparison.md")
    runs = load(root)
    if not runs:
        print(f"{root} に Run がない", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("# ALSR Run 比較レポート\n")
    lines.append("> 本レポートは `tools/make_report.py` が summary.json のみから生成する。")
    lines.append("> 手作業で数値を書き換えないこと（設計書 §16.4）。\n")

    lines.append("## Run 一覧\n")
    lines.append("| run_id | scheduler | H2D設定 | 実効H2D [GB/s] | token/s | GPU待機率 | "
                 "終盤待機率 | 重複率 | starvation | gate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in runs:
        s, c = r["summary"], r["config"]
        th, ov = s["throughput"], s["overlap_and_starvation"]
        h2d_cfg = (c.get("backend", {}).get("h2d", {}) or {}).get("gb_per_s", "?")
        lines.append(
            f"| `{s['run_id']}` | {s['scheduler']} | {h2d_cfg} | "
            f"{th['effective_h2d_gbps']:.2f} | {th['tokens_per_s']:.4f} | "
            f"{ov['gpu_wait_ratio']*100:.2f}% | {ov['late_wait_ratio']*100:.2f}% | "
            f"{ov['copy_compute_overlap_ratio']*100:.2f}% | "
            f"{ov['starvation_events']} | "
            f"{'PASS' if s['mandatory_gates']['pass'] else 'FAIL'} |"
        )

    lines.append("\n## 設計書 §10 の比較対象の充足状況\n")
    lines.append("| ID | 方式 | 状態 |")
    lines.append("|---|---|---|")
    lines.append("| B0 | CPU-only / GPU offload なし | 未測定（Phase 5 で llama.cpp 経路が要る） |")
    lines.append("| B1 | llama.cpp/Ollama 標準の最良設定 | 部分的（M5 baseline のみ。RTX 未測定） |")
    lines.append("| B2 | 固定 GPU レイヤー数 | 未測定（Phase 5） |")
    lines.append("| B3 | Pageable + 同期 copy | 未測定（Phase 3、RTX 機） |")
    lines.append("| B4 | Pinned + 同期 copy | 未測定（Phase 3、RTX 機） |")
    lines.append("| B5 | Pinned + 非同期 copy + 固定 Chunk | 未測定（Phase 3、RTX 機） |")
    lines.append("| A1 | ALSR 固定 Chunk + Ring Buffer | 仮想時間で測定済（本 Run 群） |")
    lines.append("| A2 | ALSR 可変 Chunk + 適応 Scheduler | 仮想時間で測定済（本 Run 群） |")
    lines.append("\n> 仮想時間の結果は CUDA 実測の代わりにならない。B3〜B5 が埋まるまで")
    lines.append("> ALSR の優位・劣位を主張してはならない（設計書 §3.1 / §4.3）。\n")

    lines.append("## 分位点 [ms]\n")
    for r in runs:
        s = r["summary"]
        lat = s["latency"]
        lines.append(f"### `{s['run_id']}`\n")
        lines.append("| 区間 | p50 | p95 | p99 | mean | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in ("prepare_ms", "copy_ms", "compute_ms", "wait_ms"):
            p = lat[name]
            lines.append(
                f"| {name.replace('_ms','')} | {p['p50']:.4f} | {p['p95']:.4f} | "
                f"{p['p99']:.4f} | {p['mean']:.4f} | {p['max']:.4f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
