#!/usr/bin/env python3
"""results/ 以下の Run を横断で集計する。

設計書 §16.4「結果の改変防止」に従い、このスクリプトは
  - raw metrics (metrics.jsonl) を読み取り専用でしか触らない
  - summary.json に記録された SHA-256 と実ファイルのハッシュを突き合わせる
  - 不一致があれば必ず表示する（黙って通さない）
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        s = json.loads(summary_path.read_text(encoding="utf-8"))

        integrity = "n/a"
        metrics = run_dir / "metrics.jsonl"
        recorded = s.get("metrics_sha256", "")
        if metrics.exists() and recorded:
            integrity = "OK" if sha256_file(metrics) == recorded else "MISMATCH"
        elif metrics.exists():
            integrity = "no-hash"

        runs.append({"dir": run_dir, "summary": s, "integrity": integrity})
    return runs


def fmt(v, spec="{:.3f}"):
    return spec.format(v) if isinstance(v, (int, float)) else str(v)


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "results")
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    runs = load_runs(root)
    if not runs:
        print(f"{root} に Run がない")
        return 0

    header = [
        "run_id", "sched", "chunks", "GiB", "virt_s", "tok/s",
        "H2D GB/s", "wait%", "late%", "overlap%", "starv", "crc_fail",
        "gate", "metrics_sha256",
    ]
    rows = []
    for r in runs:
        s = r["summary"]
        th, ov, corr = s["throughput"], s["overlap_and_starvation"], s["correctness"]
        rows.append([
            s["run_id"],
            s["scheduler"],
            th["chunk_count"],
            fmt(th["total_bytes"] / (1024 ** 3), "{:.2f}"),
            fmt(th["virtual_seconds"], "{:.4f}"),
            fmt(th["tokens_per_s"], "{:.4f}"),
            fmt(th["effective_h2d_gbps"], "{:.2f}"),
            fmt(ov["gpu_wait_ratio"] * 100, "{:.2f}"),
            fmt(ov["late_wait_ratio"] * 100, "{:.2f}"),
            fmt(ov["copy_compute_overlap_ratio"] * 100, "{:.2f}"),
            ov["starvation_events"],
            corr["checksum_failures"],
            "PASS" if s["mandatory_gates"]["pass"] else "FAIL",
            r["integrity"],
        ])

    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows))
              for i, h in enumerate(header)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    bad = [r for r in runs if r["integrity"] == "MISMATCH"]
    if bad:
        print()
        print("!! metrics.jsonl のハッシュが summary.json と一致しない Run がある。")
        print("!! 集計結果を採用してはならない (設計書 §16.4)。")
        for r in bad:
            print(f"   - {r['dir']}")

    failed = [r for r in runs if not r["summary"]["mandatory_gates"]["pass"]]
    if failed:
        print()
        print("必須ゲート未達の Run（性能比較から除外すること, §12.1）:")
        for r in failed:
            print(f"  - {r['summary']['run_id']}")
            for f in r["summary"]["mandatory_gates"]["failures"]:
                print(f"      {f}")

    out_csv = root / "all_runs.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
