# ALSR PoC — Adaptive Layer Streaming Runtime

RAM 上のモデル重みを Chunk に分割し、GPU 演算と RAM→VRAM 転送を重ねることで、
VRAM 容量を超えるモデルの推論にどの程度の実用領域があるかを **測定する** ための
研究用 PoC。

要件定義・基本設計の正本は [`docs/ALSR_REQ_DESIGN.md`](docs/ALSR_REQ_DESIGN.md)。

**この実装は「成功を演出しない」。成立しない条件を定量的に確定することも正式な
成果として扱う。** 何を主張してよくて何を主張してはいけないかは
[`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) にまとめてある。
性能の話をする前に必ず読むこと。

現在の到達点（2026-08-04）:

- **M2 GREEN** — Gemma 4 12Bの実推論で、Model Storeからの明示range read、
  Preparation、Queue、固定Scheduler、有限Ring、Backend、llama.cpp Executorまでを接続した。
- **M5 FROZEN** — M5での高負荷検証は終了。31B、scale、CUDA、PCIe、VRAM検証は
  Windows／RTX機へ移す。
- **NEXT: X0 → R0** — Git checkpointを復元後、推測を入れずRTX環境を採取する。

Windowsで作業を再開するときは、最初に
[`docs/WINDOWS_RTX_HANDOFF.md`](docs/WINDOWS_RTX_HANDOFF.md) と
[`docs/checkpoints/M2_M5_FREEZE_20260804/CHECKPOINT.md`](docs/checkpoints/M2_M5_FREEZE_20260804/CHECKPOINT.md)
を読むこと。M2のraw Evidence PackはGit管理外であり、小型判定成果物と期待SHA-256だけを
checkpointへ保存している。

旧Rev.1.1／Phase 0-2に対する監査履歴は
[`docs/CONFORMANCE.md`](docs/CONFORMANCE.md)に残すが、現行Rev.3.7への適合表ではない。
今後のGate判定は要件正本、M2 checkpoint、Windows handoffの受入条件で行う。

---

## クイックスタート

以下は開発用の一般手順である。**M5凍結後はM5上でmodel load、実推論、benchmark、
M2再実行を行わない。** Windows側もR0の環境採取とport境界の確認を終えるまでは、
この手順を受入試験として実行しない。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

```bash
cd build && ctest --output-on-failure
```

```bash
./build/bin/alsr-inspect --ollama gemma4:12b --manifest-out results/manifest.json
```

```bash
./build/bin/alsr-plan --ollama gemma4:12b --chunk-mib 256 --out results/plan_gemma4_256mib.json
```

```bash
bash scripts/run_samples.sh
```

必要なもの: C++20 コンパイラ、CMake 3.24+、Python 3.11+（集計のみ）。
テストは Catch2 v3.15.3 を FetchContent で取得する（commit 固定、初回のみ通信）。

---

## 何ができるか

### `alsr-inspect` — GGUF 解析と実行順序マニフェスト (FR-01 / FR-02)

読み取り専用で GGUF を解析し、テンソル一覧・レイアウト整合・実行順序・
S_stream・帯域から導かれる T_max 上限を出す。

```bash
./build/bin/alsr-inspect --ollama gemma4:12b \
  --manifest-out results/manifest.json \
  --gguf-out     results/gguf.json \
  --env-out      results/environment.json \
  --bandwidths   5,12,25,113
```

`--model <path>` で GGUF を直接指定してもよい。`--sha256` でモデル全体のダイジェスト。

終了コード: `4` = レイアウト不整合、`5` = 未分類テンソルあり（人間の確認が必要）。

### `alsr-plan` — Chunk 計画 (FR-03)

```bash
./build/bin/alsr-plan --ollama gemma4:12b --mode fixed --chunk-mib 256 \
  --out results/plan.json
./build/bin/alsr-plan --ollama gemma4:12b --mode variable --min-mib 16 --max-mib 512 \
  --out results/plan_var.json
```

生成した計画は毎回 `verify_plan()` で以下を検査する。破れていれば非ゼロ終了。

1. Chunk 群が `[data_start, file_size)` を隙間なく重複なく敷き詰める
2. 各テンソルの各バイトがちょうど 1 つの Chunk に属する
3. テンソル内部の切れ目が `lcm(量子ブロック, alignment)` の倍数

### `alsr-sim` — 仮想時間シミュレータ + Evidence Pack (FR-04〜FR-07 / FR-10 / FR-11)

```bash
./build/bin/alsr-sim --config configs/run01_gemma4_fixed_pcie4.json --results results
```

`results/<run_id>/` に設計書 FR-11 の構成で証拠を残す。

```text
results/<run_id>/
├── config.json          正規化した設定（run_id はこの内容から決定論的に決まる）
├── environment.json     §3.3 の環境証拠
├── metrics.jsonl        Chunk ごとの生ログ（追記専用）
├── summary.json         集計。metrics.jsonl の SHA-256 を含む
├── summary.csv
├── report.md
├── scheduler_log.json   制御変更の理由 (FR-06)
├── stdout.log           scripts/run_samples.sh 経由のとき
└── stderr.log
```

### `alsr-bw` — Phase 0 帯域実測

```bash
./build/bin/alsr-bw --buffer-mib 512 --repeats 7 --ollama gemma4:12b \
  --out results/bandwidth.json --md-out results/bandwidth_report.md
```

M5 で測れるのは RAM / ファイル読み出し / CRC32C まで。
PCIe H2D は Phase 3（RTX 機）。

### 集計

```bash
python3 tools/analyze_results.py results      # 横断表 + ハッシュ照合 + ゲート未達の列挙
python3 tools/make_report.py    results       # §10 比較表の充足状況つきレポート
```

`analyze_results.py` は `summary.json` に記録された SHA-256 と実際の
`metrics.jsonl` を突き合わせる。不一致があれば必ず表示する（§16.4）。

---

## 設計上の要点

### 仮想時間で駆動する

時刻は決定論的な漸化式で進める。リングバッファの状態遷移・実データの読み出し・
checksum は**実際に行う**ので、正確性の検証は本物。仮想化しているのは時刻だけ。

```text
prepare_end[i] = max(producer_busy,  slot_released[slot(i)]) + prepare(i)
copy_end[i]    = max(transfer_busy,  prepare_end[i])         + h2d(i)
compute_end[i] = max(consumer_busy,  copy_end[i])            + compute(i)
wait[i]        = max(consumer_busy, copy_end[i]) - consumer_busy
```

理由と反証条件は `docs/DECISIONS.md` DR-0004。壁時計は計測オーバーヘッドの
把握用に別途記録する（NFR-05）。

### 推測で処理を続行しない

- 未知の ggml 型 → `Error(Unsupported)`
- GGUF v3 以外 → `Error(Unsupported)`
- 未分類のテンソル名 → `unknown` として記録し、`requires_human_review` を立てる
- 非有限の実数を JSON へ書こうとしたら → `Error(Invalid)`（0 除算を黙って null にしない）
- pinned / VRAM の上限を超える設定 → 起動時に停止（NFR-03）
- 実データを流さない Run → 必須ゲートを必ず落とす（DR-0007）

### モデルファイルは読み取り専用

`open(O_RDONLY)` / `mmap(PROT_READ)` のみ。Ollama の blob を書き換えない、
コピーを作らない、Git に入れない（NFR-03 / §16.2）。

### 必須ゲート（§12.1）

checksum 一致率 100% / Chunk 欠落 0 / 重複 0 / 順序違反 0 / unhandled error 0 /
environment 保存済み。1 つでも失敗した Run は性能比較から除外する。
`alsr-sim` はゲート未達なら終了コード `6` を返す。

---

## リポジトリ構成

```text
ALSR-PoC/
├── docs/
│   ├── ALSR_REQ_DESIGN.md     現行要件定義・基本設計（Rev.3.7）
│   ├── CONFORMANCE.md         旧Rev.1.1／Phase 0-2の適合監査履歴
│   ├── DECISIONS.md           決定記録（反証条件つき）
│   ├── EXPERIMENTS.md         旧Phase 0-2記録（A0監査前、引用制限あり）
│   └── KNOWN_LIMITATIONS.md   何を主張してよくて、何を主張してはいけないか
├── include/alsr/              公開ヘッダ（GPU API から独立、NFR-04）
├── src/
│   ├── inspector/  ggml 型テーブル、GGUF パーサ、実行順序マニフェスト
│   ├── planner/    Chunk Planner
│   ├── buffer/     Host Store、リングバッファ状態機械
│   ├── scheduler/  固定 / 適応 Scheduler
│   ├── sim/        帯域エミュレータ、仮想時間シミュレータ
│   ├── telemetry/  JSON、CRC32C/SHA-256、メトリクス、環境収集
│   └── cli/        alsr-inspect / alsr-plan / alsr-sim / alsr-bw
├── backends/
│   ├── simulator/  仮想時間 backend（M5）
│   └── cuda/       Phase 3 で実装（未着手）
├── tests/          unit / integration / stress（Catch2 タグで分離）
├── configs/        サンプル実験設定 6 本（12B x3 / 31B x3）
├── scripts/        環境収集、Ollama 基準値、サンプル Run、Chunk スイープ
├── tools/          Python の集計・レポート
└── results/        Evidence Pack（Git 管理外）
```

---

## 現在証明できていること／いないこと

M2の正式判定で証明できているのは、次の限定された事実である。

- Gemma 4 12Bで128 token × 5 Runの実推論が完了した。
- 全Runで、explicit range readからfinite Ringを経由してBackendへ供給された。
- 5 Run合計の明示readは4,713,957,416,960 bytes、434,560 ranges。
- 権威Ring transitionは3,041,920件、prepared-range欠落時のfallbackは0件。
- M2 verifierは修正版PackをGREENと判定し、撤回済み旧Packを拒否する。

M2は、ALSRが既存方式より高速であること、CUDA／PCIeで機能すること、VRAM超過modelが
動くこと、31B以上の成立範囲を証明していない。12Bはport regressionの基準であり、
研究上限ではない。

旧`Phase 0-2`資料にあった帯域逆算、仮想PCIe値、simulator結果、未再実行test結果は、
実測・導出・仮定が混在していたため現行の性能事実として引用禁止である。監査上の正本は
`docs/recovery/`のA0成果物、現在の実装判定は上記M2 checkpointとする。

---

## Git 運用（§16）

- `main`: 再現済み GREEN または承認済み YELLOW のみ
- `phase/*`: Phase 単位、`experiment/*`: 一時実験
- 1 commit = 1 つの検証可能な変更、commit message に Run ID を含める
- `*.gguf` / blob / `results/` は `.gitignore` 済み。
  Evidence Pack を残したいときは `git add -f results/<run_id>` を明示的に行い、
  何を残したか commit message に書くこと。
- `work/`はupstream checkoutと中間生成物のローカル領域であり、Gitへ追加しない。

---

## ライセンス

未確定。`LICENSE` を参照。確定したら `docs/DECISIONS.md` へ決定記録を追加すること。
