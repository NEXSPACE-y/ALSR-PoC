# 設計書適合監査

`docs/ALSR_REQ_DESIGN.md` の全条項に対する実装の適合状況。
**未達・逸脱を隠さないことがこの文書の唯一の目的**である。
「実装したつもり」と「条文が要求していること」がずれていないかを条文単位で当たる。

判定:

- **適合** — 条文の要求を満たしている
- **未達** — 満たしていない（理由と対応を書く）
- **Phase外** — 本 Mission（Phase 0-2）の範囲外。設計書が別 Phase に割り当てている

最終監査日: 2026-08-02（Phase 0-2 完了直後）

---

## 第1回監査で発見した逸脱（2026-08-02）

Phase 0-2 を「完了」と報告した後の再監査で **8 件**の未達が見つかった。
うち 1 件は定義レベルの逸脱である（DR-0010）。

| # | 条項 | 内容 | 状態 |
|---|---|---|---|
| 1 | §2.1 / §2.3 / §2.4 | `S_stream` の定義を取り違えていた | **修正済** DR-0010 |
| 2 | FR-02 | 依存関係を manifest へ記録していない | **修正済** |
| 3 | FR-02 | 常駐候補が `S_stream` に接続されていない | **修正済** DR-0010 |
| 4 | FR-04 | 事前ロード手段が無い | **修正済** |
| 5 | FR-06 | 出力に「次 Chunk」が無い | **修正済** |
| 6 | FR-07 | burst が未実装 | **修正済** |
| 7 | FR-11 | Evidence Pack に `manifest.json` が無い | **修正済** |
| 8 | §5.1 / NFR-02 | Chunk サイズ 5 段のスイープ / 最低 5 回反復を実行していない | **修正済** |

---

## §2 最重要の物理条件

| 条文 | 判定 | 根拠 |
|---|---|---|
| §2.1 `S_stream` = 1トークンのForward PassでRAM→VRAM**転送が必要な**重み量 | 適合 | `Manifest::stream_bytes(resident)` が常駐分を差し引いた量を返す。常駐を考慮しない量は `per_token_demand_lower/upper` という別名で持ち、`S_stream` とは呼ばない（DR-0010） |
| §2.1 `B_h2d` / `B_ram` / `B_prepare` / `T_target` / `C_i` / `t_copy` / `t_compute` / `D` | 適合 | `BandwidthProfile`（h2d / prepare）、`SimulatorBackendConfig::consume_bytes_per_s`、`Chunk::bytes`、metrics の `h2d_ms` / `compute_ms`、`SimConfig::prefetch_depth` |
| §2.2 `min(B_ram, B_prepare, B_h2d) >= S_stream × T_target` | 適合 | stress テスト「定常帯域不足なら、どの深度でも必ず枯渇する」で深度 2〜16 を実測 |
| §2.2 有限深度のバッファは平均供給不足を隠せない | 適合 | 同上。深度 2 と 16 の終盤待機率がともに 30% 超 |
| §2.3 `T_max <= B_h2d / S_stream` | 適合 | `bandwidth_ceiling_table()` / `residency_table()`。シミュレータの実測と 1% 以内で一致 |
| §2.4 「大きな重み常駐率」 | 適合 | `residency_table()` と `SimConfig::resident_bytes` |
| §2.4 405B は数理外挿で裁定する | 適合 | `alsr-inspect --extrapolate` で任意規模へ外挿できる（実ファイルは取得しない） |
| §2.5 MoE は別系統 | Phase外 | Dense PoC と分離。`TensorRole::MoeRouter` / `MoeExpert` の枠だけ用意 |

## §3 対象環境

| 条文 | 判定 | 根拠 |
|---|---|---|
| §3.1 M5 の検証範囲（基準値／解析／Planner／キュー／帯域エミュ／ログ／スケジューラ負荷） | 適合 | すべて実施 |
| §3.1 M5 で合否判定しないもの（Pinned / CUDA コピーエンジン / PCIe / 重複率 / VRAM枯渇） | 適合 | `TransferCapabilities::measured_on_real_hardware` が常に false。文書でも明記 |
| §3.2 RTX 機 | Phase外 | Phase 3 |
| §3.3 起動時に必ず収集する環境証拠 | 適合 | `collect_environment()`。取得不能は `unavailable`（捏造しない） |

## §4 目的と成功条件

| 条文 | 判定 | 根拠 |
|---|---|---|
| §4.1-1 実行順序に沿って解析・分割 | 適合 | `build_manifest()` + `build_plan()` |
| §4.1-2 Chunk の完全性を維持して先読みキューへ投入 | 適合 | CRC32C + `verify_plan()` + リングバッファ |
| §4.1-3 仮想帯域上で固定/ダブル/リング/適応を比較 | 適合 | stress テストと sweep |
| §4.1-4 RTX 上で Pinned + Stream の重複測定 | Phase外 | Phase 3 |
| §4.1-5 最良の既存方式と比較 | Phase外 | §10 B0〜B5 は Phase 3 / Phase 5 |
| §4.1-6 31B/70B/120B/405B へ外挿可能な測定モデル | 適合 | `extrapolate_table()` |
| §4.2 成功の定義 | 適合 | 判定は Phase 3 以降。指標は出力済み |
| §4.3 製品的成功は別判定 | 適合 | `KNOWN_LIMITATIONS.md` で分離 |

## §5 スコープ

### §5.1 PoC に含む

| 項目 | 判定 | 根拠 |
|---|---|---|
| GGUF/Ollama blob の構造・テンソル一覧取得 | 適合 | `alsr-inspect` |
| テンソルサイズ、配置、量子形式、アライメントの記録 | 適合 | manifest / gguf JSON |
| レイヤー／テンソル／サブテンソル単位の Chunk 計画 | 適合 | `ChunkMode::Fixed` / `Variable`、テンソル内分割 |
| **64/128/256/512/1024MiB の固定Chunk試験** | 適合 | `scripts/sweep_chunk_sizes.sh`（5 段 × warm-up 1 + 本番 5 回）。結果は `results/sweep_chunk_sizes.json` |
| 可変Chunk試験 | 適合 | 同スイープに含む |
| ダブル、トリプル、N段リングバッファ | 適合 | 深度 2/3/4/8/16 を stress とスイープで実行 |
| 帯域とレイテンシのエミュレータ | 適合 | `BandwidthModel` |
| H2D microbenchmark | Phase外 | Phase 3（M5 に経路が無い）。RAM/ファイル側は `alsr-bw` |
| Pinned / Pageable 比較 | Phase外 | Phase 3 |
| CUDA Stream 数の比較 | Phase外 | Phase 3 |
| CUDA Event による copy/kernel 時間計測 | Phase外 | Phase 3 |
| バックプレッシャー、キャンセル、タイムアウト | 適合 | 統合テスト 3 件 |
| JSONL メトリクス、サマリー CSV、Markdown レポート | 適合 | Evidence Pack |
| **llama.cpp 系実行経路への接続可否調査と最小統合** | **未達** | 下記 |

`llama.cpp` 接続可否調査は §5.1 に含まれるが、§17 の初回 Mission の実装範囲
A〜E には無く、「実装しないもの」に `llama.cpp改造` がある。**調査は禁じられて
いないが実施していない。** Phase 3 前の宿題として `KNOWN_LIMITATIONS.md` §5 に
記載する。Mission の範囲解釈で落ちた項目であり、隠さず未達とする。

### §5.2 初期 PoC に含まない

| 項目 | 判定 |
|---|---|
| 405B モデルの実ファイル取得・実推論 | 適合（取得していない。数理外挿のみ） |
| 新しい量子化方式の学習 | 適合（していない） |
| 独自 GPU 展開カーネルの本格最適化 | 適合（していない） |
| SSD→RAM→VRAM 三段キャッシュの本実装 | 適合（していない） |
| 分散ノード間通信 | 適合（していない） |
| NeutronGate 本体への統合 | 適合（していない） |
| Ollama 本体の大規模改造 | 適合（blob を読むだけ。一切変更していない） |
| 本番 API、GUI、ユーザー管理 | 適合（していない） |

## §6 機能要件

### FR-01 モデル解析

| 条文 | 判定 | 根拠 |
|---|---|---|
| GGUFヘッダ、メタデータ、テンソル一覧を読み取る | 適合 | `read_gguf()` |
| 名称・型・shape・offset・byte length・**alignment** を出力 | 適合 | `to_json(GgufFile)`。テンソルごとに `alignment` と `alignment_ok` を出す |
| 読み取り専用で動作し、モデルファイルを変更しない | 適合 | `open(O_RDONLY)` / `mmap(PROT_READ)` のみ |
| 未対応量子形式は黙って推測せず、明示的に失敗する | 適合 | `Error(Unsupported)`。単体テストで固定 |

### FR-02 実行順序マニフェスト

| 条文 | 判定 | 根拠 |
|---|---|---|
| テンソルを論理レイヤーへ関連付ける | 適合 | `ManifestEntry::layer` |
| 実行順序を manifest へ記録 | 適合 | `exec_order` / `stages` |
| **依存関係**を manifest へ記録 | 適合 | `ManifestEntry::depends_on`（計算グラフ上の先行テンソルの exec_order 集合）と `parallel_group`（順序制約が無い組）。段の先頭は前段の末尾に依存する |
| **常駐候補**を manifest へ記録 | 適合 | `resident_candidate` + `resident_priority`。`stream_bytes()` に接続済み |
| Embedding、output、norm、attention、FFN を区別 | 適合 | `TensorRole` |
| 解析できない名前規則は unknown として保存し、人間の確認を要求 | 適合 | `unknown_count` / `requires_human_review` / 終了コード 5 |

### FR-03 Chunk Planner

| 条文 | 判定 |
|---|---|
| 固定サイズと可変サイズを切り替えられる | 適合 |
| テンソル境界を優先し、分割時は量子ブロック境界と alignment を守る | 適合（`verify_plan()` で毎回検査） |
| 連番・元offset・length・checksum・関連テンソルを付与 | 適合 |
| 同じ入力と設定から同じ計画を生成 | 適合（`plan_structure_sha256`） |

### FR-04 Host Store

| 条文 | 判定 | 根拠 |
|---|---|---|
| モデル全体または対象範囲を RAM へ**事前ロード**できる | 適合 | `HostStore::preload(offset, len)` と `--preload` |
| `mmap` と明示 read の両方を比較できる | 適合 | `HostStoreMode::Mmap` / `Read` |
| CUDA 版では Pageable と Pinned の両方を計測 | Phase外 | Phase 3 |
| Pinned Memory 量に上限を設け、OS を圧迫しない | 適合 | 起動時に上限超過を `Error(Invalid)` |

### FR-05 Queue / Ring Buffer

| 条文 | 判定 |
|---|---|
| 7 状態の遷移 | 適合 |
| 単一所有権または世代番号で ABA を防ぐ | 適合 |
| producer が consumer を上書きしない | 適合 |
| キャンセル時に全 slot を安全に解放 | 適合 |

### FR-06 Scheduler

| 条文 | 判定 | 根拠 |
|---|---|---|
| 静的、適応型の両モード | 適合 | `fixed` / `adaptive` |
| 入力: queue depth, GPU処理時間, H2D時間, 待機率, VRAM余量, 直近の分散 | 適合 | `SchedulerInput` 全項目 |
| 出力: **次Chunk**, Chunkサイズ, 先読み距離, 転送優先度 | 適合 | `SchedulerDecision::next_chunk_index` を追加。既定は逐次だが選択の責務を Scheduler に置いた |
| 初期実装は決定論的な制御則とし、機械学習は使わない | 適合 | 閾値ヒューリスティックのみ |
| 制御変更の理由をログへ残す | 適合 | `scheduler_log.json` |

### FR-07 Bandwidth Emulator

| 条文 | 判定 | 根拠 |
|---|---|---|
| 帯域 | 適合 | `bytes_per_s` |
| 固定レイテンシ | 適合 | `fixed_latency_s` |
| jitter | 適合 | `jitter_fraction` |
| **burst** | 適合 | `burst_period_s` / `burst_duration_s` / `burst_multiplier` |
| 一時停止 | 適合 | `pause_period_s` / `pause_duration_s` |
| 5,10,15,20,25,28GB/s 等を再現 | 適合 | 任意値を設定可能。スイープで実行 |
| M5 上で CUDA の成立を主張せず、スケジューラの論理試験にのみ使う | 適合 | 文書とコードで明記 |

### FR-08 CUDA Transfer Backend — Phase外（Phase 3）

### FR-09 Consumer / Executor

| 条文 | 判定 |
|---|---|
| 1. Synthetic Consumer | 適合 |
| 2. CUDA Micro-kernel Consumer | Phase外（Phase 3） |
| 3. LLM Executor Adapter | Phase外（Phase 5） |

### FR-10 Telemetry

Chunk ごとの JSONL 項目（`run_id` / `chunk_id` / `bytes` / `queue_depth_before` /
`queue_depth_after` / `prepare_ms` / `h2d_ms` / `compute_ms` / `wait_ms` /
`effective_h2d_gbps` / `checksum_ok` / `scheduler_action`）: **適合**（全項目 + 診断用の絶対時刻）

Run 単位の集計 12 項目: **適合**（token/s または Chunk/s、実効H2D、GPU待機率、
copy/compute重複率、starvation 回数と最長、p50/p95/p99 copy、p50/p95/p99 compute、
Pinned量、最大RAM/VRAM、checksum failure、scheduler変更回数）

### FR-11 Evidence Pack

| ファイル | 判定 |
|---|---|
| `config.json` | 適合 |
| `environment.json` | 適合 |
| **`manifest.json`** | 適合（実 plan の Run では manifest を再生成して同梱。合成 Run では対象が無いことを明記した stub を置く） |
| `metrics.jsonl` | 適合 |
| `summary.json` | 適合 |
| `summary.csv` | 適合 |
| `report.md` | 適合 |
| `stdout.log` | 適合（`scripts/run_samples.sh` 経由） |
| `stderr.log` | 適合（同上） |

## §7 非機能要件

| 条文 | 判定 | 根拠 |
|---|---|---|
| NFR-01 すべての Chunk で checksum を検証できる | 適合 | CRC32C |
| NFR-01 順序違反、欠落、重複を検出する | 適合 | `order_violations` / `missing_chunks` / `duplicate_chunks` |
| NFR-01 実 LLM 統合後の出力／logit 差 | Phase外 | Phase 5 |
| NFR-02 乱数seed、commit、モデルdigest、環境、設定を保存 | 適合 | seed/設定は `config.json`、commit は `environment.json`、digest は `summary.model_digest`。Ollama blob はファイル名自体が SHA-256 なのでそこから取り、`model_digest_source` に出所を明記する（捏造しない） |
| NFR-02 warm-up 後に**最低 5 回** | 適合 | `scripts/sweep_chunk_sizes.sh` が warm-up 1 + 本番 5 回。ただし**仮想時間は決定論的なので反復しても不変**であり、反復が意味を持つのは壁時計（計測オーバーヘッド, NFR-05）だけ。この事実をスクリプトと出力に明記している |
| NFR-02 平均だけでなく中央値と分位点 | 適合 | `Percentiles`（p50/p95/p99/mean/max） |
| NFR-03 モデル重みを Git へ追加しない | 適合 | `.gitignore` |
| NFR-03 実機の既存モデルや Ollama blob を変更しない | 適合 | 読み取り専用 |
| NFR-03 Pinned Memory でシステムを不安定にしない | 適合 | 上限超過は起動時に停止 |
| NFR-03 OOM 時は明示的に停止し、部分成果と原因を保存 | 適合 | `Error(Invalid)` + 統合テスト |
| NFR-04 コアロジックを GPU API から分離 | 適合 | `include/alsr/` は CUDA に依存しない |
| NFR-04 simulation backend と CUDA backend が同一 scheduler interface | 適合 | `TransferBackend` |
| NFR-04 プラットフォーム固有コードを `backends/` へ隔離 | 適合 | |
| NFR-05 ホットパスの各待ち時間を区分して計測 | 適合 | prepare / h2d / compute / wait |
| NFR-05 計測自体のオーバーヘッドを測り、無計測 Run と比較 | 適合 | 仮想時間 vs 壁時計、checksum on/off の stress テスト |
| NFR-05 エラー時も最後の状態遷移を追跡できる | 適合 | `Error` メッセージにスロット ID・状態・世代を含む |

## §12 受入基準

| 条文 | 判定 |
|---|---|
| §12.1 必須ゲート 7 項目 | 適合（`evaluate_mandatory_gates()`、未達は終了コード 6） |
| §12.2 GREEN 条件 | Phase外（Phase 3 以降で判定。指標は出力済み） |
| §12.3 YELLOW | 同上 |
| §12.4 RED を正式成果として記録 | 適合（`EXPERIMENTS.md` に不成立領域を記録） |

## §13 テスト要件

| 条文 | 判定 |
|---|---|
| 13.1 GGUFヘッダ境界 / overflow / short read / alignment / 量子ブロック境界 / Chunk完全被覆 / checksum / slot状態遷移 / scheduler上下限 / JSON schema | 適合 |
| 13.2 Inspector→Planner→HostStore / Producer→Queue→Transfer→Consumer / キャンセル / timeout / OOM模擬 / corrupted model模擬 / telemetry writer | 適合 |
| 13.3 10分・1時間 / jitter 20% / 一時停止 / buffer depth 2〜16 / 小Chunk大量 / wrap-around / telemetry on-off | 適合（既定のテストは仮想 10 分。1 時間は `max_virtual_seconds=3600` で実行可能だが、常設テストには入れていない） |
| 13.4 実 LLM 正確性 | Phase外（Phase 5） |

## §16 Git・ログ・証拠管理

| 条文 | 判定 |
|---|---|
| 16.1 branch / commit 規約 | 適合（1 commit = 1 検証可能な変更、Run ID を含める） |
| 16.2 Git へ入れないもの | 適合 |
| 16.3 Decision Record の様式 | 適合（`DECISIONS.md`、反証条件つき） |
| 16.4 raw metrics は追記専用 / 集計スクリプトの commit / summary へ SHA-256 / 手作業で書き換えない | 適合 |

## §19 実装開始前チェックリスト

| 項目 | 状態 |
|---|---|
| ALSR 専用リポジトリを作成した | ✅ |
| 本書を `docs/ALSR_REQ_DESIGN.md` へ配置した | ✅（原文のまま） |
| M5 上の対象 GGUF/Ollama blob を読み取り専用で特定した | ✅ |
| 空きストレージを確認した | ✅（281 GiB 空き） |
| モデル digest を保存した | ✅ |
| Claude Code で Sonnet 5 を選択・記録した | ⚠️ Opus 5 へ変更（DR-0001） |
| 大容量物を `.gitignore` へ追加した | ✅ |
| Phase 0-2 以外を初回 Mission へ混ぜていない | ✅ |
| RTX 機上の Neutron Gate 学習が終了するまで RTX 作業を開始しない | ✅（RTX に一切触れていない） |
