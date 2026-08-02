# Adaptive Layer Streaming Runtime（ALSR）PoC

## 要件定義書・基本設計書

| 項目 | 内容 |
|---|---|
| 文書版 | 1.1（参照会話からの復元・再検証版） |
| 作成日 | 2026-08-02 |
| 対象 | M5 MacBook Air 16GB → Ryzen 7 5700X / RTX 3060 12GB 検証機 |
| 初期モデル | Ollama `gemma4:12b` |
| 最終研究対象 | 大規模Denseモデル（最終指標: Llama 3.1 405B級） |
| 実装主担当 | Claude Code / Claude Sonnet 5 |
| 状態 | 実装開始可能 |

---

## 0. 文書の位置づけ

本書は、前チャットで作成されたものの受け渡しに失敗した設計書を、会話履歴から復元し、2026-08-02時点の公式情報と物理条件を再確認した版である。

この研究の目的は、成功を演出することではない。RAM上のモデル重みを小さなChunkに分割し、GPU演算とRAM→VRAM転送を重ねることで、VRAM容量を超えるモデルの推論にどの程度の実用領域があるかを測定する。成立しない条件を定量的に確定することも、正式な成果とする。

本書では、以下を明確に分離する。

- M5で検証する制御ロジック、モデル解析、計測系
- RTX 3060で検証するPCIe、Pinned Memory、CUDA Stream、VRAMリングバッファ
- DenseモデルとMoEモデルの成立条件
- 「モデルを起動できること」と「実用速度で継続推論できること」

---

## 1. 背景と研究仮説

### 1.1 背景

一般的なローカルLLM推論では、モデル重みの大部分をVRAMへ常駐させる。モデルがVRAMへ収まらない場合は、CPUオフロード、Unified Memory、複数GPU、または低ビット量子化を利用するが、速度はメモリ階層間の転送に強く制約される。

本研究では、次のメモリ階層を明示的に制御する。

```text
SSD（永続保存）
  ↓ 起動時ロード、将来は第3キャッシュ
RAM（モデル全体または大部分を保持）
  ↓ 非同期・先読み・Chunk転送
VRAM（高速リングバッファ／作業集合）
  ↓
GPU（推論カーネル）
```

### 1.2 仮説

モデルの実行順序を事前解析し、次に必要な重みを予測できる場合、以下を組み合わせることで要求駆動型オフロードよりGPU待機を減らせる可能性がある。

1. モデル構造とテンソル実行順の事前解析
2. レイヤー／テンソル境界を尊重した可変Chunk生成
3. Pinned Host Memory
4. 非デフォルトCUDA Streamによる非同期H2D転送
5. 演算と転送のオーバーラップ
6. ダブル／トリプル／リングバッファ
7. GPU消費速度に追従する適応型スケジューラ
8. 実行履歴に基づく先読み距離とChunkサイズの調整

### 1.3 非仮説

ALSRは次を主張しない。

- PCIeの物理帯域を超える。
- RAMがVRAMと同じ速度になる。
- RTX 3060の演算性能が上位GPUと同等になる。
- リングバッファを深くすれば、平均帯域不足を無限に隠せる。
- M5上のMetal結果だけでCUDA実装の成立性を証明できる。
- 405B DenseモデルをRTX 3060で実用速度運転できることを事前に保証する。

---

## 2. 最重要の物理条件

### 2.1 定義

| 記号 | 意味 |
|---|---|
| `S_stream` | 1トークンのForward PassでRAM→VRAM転送が必要な重み量 [byte/token] |
| `B_h2d` | 実測の持続H2D帯域 [byte/s] |
| `B_ram` | 実測のRAM読み出し帯域 [byte/s] |
| `B_prepare` | 変換・展開・再配置を含むChunk準備速度 [byte/s] |
| `T_target` | 目標生成速度 [token/s] |
| `C_i` | Chunk `i` の転送バイト数 |
| `t_copy(i)` | Chunk `i` のH2D転送時間 |
| `t_compute(i)` | GPUがChunk `i` を使用して演算する時間 |
| `D` | VRAM上の先読みバッファ深度 |

### 2.2 定常状態の必要条件

長時間連続運転でGPUを止めないための必要条件は次である。

```text
実効供給帯域 >= 実効消費帯域

min(B_ram, B_prepare, B_h2d) >= S_stream × T_target
```

Chunk単位では、理想的なオーバーラップが成立するために概ね次が必要となる。

```text
t_prepare(i+1) + t_copy(i+1) <= t_compute(i) + 吸収可能なバッファ余裕
```

バッファはOS揺らぎ、転送レイテンシ、カーネル時間のばらつきを吸収する。しかし平均供給速度が平均消費速度を下回る場合、任意の有限深度のバッファは定常状態で必ず空になる。

### 2.3 Denseモデルの帯域上限

Batch=1の自己回帰生成では、Denseモデルは原則として各トークンで全レイヤーの重みを使用する。VRAMへ保持できない部分を毎トークン転送する場合、データ転送だけから導かれる上限は次となる。

```text
T_max <= B_h2d / S_stream
```

例として、`gemma4:12b`の7.6GB相当を毎トークン全量転送し、実測H2Dを仮に25GB/sと置くと、転送だけの理想上限は約3.3 token/sである。カーネル実行、同期、KVキャッシュ、メタデータ、再配置を含めると実効値はさらに低い。

M5で観測された約15～16 token/sは、7.6GB級モデルがユニファイドメモリ上で動作した基準値であり、同じ速度をPCIeストリーミングで再現できる根拠ではない。

### 2.4 405B級の意味

Llama 3.1 405BはDenseなdecoder-only Transformerであり、最終的なストレス指標として適切である。一方、理想的な4bitだけでも重みは約202.5GBで、実際の量子形式ではスケール、メタデータ、アライメント等が加わる。

仮に毎トークン200GBを25GB/sのH2Dで供給するなら、データ転送だけの上限は約0.125 token/sである。したがって405B級を実用化するには、単純な全重みストリーミング以外の追加条件が必要になる。

- 大きな重み常駐率
- 重み再利用の増加（Batch、投機、別実行方式）
- 複数GPU／高帯域接続
- アクティブ重みが少ないMoE
- 追加圧縮と十分高速なGPU側展開
- アーキテクチャ自体の変更

405Bは「PoCの最初の実装対象」ではなく、得られた式と測定値を外挿して成立／不成立を裁定する最終研究指標とする。

### 2.5 MoEは別系統の研究対象

`gpt-oss-120b`は116.8B総パラメータ、5.13Bアクティブパラメータ、60.8GiBチェックポイントのMoEである。Denseより1トークン当たりのアクティブ重みが少なく、ALSRとの相性が良い可能性がある。

ただし、次レイヤーのExpert選択はルーター出力に依存するため、完全な事前予測はできない。MoE対応はDense PoCとは分離し、以下を追加研究する。

- ルーティング確定後の緊急転送
- 上位確率Expertの投機的プリフェッチ
- Expert常駐キャッシュ
- プリフェッチ命中率／廃棄率
- 共有重みとExpert重みの異なる優先度

---

## 3. 対象環境

### 3.1 M5事前検証機

| 項目 | 内容 |
|---|---|
| 機種 | MacBook Air M5 |
| ユニファイドメモリ | 16GB |
| 推論ランタイム | Ollama / Metal |
| 初期モデル | `gemma4:12b`（Ollama上7.6GB） |
| 実測モデル占有 | 約8.0GB |
| Context | 4096 |
| 実測生成速度 | 約15.35～16.49 token/s |
| 実測Processor | 100% GPU表示 |

上記実測値はユーザー提供結果に基づく。M5での検証範囲は次に限定する。

- Ollama通常推論の再現可能な基準値取得
- モデル／GGUFメタデータ解析
- Chunk Planner
- キュー、リングバッファ、バックプレッシャー
- 帯域エミュレーション
- ログ、可視化、再現性
- スケジューラのCPU負荷と正当性

M5では独立VRAMへのPCIe H2D経路がないため、以下は合否判定しない。

- Pinned Memoryの効果
- CUDAコピーエンジン
- PCIe実効帯域
- CUDAカーネルとH2Dの重複率
- VRAM枯渇時の実GPU待機

### 3.2 RTX/CUDA本検証機

| 項目 | 構成 |
|---|---|
| CPU | AMD Ryzen 7 5700X（8C/16T、Zen 3） |
| Motherboard | MSI MEG X570 UNIFY |
| RAM | DDR4-3200 CL22、16GB×2 + 32GB×2 = 96GB |
| GPU | NVIDIA GeForce RTX 3060 12GB |
| OS | 実装開始時に記録すること |
| CUDA / Driver | 実装開始時に固定・記録すること |

注意事項:

- 異容量DIMM 4枚構成のため、理論51.2GB/sを実効値として扱わない。STREAM系ベンチマークで読み出し帯域を測る。
- RTX 3060の製品仕様は12GB GDDR6、192-bitである。ボード固有仕様は実機から取得する。
- MEG X570 UNIFYは対応CPUで1枚時x16、2枚時x8/x8となる。将来の2 GPU構成ではGPUごとのPCIe上限が半減し得るため、必ずリンク幅を実測する。
- `nvidia-smi`表示だけに頼らず、CUDA sample、PCIeトポロジ、H2D microbenchmarkで確認する。

### 3.3 起動時に必ず収集する環境証拠

```text
OS / kernel
CPU model / governor
RAM容量・速度・チャネル状態
GPU名・VRAM・PCIe generation・current/max link width
NVIDIA driver
CUDA runtime / toolkit
CMake / compiler
Ollama / llama.cpp commit
モデル名・digest・量子化形式・ファイルサイズ
常駐プロセスと空きRAM/VRAM
```

---

## 4. 目的と成功条件

### 4.1 PoC目的

1. モデル重みを実行順序に沿って解析・分割できること。
2. Chunkの完全性を維持したまま、先読みキューへ投入できること。
3. 仮想帯域上で、固定Chunk／ダブルバッファ／リングバッファ／適応制御を比較できること。
4. RTX 3060上で、Pinned MemoryとCUDA StreamによりH2Dと演算の重複を測定できること。
5. 最良の既存方式と比較し、ALSRの改善量または不成立理由を示せること。
6. 31B、70B、120B、405B級への外挿可能な測定モデルを作ること。

### 4.2 研究成果としての成功

次のいずれかを証拠付きで達成すればPoCは研究成果として成功とする。

- ALSRが比較対象より有意にGPU待機を削減する領域を発見した。
- Chunkサイズ、バッファ深度、先読み距離の最適領域を特定した。
- 特定のモデル規模／量子形式／GPUで物理的に成立しないことを定量的に証明した。
- DenseとMoEで必要なスケジューラ設計が異なることを実測した。

### 4.3 製品的な成功は別判定

PoCが研究として成功しても、次を満たさなければ実用ランタイムとは判定しない。

- 出力品質が基準実装と一致する。
- 長時間運転で枯渇、メモリリーク、デッドロックがない。
- 最良の既存オフロード実装に対して再現可能な速度優位がある。
- セットアップ複雑性に見合う効果がある。

---

## 5. スコープ

### 5.1 PoCに含む

- GGUF/Ollama blobの構造・テンソル一覧取得
- テンソルサイズ、配置、量子形式、アライメントの記録
- レイヤー／テンソル／サブテンソル単位のChunk計画
- 64、128、256、512、1024MiBの固定Chunk試験
- 可変Chunk試験
- ダブル、トリプル、N段リングバッファ
- 帯域とレイテンシのエミュレータ
- H2D microbenchmark
- Pinned / Pageable比較
- CUDA Stream数の比較
- CUDA Eventによるcopy/kernel時間計測
- バックプレッシャー、キャンセル、タイムアウト
- JSONLメトリクス、サマリーCSV、Markdownレポート
- llama.cpp系実行経路への接続可否調査と最小統合

### 5.2 初期PoCに含まない

- 405Bモデルの実ファイル取得・実推論
- 新しい量子化方式の学習
- 独自GPU展開カーネルの本格最適化
- SSD→RAM→VRAM三段キャッシュの本実装
- 分散ノード間通信
- NeutronGate本体への統合
- Ollama本体の大規模改造
- 本番API、GUI、ユーザー管理

### 5.3 将来スコープ

- SSDを第3キャッシュとする非同期I/O
- DirectStorage / GPUDirect Storage相当の調査
- MoE Expertキャッシュ
- 複数GPU
- 圧縮転送とGPU側展開
- 実行履歴を使った予測プリフェッチ
- NeutronGateへのモジュール統合

---

## 6. 機能要件

### FR-01 モデル解析

- GGUFヘッダ、メタデータ、テンソル一覧を読み取る。
- 各テンソルの名称、型、shape、offset、byte length、alignmentを出力する。
- 読み取り専用で動作し、モデルファイルを変更しない。
- 未対応量子形式は黙って推測せず、明示的に失敗する。

### FR-02 実行順序マニフェスト

- テンソルを論理レイヤーへ関連付ける。
- 実行順序、依存関係、常駐候補をmanifestへ記録する。
- Embedding、output、norm、attention、FFNを区別する。
- 解析できない名前規則は`unknown`として保存し、人間の確認を要求する。

### FR-03 Chunk Planner

- 固定サイズと可変サイズを切り替えられる。
- テンソル境界を優先し、分割が必要な場合は量子ブロック境界とアライメントを守る。
- 各Chunkへ連番、元offset、length、checksum、関連テンソルを付与する。
- 同じ入力と設定から同じ計画を生成する。

### FR-04 Host Store

- モデル全体または対象範囲をRAMへ事前ロードできる。
- `mmap`と明示readの両方を比較できる。
- CUDA版ではPageableとPinnedの両方を計測できる。
- Pinned Memory量に上限を設け、OSを圧迫しない。

### FR-05 Queue / Ring Buffer

- `FREE → FILLING → READY → COPYING → RESIDENT → CONSUMING → FREE`の状態遷移を持つ。
- 単一所有権または世代番号によりABA問題を防ぐ。
- producerがconsumerを上書きしない。
- キャンセル時に全slotを安全に解放する。

### FR-06 Scheduler

- 静的、適応型の両モードを持つ。
- 入力: queue depth、GPU処理時間、H2D時間、待機率、VRAM余量、直近の分散。
- 出力: 次Chunk、Chunkサイズ、先読み距離、転送優先度。
- 初期実装は決定論的な制御則とし、機械学習は使わない。
- 制御変更の理由をログへ残す。

### FR-07 Bandwidth Emulator

- 帯域、固定レイテンシ、jitter、burst、一時停止を設定できる。
- 5、10、15、20、25、28GB/s等を再現できる。
- M5上でCUDAの成立を主張せず、スケジューラの論理試験にのみ使う。

### FR-08 CUDA Transfer Backend

- `cudaMemcpyAsync`を非デフォルトStreamで使用する。
- H2Dとkernelを異なる非デフォルトStreamで実行できる。
- CUDA Eventで処理区間を計測する。
- `deviceOverlap` / `asyncEngineCount`等の能力を取得する。
- 同期範囲を最小化し、全Device同期をホットパスで乱用しない。

### FR-09 Consumer / Executor

段階的に3種類を実装する。

1. Synthetic Consumer: 指定時間でChunkを消費し、完全性を検証する。
2. CUDA Micro-kernel Consumer: 実データへ軽量演算し、copy/compute重複を測る。
3. LLM Executor Adapter: llama.cpp等の実推論経路へ接続する。

### FR-10 Telemetry

最低限、次をChunkごとにJSONLで記録する。

```json
{
  "run_id": "...",
  "chunk_id": 42,
  "bytes": 268435456,
  "queue_depth_before": 3,
  "queue_depth_after": 2,
  "prepare_ms": 0.0,
  "h2d_ms": 0.0,
  "compute_ms": 0.0,
  "wait_ms": 0.0,
  "effective_h2d_gbps": 0.0,
  "checksum_ok": true,
  "scheduler_action": "keep"
}
```

Run単位では次を集計する。

- token/sまたはChunk/s
- 実効H2D GB/s
- GPU待機率
- copy/compute重複率
- queue starvation回数と最長時間
- p50/p95/p99 copy時間
- p50/p95/p99 compute時間
- Pinned Memory量
- 最大RAM/VRAM使用量
- checksum failure
- scheduler変更回数

### FR-11 Evidence Pack

各Runは次の構造で保存する。

```text
results/<run_id>/
├── config.json
├── environment.json
├── manifest.json
├── metrics.jsonl
├── summary.json
├── summary.csv
├── report.md
├── stdout.log
└── stderr.log
```

---

## 7. 非機能要件

### NFR-01 正確性

- すべてのChunkでchecksumを検証できる。
- 順序違反、欠落、重複を検出する。
- 実LLM統合後は同一seed・設定で基準実装との出力／logit差を評価する。

### NFR-02 再現性

- 乱数seed、commit、モデルdigest、環境、設定を保存する。
- ベンチマークはwarm-up後に最低5回実施する。
- 平均だけでなく中央値と分位点を報告する。

### NFR-03 安全性

- モデル重みをGitへ追加しない。
- 実機の既存モデルやOllama blobを変更しない。
- システム全体を不安定にするほどPinned Memoryを確保しない。
- OOM時は明示的に停止し、部分成果と原因を保存する。

### NFR-04 移植性

- コアロジックをGPU APIから分離する。
- M5のsimulation backendとCUDA backendが同一scheduler interfaceを使う。
- プラットフォーム固有コードを`backends/`へ隔離する。

### NFR-05 観測可能性

- ホットパスの各待ち時間を区分して計測する。
- 計測自体のオーバーヘッドを測り、無計測Runと比較する。
- エラー時も最後の状態遷移を追跡できる。

---

## 8. 基本アーキテクチャ

```text
Model File / Ollama Blob
          │
          ▼
┌────────────────────┐
│ Model Inspector    │─→ tensor_manifest.json
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Chunk Planner      │─→ chunk_plan.json
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Host Store         │ mmap/read/pinned slabs
└─────────┬──────────┘
          ▼
┌────────────────────┐     telemetry
│ Adaptive Scheduler │──────────────────┐
└─────────┬──────────┘                  │
          ▼                             ▼
┌────────────────────┐         ┌────────────────┐
│ Ready Queue        │         │ Metrics Writer │
└─────────┬──────────┘         └────────────────┘
          ▼
┌────────────────────┐
│ Transfer Backend   │ simulator / CUDA H2D
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Device Ring Buffer │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Consumer / Executor│ synthetic / kernel / LLM
└────────────────────┘
```

### 8.1 スレッド／責務

| 実行単位 | 責務 |
|---|---|
| Inspector | モデル解析。Run開始前に完了させる。 |
| Planner | Chunk計画の生成。初期版はRun開始前に完了。 |
| Producer | Host Storeから空slotへデータを準備。 |
| Scheduler | 次の転送と適応制御を決定。 |
| Transfer | H2D発行と完了イベント管理。 |
| Consumer | Chunkを使用し、完了イベントを通知。 |
| Telemetry | 非同期にログを永続化。ホットパスを塞がない。 |

初期版で無理に全処理を別スレッド化しない。まず状態機械の正しさを優先し、プロファイル結果に基づいて分離する。

### 8.2 Buffer Slot

```text
slot_id
generation
state
host_ptr
device_ptr
capacity
payload_bytes
chunk_id
source_offset
checksum
copy_event
compute_event
```

### 8.3 Schedulerの初期制御則

初期版は以下の単純なヒューリスティックとする。

```text
if starvation_detected:
    prefetch_depth += 1
    chunk_size = next_lower_size(chunk_size)
elif p95_copy_time < 0.5 * p50_compute_time and queue_is_stable:
    chunk_size = next_higher_size(chunk_size)
elif pinned_memory_pressure or vram_pressure:
    prefetch_depth -= 1
else:
    keep
```

最適化前に、各変更が本当に待機を減らしたかをA/Bで検証する。GPU使用率だけを目的関数にせず、end-to-end throughputと正確性を最優先する。

---

## 9. 技術構成

### 9.1 推奨言語

- コア／CUDA統合: C++20
- ビルド: CMake
- CUDA backend: CUDA C++
- 解析、集計、グラフ: Python 3.11+
- テスト: Catch2またはGoogleTest（依存を固定）
- 実験設定: JSON
- ログ: JSONL + CSV + Markdown

理由は、最終的にllama.cpp系のC/C++経路とCUDAへ接続するためである。Pythonだけでスケジューラの概念実証を終わらせず、ホットパスの実装へ移行可能な構造にする。

### 9.2 予定リポジトリ構成

```text
ALSR-PoC/
├── README.md
├── LICENSE
├── CMakeLists.txt
├── .gitignore
├── docs/
│   ├── ALSR_REQ_DESIGN.md
│   ├── DECISIONS.md
│   └── EXPERIMENTS.md
├── include/alsr/
├── src/
│   ├── inspector/
│   ├── planner/
│   ├── scheduler/
│   ├── buffer/
│   ├── telemetry/
│   └── cli/
├── backends/
│   ├── simulator/
│   └── cuda/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── stress/
├── tools/
│   ├── analyze_results.py
│   └── make_report.py
├── configs/
├── scripts/
├── results/
└── third_party/
```

### 9.3 llama.cpp統合方針

1. 最初は外部依存なしのSynthetic Consumerを作る。
2. 次にCUDA micro-kernelでcopy/compute重複を測る。
3. llama.cppの固定GPUレイヤーオフロードを比較基準として取得する。
4. 既存APIだけで必要なレイヤー交換ができない場合は、固定commitのforkまたは小さなpatchとして統合する。
5. Ollamaは基準値取得とモデル管理に使用し、初期PoCで本体へ直接改造を入れない。

---

## 10. 比較対象

すべて同じモデル、量子形式、context、prompt、seed、生成長で比較する。

| ID | 方式 |
|---|---|
| B0 | CPU-onlyまたはGPU offloadなし |
| B1 | llama.cpp/Ollama標準の最良設定 |
| B2 | 固定GPUレイヤー数 |
| B3 | Pageable Memory + 同期copy |
| B4 | Pinned Memory + 同期copy |
| B5 | Pinned Memory + 非同期copy + 固定Chunk |
| A1 | ALSR固定Chunk + Ring Buffer |
| A2 | ALSR可変Chunk + 適応Scheduler |

標準実装の最良値を恣意的に弱く設定してはならない。

---

## 11. 実験計画

### Phase 0: 基準値と環境固定

- M5上の`gemma4:12b`基準値を同一スクリプトで再取得する。
- model digest、Ollama版、macOS、context、生成長を保存する。
- RTX機のハードウェア、Driver、CUDA、PCIe linkを採取する。
- RAM、Pageable H2D、Pinned H2Dの帯域を測る。

成果物:

```text
environment.json
baseline_m5.json
baseline_rtx.json
bandwidth_report.md
```

### Phase 1: Model Inspector

- Ollama blobまたはGGUFを発見する。
- 読み取り専用でmanifestを生成する。
- 総byte数とファイルサイズの整合性を検証する。
- 代表テンソルのoffsetとchecksumを確認する。

合格条件:

- 解析対象byte範囲の重複／欠落が説明可能。
- 同じモデルでmanifestが決定論的。
- 未対応形式を明示エラーにできる。

### Phase 2: M5 Chunk Simulator

- 64～1024MiB固定Chunkを比較する。
- bandwidth 5～28GB/s、jitter 0～20%を注入する。
- buffer depth 2、3、4、8を比較する。
- 固定Schedulerと適応Schedulerを比較する。

目的はキューと制御則の検証であり、CUDA性能の予測ではない。

### Phase 3: RTX H2D / CUDA Microbenchmark

- Pageable vs Pinned
- Sync vs Async
- Stream 1、2、3、4
- Chunk 16、32、64、128、256、512、1024MiB
- copy only / compute only / overlap
- VRAM圧迫なし／あり

`cudaEvent`と壁時計の両方を保存する。

### Phase 4: Synthetic Streaming Runtime

- 実モデルChunkをH2Dへ送り、consumer kernelで内容を使用する。
- checksum、順序、枯渇、オーバーラップを測る。
- 10分の定常運転を実施する。
- schedulerの適応が振動しないか確認する。

### Phase 5: 実LLM経路

- Gemma 4 12Bと互換な実行経路を選定する。
- まず通常推論のlogit／出力を固定する。
- 既存の固定offloadを基準にする。
- ALSR経路を段階的に接続する。
- 正確性が確認できるまで性能値を正式採用しない。

### Phase 6: 規模拡張

順序は次とする。

```text
12B（機構検証）
  ↓
31B前後（VRAM超過の実用検証）
  ↓
70B（96GB RAM内のDense検証）
  ↓
120B級（MoEとDenseを分離）
  ↓
405B（原則は数理外挿。実ファイル試験は別承認）
```

---

## 12. 受入基準

### 12.1 必須ゲート

以下の1つでも失敗したRunは性能比較から除外する。

- checksum一致率100%
- Chunk欠落0
- Chunk重複0
- 順序違反0
- unhandled error 0
- 出力正確性が基準内
- environmentと設定が保存済み

### 12.2 GREEN

機構PoCをGREENとする条件:

- 10分定常運転で進行性のqueue starvationがない。
- GPU待機率p95が5%未満、または比較対象から50%以上減少する。
- 実効H2Dが同一条件のmicrobenchmark上限の90%以上。
- end-to-end throughputが最良互換baselineより20%以上高い。
- 最大RAM/VRAMが設定上限内。
- 5回以上のRunで傾向が再現する。

### 12.3 YELLOW

- 正確性は満たすが改善が0～20%。
- 待機率5～20%。
- 特定Chunkや特定先読み距離でのみ改善。
- 効果はあるがメモリ量、複雑性、安定性のコストが高い。

YELLOWは追加実験対象とする。

### 12.4 RED

- データ破損、欠落、順序違反、デッドロック。
- 平均供給帯域不足による進行性の枯渇。
- 最良baselineより10%以上遅い状態が再現。
- Pinned MemoryやVRAMの圧迫でOS／Driverが不安定。
- 物理上限から目標token/sが不可能。

REDは失敗の隠蔽ではなく、成立しない領域を確定した正式成果として記録する。

---

## 13. テスト要件

### 13.1 Unit Test

- GGUFヘッダ境界
- overflow / short read
- tensor alignment
- 量子ブロック境界
- Chunk分割の完全被覆
- checksum
- slot状態遷移
- scheduler上下限
- JSON schema

### 13.2 Integration Test

- Inspector → Planner → Host Store
- Producer → Queue → Transfer → Consumer
- キャンセル
- timeout
- OOM模擬
- corrupted model模擬
- telemetry writer遅延

### 13.3 Stress Test

- 10分、1時間
- jitter 20%
- 一時停止100ms～1s
- buffer depth 2～16
- 小Chunk大量投入
- wrap-aroundを多数回発生
- telemetry on/off比較

### 13.4 実LLM正確性

- 固定prompt set
- temperature 0
- 固定seed
- 同一context
- 同一生成上限
- token列またはlogit差
- perplexity差（可能なら）

---

## 14. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| 平均PCIe帯域が不足 | 定常状態で必ずGPUが待つ | 実装前に式とmicrobenchmarkで上限判定 |
| M5結果の過大解釈 | CUDAで再現しない | MetalとCUDAの合否を完全分離 |
| Mixed DIMMの帯域低下 | 想定より供給が遅い | 実測値のみ採用 |
| 2 GPUでx8/x8 | GPUごとのH2D上限低下 | topologyとcurrent link widthをRunごとに保存 |
| Pinned Memory過多 | OS性能悪化 | 上限、pool化、明示解放 |
| 細かすぎるChunk | 発行・同期オーバーヘッド増 | 16MiBから1GiBまでスイープ |
| 大きすぎるChunk | 待機とVRAM浪費 | 適応制御、VRAM watermarks |
| Scheduler振動 | 性能不安定 | hysteresis、変更間隔、A/Bログ |
| GGUF内部形式差 | 誤分割 | 量子ブロックとalignmentを検証 |
| llama.cpp内部変更 | patchが壊れる | commit pin、薄いadapter、CI |
| モデル出力劣化 | 性能値が無意味 | 正確性ゲートを性能より先に判定 |
| 計測オーバーヘッド | 結果が歪む | telemetry on/off比較 |
| 巨大モデル取得 | ストレージ・時間浪費 | 数理外挿後に別承認 |

---

## 15. 担当モデルの選定

### 15.1 選定結果

| 役割 | 担当 | 理由 |
|---|---|---|
| 主実装 | Claude Code + Claude Sonnet 5 | 長時間のagentic coding、ツール利用、実装コストのバランス |
| 難所の設計・監査 | Claude Fable 5 | 複雑な長期推論、未知のツール、研究判断。常用せずゲート式で利用 |
| 代替難所担当 | Claude Opus（利用可能な最新alias） | Fable制限時、または保守的な実装監査 |
| 軽作業補助 | Gemini Flash（ユーザー環境で利用可能な版） | ログ整形、表作成、独立した低リスク作業のみ |
| 最終監査 | 人間 + 必要時のみCodex | ユーザー申告でCodex週間残量8%のため常用しない |

### 15.2 モデル名の扱い

2026-08-02時点で、公開一次情報からClaude Sonnet 5、Fable 5、Opus 4.8を確認した。ユーザー環境で「Opus 5」が表示される場合でも、CLIに未確認のmodel IDを固定しない。

- Sonnet 5はAPI名`claude-sonnet-5`が公式に公開されている。
- Fable 5は`claude-fable-5`が公開されているが、利用枠・セーフガードによるフォールバックを考慮する。
- OpusはClaude Codeの`opus` aliasまたは実機で確認した正式IDを使う。
- モデルIDと選択結果は各Missionログへ残す。

### 15.3 エスカレーション条件

Sonnet 5からFable 5またはOpusへ上げる条件:

- 同じ失敗を2回修正しても再現する。
- race、deadlock、memory corruptionの原因が局所化できない。
- llama.cppの改造範囲が3モジュール以上へ広がる。
- 数理モデルと実測が20%以上乖離し、原因仮説が複数残る。
- GREEN/YELLOW/REDの判定に研究上の判断が必要。

上位モデルへ丸ごと投げ直さず、失敗ログ、diff、最小再現、未解決質問を渡す。

---

## 16. Git・ログ・証拠管理

### 16.1 Branch / Commit

- `main`: 再現済みGREENまたは承認済みYELLOWのみ。
- `phase/*`: Phase単位。
- `experiment/*`: 一時実験。結果採用時は文書化して整理。
- 1 commit = 1つの検証可能な変更。
- commit messageへRun IDを含める。

### 16.2 Gitへ入れないもの

```gitignore
*.gguf
*.safetensors
*.bin
models/
ollama-blobs/
results/**/raw-model-data/
build/
```

### 16.3 Decision Record

重要判断は`docs/DECISIONS.md`へ以下の形式で残す。

```text
ID / 日付
状況
選択肢
決定
根拠
反証条件
関連Run ID
```

### 16.4 結果の改変防止

- raw metricsは追記専用。
- 集計スクリプトのcommitを記録。
- summaryにはraw metricsのSHA-256を記録。
- 手作業で数値を書き換えない。

---

## 17. Claude Codeへ渡す初回Mission

以下をそのままClaude Codeへ渡せる。

```text
Mission: ALSR PoC Phase 0-2（M5上の解析器・Chunk計画・帯域シミュレータ）

目的:
Adaptive Layer Streaming Runtimeの最初の実装として、実モデルを変更せず、
GGUF/Ollama blobの解析、Chunk計画、リングバッファ状態機械、帯域シミュレーション、
再現可能な証拠出力までを作る。

参照文書:
docs/ALSR_REQ_DESIGN.md

重要制約:
1. M5はCUDA/PCIe検証機ではない。CUDA性能を主張しない。
2. モデルファイルは読み取り専用。コピー、変換、削除をしない。
3. 平均供給帯域が平均消費帯域を下回る場合、有限バッファは必ず枯渇する。
   この条件をテストとレポートに含める。
4. 既存のOllama環境を変更しない。
5. モデル重みをGitへ入れない。
6. 実装とテストを同じMission内で完了する。
7. 未確認のテンソル形式を推測で処理しない。

技術:
- C++20 + CMake
- Python 3.11+は結果集計だけに使用可
- JSON/JSONL出力
- テストフレームワークは1つに固定

実装範囲:
A. alsr-inspect
   - 入力GGUFのmetadata/tensor一覧
   - name/type/shape/offset/bytes/alignment
   - model digest
   - manifest.json

B. alsr-plan
   - 64/128/256/512/1024MiB固定Chunk
   - tensor/quant-block/alignment境界を尊重
   - chunk_id/source_offset/bytes/checksum/tensors
   - plan.json

C. alsr-sim
   - producer/ready queue/ring buffer/consumer
   - bandwidth, latency, jitter, consumer rateを設定可能
   - buffer depth 2/3/4/8
   - fixed schedulerとadaptive scheduler
   - starvation、wait、queue depth、throughputをJSONL出力

D. テスト
   - 完全被覆、欠落なし、重複なし
   - 状態遷移
   - wrap-around
   - cancellation
   - 定常帯域不足では必ず枯渇すること
   - 供給帯域十分かつjitterがバッファ余裕内なら枯渇しないこと

E. 文書
   - README: build/run/test
   - docs/DECISIONS.md
   - docs/EXPERIMENTS.md
   - 既知の制約

完了条件:
- clean build
- 全テスト成功
- sample configで3つのRunを生成
- 各Runにconfig/environment/metrics/summary/report
- 同一seedでsummaryが再現
- git diffを自己レビューし、未完了・推測・TODOを列挙

実装しないもの:
- CUDA backend
- llama.cpp改造
- Ollama改造
- 405Bモデル取得
- 独自圧縮

最後に報告する内容:
- 変更ファイル
- 実行したコマンド
- テスト結果
- sample Run ID
- 判明した制約
- Phase 3へ進む前の未解決事項
```

---

## 18. Phase 3用Mission概要

M5 PhaseがGREENになり、RTX機が空いた後に実行する。

```text
Mission: ALSR PoC Phase 3（RTX 3060 CUDA Transfer Backend）

- 実機情報を自動採取
- RAM bandwidth benchmark
- Pageable/Pinned H2D benchmark
- Sync/Async、Stream 1-4、Chunk 16MiB-1GiB
- CUDA Event + wall clock
- copy/compute overlap ratio
- simulator backendと同じscheduler interface
- 10分stress test
- OS/Driver不安定化を避けるPinned Memory上限
- GREEN/YELLOW/RED判定レポート

実LLM統合はこのMissionに含めない。
```

---

## 19. 実装開始前チェックリスト

- [ ] ALSR専用リポジトリを作成した。
- [ ] 本書を`docs/ALSR_REQ_DESIGN.md`へ配置した。
- [ ] M5上の対象GGUF/Ollama blobを読み取り専用で特定した。
- [ ] 空きストレージを確認した。
- [ ] モデルdigestを保存した。
- [ ] Claude CodeでSonnet 5を選択・記録した。
- [ ] `*.gguf`、blob、resultsの大容量物を`.gitignore`へ追加した。
- [ ] Phase 0-2以外を初回Missionへ混ぜていない。
- [ ] RTX機上のNeutron Gate学習が終了するまでRTX作業を開始しない。

---

## 20. 未確定事項

以下は実測またはユーザー判断が必要であり、本書で捏造しない。

- RTX検証機のOS
- RTX 3060の正確なボードメーカー／型番
- 実機のPCIe current link speed / width
- 96GB Mixed DIMM構成の実効帯域
- CUDA Driver / Toolkit版
- 対象Ollama blobの実ファイルパスとdigest
- Gemma 4 12Bを実LLM経路で扱えるllama.cpp commit
- ユーザー環境に表示される「Opus 5」の正式CLI model ID
- 70B以降で使う具体的モデルと量子形式

---

## 21. 公式参照資料

1. Ollama, Gemma 4 model library  
   https://ollama.com/library/gemma4
2. MSI, MEG X570 UNIFY Specifications  
   https://www.msi.com/Motherboard/MEG-X570-UNIFY/Specification
3. AMD, Ryzen 7 5700X Specifications  
   https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-5000-series/amd-ryzen-7-5700x.html
4. NVIDIA, GeForce RTX 3060 announcement/specification  
   https://nvidianews.nvidia.com/news/nvidia-introduces-geforce-rtx-3060-next-generation-of-the-worlds-most-popular-gpu/
5. NVIDIA, CUDA Programming Guide — Asynchronous Execution  
   https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
6. NVIDIA, CUDA C++ Best Practices Guide  
   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
7. Meta, Llama 3.1 / 405B overview  
   https://ai.meta.com/blog/meta-llama-3-1/
8. OpenAI, gpt-oss model card  
   https://deploymentsafety.openai.com/gpt-oss/evaluation
9. Anthropic, Claude Sonnet 5  
   https://www.anthropic.com/news/claude-sonnet-5
10. Anthropic, Claude Fable 5 / Mythos 5  
    https://www.anthropic.com/news/claude-fable-5-mythos-5
11. Anthropic, Claude Opus 4.8  
    https://www.anthropic.com/news/claude-opus-4-8
12. Claude Code CLI reference  
    https://docs.anthropic.com/en/docs/claude-code/cli-usage

---

## 22. 最終結論

ALSRの核心は「データを細かく送れば速くなる」ことではなく、次の2点を分離して測ることである。

1. 転送と演算を重ね、要求駆動方式に存在する無駄な待機をどこまで消せるか。
2. 無駄を消した後も残る平均帯域不足によって、どこから先が物理的に成立しないか。

M5では制御系と証拠生成を完成させ、RTX 3060ではPCIeとCUDAの物理を測る。最初の実装対象はGemma 4 12B、段階的な研究ロードマップは31B、70B、120B、405B級とする。

結果がGREENでもREDでも、同じ条件で再現でき、式・ログ・実測値が一致することを最優先とする。
