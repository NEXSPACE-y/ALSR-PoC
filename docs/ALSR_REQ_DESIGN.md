# Adaptive Layer Streaming Runtime（ALSR）
## 要件定義書・基本設計書 — Requirements Rebaseline Rev.3.7

| 項目 | 内容 |
|---|---|
| 文書ID | `ALSR-REQ-DESIGN-REBASELINE-REV3` |
| 文書版 | 3.7 |
| 作成日 | 2026-08-03 JST |
| 最終更新 | 2026-08-04 JST — M5凍結・Windows／RTX移行決定 |
| 復元元 | 参照チャット全 57 ターン、現チャットのユーザー訂正、現行 Rev.1.1 / Rev.02 |
| 研究対象 | 限られた高速メモリ／VRAMで大規模量子化 LLM をどこまで実推論できるか |
| 前段検証機 | MacBook Air M5 / Unified Memory 16GB |
| 本検証機 | Ryzen 7 5700X / RAM 96GB / RTX 3060 12GB |
| 基準モデル | Gemma 4 12B（既存経路の基準。研究上限ではない） |
| 規模研究帯 | 16GB制約超過モデル → 20〜31B級 → 70B級 → 120B級 → 405B級 |
| 状態 | `APPROVED_REQUIREMENTS_BASELINE / M2_GREEN / M5_FROZEN / X0_NEXT` |

---

## 0. 文書の権威と変更規則

### 0.1 本書の目的

本書は、実装容易性を優先した過去の縮約を除去し、ユーザーが会話で明示した研究目的を正本として再構成する。
現行 `docs/ALSR_REQ_DESIGN.md` Rev.1.1 とチャット上の Rev.02 は、技術資料および差分監査対象として参照するが、本書の要求を縮小する根拠にはしない。

### 0.2 権威順位

要求の権威順位は次のとおり。

1. ユーザーの明示要求・訂正・禁止
2. ユーザーが選択肢として明示承認した範囲
3. ユーザーが内容を確認し明示承認した設計
4. AI の提案、推測、便宜上の段階分け

第4項は要求ではない。AIは、工数・計算資源・利用制限・実装難度を理由に第1〜3項を縮約してはならない。

### 0.3 変更統制

- 研究対象、成功条件、Gate順序、モデル規模範囲を変更する場合はユーザー裁定を要する。
- 実装担当は受入基準を変更しない。
- 未確定事項は `OPEN` のまま記録し、実装しやすい値で埋めない。
- Component、Simulator、Hardware、Real Inference、Research Conclusionの判定を混在させない。
- 本書の承認だけでは既存コードの適合性は証明されない。既存コードはP0以降の各Gateで本書との適合を確認するまで `UNVERIFIED` とする。

---

## 1. 研究目的

### 1.1 中核研究質問

> 限られた GPU VRAM または高速メモリ環境で、量子化された大規模 LLM の重みを実行順序に沿って事前解析・分割・準備し、有限バッファと先読み制御を用いて演算系へ継続供給することで、どのモデル規模・量子化・Context・速度まで実推論を成立させられるか。

### 1.2 研究成果

研究成果は単一モデルの成功ではなく、次の測定可能な成立領域である。

```text
モデル構造
× 実weight bytes
× 量子形式
× Context
× 高速メモリ容量
× Host memory容量
× Storage条件
× Chunk計画
× Buffer深度
× Backend
→ 正確性・安定性・速度・待機・メモリ使用量
```

成立しない境界、支配要因、撤退条件を証拠付きで確定することも正式な成果とする。

### 1.3 非目標

ALSRは次を事前には主張しない。

- PCIeやストレージの物理帯域を超えること
- RAMがVRAMと同じ速度になること
- M5の結果だけでCUDA性能を証明すること
- 1つのChunk設定が全モデルで最適であること
- 31Bが動けば70B・120B・405Bも自動的に動くこと
- Llama 3.1 405Bを含む405B級がRTX 3060で実用速度になること
- Simulatorの結果だけで実LLM推論が成立すること
- OS pagingで起動しただけの状態をALSR成立とみなすこと

---

## 2. ユーザー要求

| ID | 区分 | 要求 | 履歴根拠 |
|---|---|---|---|
| UR-001 | `USER_EXPLICIT` | RAM等の大容量層にあるモデルデータを、演算系が空にならないよう事前準備して継続供給すること | `2e08af4a-6b6a-4a0d-ba62-a2b55ce9cd58` |
| UR-002 | `USER_EXPLICIT` | 実行開始時に、モデル構造、変換要否、供給可能量、メモリ階層差を解析すること | `a1aca011-9d83-4ea9-9365-e4fd26549b0d` |
| UR-003 | `USER_EXPLICIT` | データをChunk化し、1 Chunkの供給中に次Chunkを準備済みにすること | `a1aca011-9d83-4ea9-9365-e4fd26549b0d` |
| UR-004 | `USER_EXPLICIT` | 高額・大容量GPUを前提にせず、少ないVRAMで大規模ローカルLLMが成立する範囲を測ること | `2f565a5c-5990-48b5-8e89-6b4d3c83b2f4` |
| UR-005 | `USER_EXPLICIT_CURRENT` | 12Bは基準点として扱い、研究対象を31B単体へ限定しないこと | 現チャットの明示訂正 |
| UR-006 | `USER_EXPLICIT` / `APPROVED_SCOPE` | 量子化Denseモデルを主対象とし、20〜31B、70B、120B、Llama 3.1 405Bを含む405B級へ段階的に拡張すること | `d9b15758-b452-4b9a-aa73-72b9d61b75c2` / `e325a9ce-ca7c-438b-92f7-9cf2e5705848` / `4934ca06-a891-4d77-8109-7b03ae92e8c3`（C＋1） |
| UR-007 | `USER_EXPLICIT_CURRENT` | M5でシステム構造の正常動作を実モデル・実推論で確認してからRTX用に組み替えること | 現チャットの明示訂正 |
| UR-008 | `USER_EXPLICIT_CURRENT` | M5ではPCIe固有性能を主目的にせず、実行構造、データ供給、正確性、安定性を確認すること | 現チャットの明示訂正 |
| UR-009 | `USER_EXPLICIT_CURRENT` | RTXでは同じコア構造を維持し、CUDA／PCIe／VRAM固有部分を接続して検証すること | 現チャットの明示訂正 |
| UR-010 | `CONTROL_DERIVED_APPROVED` | 実測、ユーザー観測、Simulation、導出、仮定、仕様値を混同しないこと | 過去事故から導出。Rev.3全体承認により採用 |
| UR-011 | `CONTROL_DERIVED_APPROVED` | 成立しない領域も定量的な研究成果として残し、成功へ見せかけないこと | 過去事故から導出。Rev.3全体承認により採用 |
| UR-012 | `USER_EXPLICIT_CURRENT` | AIは実装しやすい最小MVPへ要件を無断変更しないこと | 現チャットの明示確認 |
| UR-013 | `USER_EXPLICIT_CURRENT` | Gemma 4 31BをM5の最初のmemory-pressure実推論対象に含める。ただし31Bだけで研究を終了しないこと | 現チャットの明示訂正 |
| UR-014 | `USER_EXPLICIT_CURRENT` | M2完了をもってM5の高負荷検証を凍結し、以後の31B／規模検証はWindows RTX機へ移すこと。これはUR-013のモデル対象を維持し、実行platformだけを上書きする | 2026-08-04の明示指示 |

---

## 3. 検証環境

### 3.1 M5前段検証機

| 項目 | 内容 |
|---|---|
| 機種 | MacBook Air M5 |
| Unified Memory | 16GB |
| 基準Runtime | Ollama / Metal |
| 基準モデル | Gemma 4 12B |
| 役割 | M2までのコア構造と実LLM経路の成立確認（UR-014により以後凍結） |

M5には独立したRAM→PCIe→VRAM経路がない。このためPCIe、Pinned Memory、CUDA copy engine、CUDA Streamの性能は認定しない。
一方でM5をParser／Simulator専用機へ縮小してはならない。M5 Gateには、実モデルを使った実推論経路が必須である。

### 3.2 RTX本検証機

| 項目 | 構成 |
|---|---|
| CPU | AMD Ryzen 7 5700X |
| Motherboard | MSI MEG X570 UNIFY |
| RAM | DDR4-3200 CL22 16GB×2 + 32GB×2 = 96GB |
| Compute GPU | NVIDIA GeForce RTX 3060 12GB |
| Display GPU | GTX 1660 SUPER |
| 役割 | 分離VRAM環境でのCUDA、PCIe、実LLM、既存方式比較 |

RTX側のOS、Driver、CUDA、ボード型番、PCIe current generation／width、RAM実効帯域、H2D実効帯域は実機から取得する。

---

## 4. モデル規模戦略

### 4.1 モデル名と研究目的を分離する

モデル名は測定点であって研究目的ではない。特定モデルが取得不能、Runtime非対応、構造不適合の場合は、同等の規模・構造・量子形式へ置換し、研究質問を維持する。

| 段階 | 目的 | 合格の意味 |
|---|---|---|
| 12B級 | 通常経路、正確性、計測、回帰基準 | 基準が固定された |
| 16GB制約超過 | M5でALSR実行構造が必要となる最初の実証 | Simulatorではなく実推論が成立した |
| 20〜31B級 | 初期大規模測定帯 | 1つ以上の実モデルで成立条件を測定した |
| 70B級 | 96GB RAM内の大規模Dense候補 | RTX環境で実推論または明確な不成立証拠を得た |
| 120B級 | 量子化と構造差を含む拡張 | 測定モデルの適用限界を確認した |
| 405B級（Llama 3.1 405Bを含む） | 最終研究ターゲット | 実行または実測値に拘束された不成立境界を裁定した |

31Bの1回成功を研究完了としない。405Bを数式だけで処理して全体完了とも判定しない。
Gemma 4 31Bは当初M5 Gate M3の初回選定モデルだったが、UR-014によりWindows RTX側R3Bへ移設した。Runtime非対応・取得不能等で置換が必要な場合、AIが自動で別モデルへ変更せずユーザーGateへ戻す。

### 4.2 DenseとMoE

最初の中核対象は量子化Denseモデルとする。MoEはExpert選択、resident set、先読み可能性が異なるため、Denseの結果へ混在させず別Gateで扱う。

---

## 5. 最重要の物理条件

### 5.1 定義

| 記号 | 意味 |
|---|---|
| `W_model` | モデル実体の総byte数 |
| `W_resident` | 高速メモリ層へ保持できる重みbyte数 |
| `W_wire/token` | 1 tokenの処理で低速層から高速層へ実際に供給するbyte数 |
| `B_store` | StorageからHost／Unified Memoryへ準備できる実効帯域 |
| `B_prepare` | 読取、検証、変換、dequant準備を含む実効準備帯域 |
| `B_transfer` | Backendがconsumer側へ供給できる実効帯域。RTXではH2D、M5では同一意味へ無理に読み替えない |
| `B_supply` | 実行経路上で継続供給を制約する最小実効帯域 |
| `T_prepare(i)` | Chunk iの準備時間 |
| `T_transfer(i)` | Chunk iのBackend供給時間 |
| `T_compute(i)` | Chunk iに対応する実演算時間 |
| `C_chunk` | Chunk byte数 |
| `D` | 有限Ring Buffer深度 |
| `C_fast` | VRAMまたはUnified Memory上でALSRが利用できる高速作業領域 |

### 5.2 定常供給条件

目標生成速度を `r` token/s とすると、定常状態の必要条件は次である。

```text
B_supply >= W_wire/token × r
```

実際の `B_supply` は存在する経路に応じて実測する。

```text
B_supply = min(B_store, B_prepare, B_transfer, その他の実測ボトルネック)
```

M5には独立H2D経路がないため、`B_transfer`へ仮想PCIe値を代入してM5実測と呼ばない。
RTXではStorage／RAM／H2D／dequant／computeを分離し、最小値を別の段階の値で代用しない。

平均供給帯域が平均消費要求を下回る場合、有限Bufferは枯渇時刻を遅らせるだけで定常不足を解決しない。

### 5.3 Overlap条件

Chunk iの演算中に次Chunkを準備・供給するための基本条件は次である。

```text
T_prepare(i+1) + T_transfer(i+1)
<= T_compute(i) + measured_buffer_slack(i)
```

この式は設計仮説であり、実際の依存関係、同期、Runtime挙動、cache、jitterを実測して判定する。

### 5.4 Memory Budget

RTXの高速メモリ予算は最低限次を含む。

```text
VRAM_total
= runtime
+ KV cache
+ activation
+ CUDA workspace
+ dequant / temporary buffers
+ resident weights
+ ring buffers
+ safety margin
```

M5ではUnified MemoryをCPU用RAMとGPU用VRAMとして二重計上しない。最低限、OS、Runtime、model mapping、active working set、KV、activation、staging、compressed memory、swapの観測可能範囲を分離する。

`resident_bytes`、Chunk数、Ring深度はMemory Budgetを通過した組合せだけをRun可能とする。

### 5.5 Host Memoryを超えるモデル

`W_model`が物理Host／Unified Memoryを超える場合、モデル全体をRAM保持する前提は成立しない。
Storage-backed mmap、OS paging、明示I/O等のどの経路を使用したかを記録し、ALSR管理の供給とOS任せのpagingを同一方式として報告しない。

Windows RTX側のmemory-pressure Gateでは、モデルファイルが存在すること、起動すること、正確に継続生成できることを分離する。M5ではこのGateを実行しない。

### 5.6 PrefillとDecode

PrefillとDecodeはデータ再利用、並列度、KV増加、token当たり供給量が異なるため、別メトリクス・別判定とする。

```text
prompt_eval速度 + decode速度 = 推論速度
```

のような単純加算を行わない。time-to-first-token、prefill throughput、decode throughputを独立保存する。

### 5.7 圧縮転送

圧縮転送を導入する場合、帯域要求には実wire bytesを、演算要求には展開・dequant時間と追加bufferを含める。
圧縮率だけで速度改善を主張しない。圧縮転送はO-07が解決するまで中核必須要件へ昇格させない。

---

## 6. 基本アーキテクチャ

```text
┌─────────────────────────────┐
│ Model Store                 │
│ file / mmap / explicit I/O  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Model Inspector             │
│ tensor / type / shape / byte│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Execution Manifest          │
│ dependency / order / reuse  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Chunk Planner               │
│ integrity-aware boundaries  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Preparation / Ready Queue   │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Finite Ring + Backpressure  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Platform Backend            │
│ M5 backend / CUDA H2D       │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Real LLM Executor Adapter   │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Correctness + Telemetry     │
└─────────────────────────────┘
```

### 6.1 移植境界

次の契約はM5とRTXで共通とする。

- Chunk identityとbyte range
- Execution Manifestの意味
- Queue順序とexactly-once条件
- Ring slot状態遷移
- Backpressureとcancellation
- Scheduler input/output
- Telemetry schema
- Evidence Pack schema

M5→RTX移行時に交換できるのはPlatform Backend、Memory Allocator、Consumerのプラットフォーム依存部分である。

### 6.2 Simulatorの位置づけ

Simulatorは以下の検証にのみ使う。

- Queue／Ring状態機械
- 帯域不足時の枯渇条件
- jitterとbuffer余裕
- timeout／cancel／error注入
- Telemetryと数式の整合

Simulatorは、実モデルのトークン生成、出力正確性、実メモリ挙動、実GPU性能、既存Runtime比較を認定しない。

---

## 7. 機能要件

### FR-01 Model Resolution

- Runtime管理下のモデル実体を読み取り専用で解決する。
- digest、量子形式、ファイル集合、総byte数を保存する。
- 未対応形式を推測で処理しない。

### FR-02 Model Inspection

- metadata、tensor名、type、shape、offset、byte数、alignmentを取得する。
- file size、tensor ranges、padding、metadataの関係を説明可能にする。
- 同一モデルから決定論的なManifestを生成する。

### FR-03 Execution Manifest

- tensor名の並びだけで実行順を確定しない。
- 実Executor、Runtimeソース、計装または検証済み対応表に根拠を持つ。
- dependency、reuse、resident候補、prefill/decode差を表現する。

### FR-04 Chunk Planning

- tensor／quant block／alignment境界を破壊しない。
- Chunkごとにid、source range、bytes、checksum、依存、消費順を持つ。
- 固定Chunkと可変Chunkを同じManifestから生成できる。

### FR-05 Model Store / Preparation

- Storage、Host memory、staging memoryの所在を区別する。
- preload、mmap、明示I/O、OS pagingをTelemetry上で区別する。
- 次Chunkをconsumer要求前に準備可能にする。
- 準備失敗を欠落として隠さず、明示エラーにする。

### FR-06 Queue / Ring Buffer

- 状態は最低限 `EMPTY / PREPARING / READY / IN_USE / RETIRED / ERROR` を持つ。
- 欠落、重複、順序違反、上書きを検出する。
- finite bufferであり、平均供給不足を無限に隠せない。
- cancellation、timeout、shutdownを安全に扱う。

### FR-07 Scheduler

- 初期認定は固定方式で行う。
- 入力として準備時間、consumer時間、queue depth、buffer余量、直近分散を使用できる。
- 出力としてprefetch距離、Chunk選択、準備開始時刻を決定する。
- Adaptive制御は固定方式の実測ボトルネックを取得後、独立Gateで認定する。

### FR-08 Platform Backend

- M5 backendとCUDA backendが同一のChunk／Scheduler契約を実装する。
- Backend固有の時間・メモリ・同期情報を共通Telemetryへ写像する。
- Backendが未実装の場合、Simulator結果を実Backend結果として返さない。

### FR-09 Real LLM Executor Adapter

- 実際のprompt処理とtoken生成に、ALSRが供給したデータを使用する。
- Executorが独自にモデル全体を別経路で読み込んだ場合はALSR Runを無効とする。
- 基準Runtimeとの出力比較を可能にする。
- PrefillとDecodeを分離計測する。

### FR-10 Telemetry

全メトリクスは次を持つ。

```text
name
value
unit
provenance
platform
model_digest
run_id
scope
evidence_path
```

Provenanceは最低限次を区別する。

```text
USER_OBSERVED
MEASURED_M5
MEASURED_RTX
EMULATED
DERIVED
ASSUMED
DECLARED_SPEC
UNKNOWN
```

### FR-11 Evidence Pack

各Runは最低限次を保存する。

```text
command.txt
config.resolved.json
environment.json
model_manifest.json
execution_manifest.json
chunk_plan.json
metrics.jsonl
summary.json
correctness.json
artifacts.sha256
git_status.txt
git_diff.patch
```

Evidence未保存のRunは性能・成立性の正式Claimに使用しない。

---

## 8. 非機能要件

### NFR-01 正確性

- Chunk checksum一致率100%
- 欠落0、重複0、順序違反0
- 実LLM出力を基準経路と比較する
- 正確性基準を満たさないRunは性能比較から除外する

### NFR-02 再現性

- 同一model digest、config、seed、prompt、Runtime versionを保存する
- 反復Runを個別保存し、代表値だけに潰さない
- cold／warm、prefill／decodeを分離する

### NFR-03 観測可能性

- wall timeと仮想時間を混同しない
- preparation、I/O、queue wait、backend wait、computeを分離する
- resident、staged、mapped、paged、swappedの可能な範囲を区別する
- M5で取得不能な情報を推測値で補わない

### NFR-04 安全性

- モデルファイルは読み取り専用
- メモリ予算超過は起動前または明示エラーで停止
- hang時にEvidenceを残して終了可能
- model download、変換、削除はMissionの明示範囲外では行わない

### NFR-05 移植性

- コアロジックをMetal／CUDA APIから分離する
- Platform Backend以外にプラットフォーム固有分岐を拡散させない
- 同じテストベクトルをM5／Simulator／RTXで再利用可能にする

### NFR-06 テスト階層

テストは次の層を分離し、下位層のPASSを上位層の証拠に読み替えない。

| 層 | 必須内容 | 認定しないもの |
|---|---|---|
| Unit | range、alignment、checksum、state transition、budget arithmetic | Runtime統合、実推論 |
| Component | Inspector、Planner、Queue、Ring、Telemetry単体 | end-to-end正確性 |
| Simulator Integration | jitter、帯域不足、cancel、timeout、fault injection | 実I/O、実GPU、実token/s |
| M5 Real Integration | 実モデル、実Executor、実token生成、memory pressure | CUDA／PCIe性能 |
| RTX Synthetic | 実H2D、consumer kernel、overlap、VRAM pressure | 実LLM正確性 |
| RTX Real Inference | 実モデル、正確性、既存方式比較、長時間運転 | 未試験規模への一般化 |

テストソースの存在は実行結果ではない。実行ログとEvidence Packが無いテストは `NOT_EXECUTED` とする。

### NFR-07 Stress / Fault Injection

- Queue depth最小／最大
- Ring wrap-around
- 準備遅延、帯域jitter、一時停止
- consumer停止、producer停止
- I/O短読、checksum不一致、未対応tensor
- cancellation、timeout、graceful shutdown
- memory budget境界直下／直上
- 長時間運転とresource leak

障害注入に対して、silently continue、偽の成功summary、欠落Chunkの補間を禁止する。

### NFR-08 Evidenceの不変性

- Run成果物を事後に上書きせず、新しいRun IDで再実行する。
- 文書訂正は元のClaim／値を消去せず、amendmentと理由を記録する。
- hash一致が同一Pack内で閉じている場合、外部真正性の証明とは呼ばない。
- 実行者の自己検証と独立監査を区別する。
- 実装担当の状態は `COMPLETED_NOT_AUDITED` までとし、GREENを自己発行しない。

### NFR-09 Scope Control

- 1 Gate = 1 Missionを原則とする。
- Mission外のモデル取得、Runtime改造、Adaptive実装、規模拡張を行わない。
- 便利機能や追加最適化を無断追加しない。
- 要件の削除・延期・モデル固定・受入緩和はscope逸脱として停止する。
- 実装中に別の必要作業を発見した場合は、実装せずOPEN事項として報告する。

---

## 9. Gate構成

### Gate RQ0 — Requirements Rebaseline

目的: 原要求を復元し、無断縮約を排除する。

GREEN条件:

- ユーザー要求にIDが付いている
- 現行文書との差分が列挙されている
- 未決事項を推測で埋めていない
- ユーザーが本書を正本として承認した

### Gate P0 — Prior Art / Integration Feasibility

目的: 既存Runtimeの実データ経路と統合点を確認する。

対象:

- llama.cpp／Ollamaのmodel loading、mmap、offload、Metal、CUDA経路
- tensor execution orderを裏付ける箇所
- weight供給を差し替え可能な境界
- 既存prefetch／async copy／buffer管理
- 出力正確性比較方法

GREEN条件:

- ALSRが置換・追加する境界をソース根拠付きで特定
- M5実LLM Adapterの実装可能性を確認
- 既存方式との差分仮説と撤退条件を固定

### Gate M0 — M5 Baseline

目的: 12B基準経路と環境を固定する。

GREEN条件:

- 同一条件で5回以上完走
- 生レスポンス保存
- model digest、Runtime、OS、Context、prompt、生成長、cold／warmを保存
- 正確性と中央値を算出

### Gate M1 — Inspector / Manifest / Chunk Integrity

目的: 実モデルから実行可能なデータ計画を生成する。

GREEN条件:

- byte rangeの重複／欠落が説明可能
- Manifestが決定論的
- Execution orderの根拠が存在
- Chunk完全性100%

### Gate M2 — M5 Real Executor Integration

目的: 12B基準モデルでALSR経路を実LLMへ接続する。

GREEN条件:

- Simulatorではなく実LLM ExecutorがALSR供給データを使用
- Model Storeが要求rangeを明示的に読み、Preparation、Ready Queue、有限Ring、Backend、Executorの順に実データが通る
- Runtimeの既存mmap tensor pointerを観測・検査するだけのhookや、既存backend copyの追認をALSR供給と数えない
- promptからtoken生成まで完走
- 正確性基準を満たす
- Chunk／Queue／Buffer traceが実行と対応
- Executorの別経路全量ロードがないことを確認

### Gate M3 — Memory-Pressure Real Inference（旧M5配置、RTXへ移設）

状態: `RELOCATED_TO_R3_BY_UR-014 / NOT_RUN`

旧目的: 16GB制約を実質的に超える実モデルでALSR構造を検証する。UR-014後はこの受入意図をWindows RTX側R3Bで実施し、M5では実行しない。

初回対象: Gemma 4 31B。変更にはユーザー裁定を要する。

GREEN条件:

- 少なくとも1つの実モデルでtoken生成まで完走
- 実model bytes、量子形式、Context、最大メモリ、I/O、paging／swap観測を保存
- OOM、deadlock、integrity failureなし
- ALSR経路とOS任せのpaging baselineを区別
- 反復可能、または不成立原因を証拠付きで確定

### Gate M4 — Scale Envelope（旧M5配置、RTXへ移設）

状態: `RELOCATED_TO_R3_AND_S0_BY_UR-014 / NOT_RUN`

旧目的: M5上の成立境界をモデル規模・量子化・Contextで測る。UR-014後はWindows RTX側R3B／S0で測定し、M5では実行しない。

成果物:

- operational frontier
- correctness frontier
- sustained-throughput frontier
- failure taxonomy

単一31B Runだけで完了しない。

### Gate X0 — Cross-platform Core Freeze

目的: M5で確認したコア契約をRTX移植前に固定する。

GREEN条件:

- 共通interfaceとschema固定
- M5 Evidenceと同じ意味でRTX Evidenceを生成可能
- Backend交換以外の無断再設計なし

### Gate R0 — RTX Environment

目的: 実機構成を取得し仮定値を排除する。

### Gate R1 — CUDA Transfer Microbenchmark

目的: Pageable／Pinned、Sync／Async、Stream、Chunk、copy／computeを実測する。

### Gate R2 — RTX Synthetic Streaming Runtime

目的: 実ChunkをCUDA Backendへ供給し、consumer kernelで順序・完全性・overlapを確認する。

### Gate R3 — RTX Real LLM Integration

目的: VRAM超過実モデルをALSR経路で実推論し、既存方式と比較する。

### Gate S0 — Scale Research

目的: 20〜31B、70B、120B、Llama 3.1 405Bを含む405B級へ段階的に拡張し、成立領域または不成立境界を確定する。

各Gateは前Gateの成果物を消費するが、前GateのGREENを別種のGREENへ読み替えない。

---

## 10. Run判定体系

| 判定 | 意味 |
|---|---|
| `COMPONENT_VALID` | 単一構成要素の正当性を確認 |
| `SIMULATOR_VALID` | 仮想条件に対する論理挙動を確認 |
| `REAL_INFERENCE_VALID` | 実モデル・実Executorで正確なtoken生成を確認 |
| `SUSTAINED_VALID` | 指定時間／生成長で安定運転を確認 |
| `RESEARCH_BOUNDARY_FOUND` | 成立または不成立境界をEvidence付きで確定 |
| `PRODUCT_CANDIDATE` | 正確性・速度・安定性・運用性を満たす候補 |

下位判定を上位判定として報告しない。

### 10.1 M5 Real Inference必須証拠

- 実際に使用したmodel digestとweight bytes
- prompt、token列、生Runtime出力
- ALSR path trace
- Chunk consumption trace
- memory / I/O telemetry
- correctness comparison
- failure／warning全文
- Run全体のSHA-256一覧

### 10.2 性能Claim

全性能Claimは次の鎖を持つ。

```text
Value
  ↓
Provenance
  ↓
Evidence
  ↓
Scope
  ↓
Gate
```

1つでも欠けるClaimは正式採用しない。

---

## 11. 比較対象

ALSRは最低限次と比較する。

- 通常Runtime／通常Ollama経路
- OS管理mmap／paging経路
- llama.cpp CPU／GPU offload
- 固定offload
- ALSR固定Scheduler
- ALSR Adaptive Scheduler（独立Gate通過後）

比較条件はmodel digest、量子形式、Context、prompt、生成長、正確性基準を揃える。

---

## 12. 失敗と撤退条件

次の場合は成功へ見せかけず、研究結果として記録する。

- 実LLM Executorへ安全なweight供給境界を作れない
- 既存Runtimeが同等以上のprefetch／overlapを既に実装している
- ALSR経路の正確性を維持できない
- 平均供給帯域不足が支配し、Schedulerで改善できない
- metadata／依存関係により安全な先読みができない
- メモリ節約量に対して速度低下が実用上大きすぎる
- 既存方式に対する改善が測定誤差内

撤退はプロジェクト失敗ではなく、適用領域の確定である。

---

## 13. Requirement Traceability

| User Requirement | Design | Gate | Required Evidence |
|---|---|---|---|
| UR-001 / 継続供給 | FR-05〜FR-09 | M2 / M3 / R2 / R3 | queue、buffer、consumer trace |
| UR-002 / 事前解析 | FR-01〜FR-03 | P0 / M1 | model・execution manifest |
| UR-003 / Chunk準備 | FR-04〜FR-07 | M1〜M3 / R2 | chunk plan、state transitions |
| UR-004 / 少ないVRAM | §3 / §4 | R3 / S0 | memory budget、scale envelope |
| UR-005 / 31B限定禁止 | §4.1 | S0 | 複数規模のRunまたは境界表 |
| UR-006 / Llama 3.1 405Bを含む405B級まで | §4 / S0 | S0 | 実測拘束されたscale report |
| UR-007 / M5先行実動 | FR-09 | M2（M3はUR-014で移設） | real inference Evidence Pack |
| UR-008 / PCIe非依存 | §3.1 / §6.2 | M0〜M2 | MEASURED_M5とscope |
| UR-009 / RTX組替え | §6.1 | X0 / R0〜R3 | interface conformance |
| UR-010 / Provenance | FR-10 | 全Gate | metric provenance |
| UR-011 / 不成立も成果 | §12 | M4 / S0 | failure taxonomy |
| UR-012 / 縮約禁止 | §0 | RQ0 | requirements diff、承認記録 |
| UR-013 / Gemma 4 31B実推論 | §4.1 / FR-09 | R3（UR-014によりM3から移設） | real inference Evidence Pack |
| UR-014 / M5凍結・RTX移行 | §3 / §9 | X0 / R0〜R3 | freeze checkpoint、Windows環境証拠、RTX Evidence Pack |

---

## 14. 現行Rev.1.1／Rev.02からの主要是正

| 旧状態 | Rev.3正本での是正 |
|---|---|
| M5 Phaseを解析・Simulatorで完了可能 | M5 Real Executor IntegrationとMemory-Pressure Real Inferenceを必須化 |
| RTXより後に実LLM接続 | M5で実LLM接続後にCross-platform Coreを固定しRTXへ移行 |
| 31BをVRAM超過代表として実質固定 | モデル規模は測定点。RTX scale envelopeとScale Researchを追加 |
| 70B／120Bを後段へ延期 | 研究範囲に維持し、Gate条件で段階化 |
| 405Bは原則外挿 | 実測拘束された裁定を必須化。実ファイル試験の可否はユーザーGate |
| SSD→RAM→VRAMを初期PoC外 | Model Storeを抽象化し、M2でI/O経路を実証。VRAM経路はRTX R2／R3で観測 |
| Simulator GREENと機構PoC GREENが混在 | 判定体系をComponent／Simulator／Real Inference／Sustained／Researchへ分離 |
| 12Bの正常推論を中心化 | 12Bは基準・回帰のみ。制約超過実モデルをM5 Gateへ追加 |

---

## 15. OPEN事項とP0決定

| ID | 事項 | 状態 | 決定Gate |
|---|---|---|---|
| O-01 | M5でOS paging／swapを基準方式として許容する範囲 | `DECIDED_P0` | P0 |
| O-02 | M5で明示的Storage I/O制御まで実装するか | `DECIDED_P0` | P0 / M1 |
| O-03 | M5 acceptanceの生成長・反復数・最大時間 | `DECIDED_P0` | P0 |
| O-04 | 出力正当性の主基準（logit／perplexity／token列） | `DECIDED_P0` | P0 |
| O-05 | 405B実ファイル取得・実Runの条件 | `OPEN` | S0前ユーザーGate |
| O-06 | MoE研究の開始条件 | `OPEN` | Dense scale結果後 |
| O-07 | 圧縮転送／GPU展開をALSR本体へ含めるか | `OPEN` | R3後 |

OPEN事項は未実装扱いではない。決定Gateより前にAIが値や方式を固定してはならない。

### 15.1 P0ユーザー決定記録 — 2026-08-03 JST

ユーザーは、現行状態 `Real inference = False` を `Real inference = True` にすることを前提として、P0提案のO-01〜O-04をすべて承認した。この前提は受入緩和ではなく、M2の達成目的そのものである。Simulator完走、もっともらしい出力、Ollamaの別経路推論を `True` と数えてはならない。

#### O-01 — OS paging／swap

- 通常OS mmap／pagingを必須比較baselineとして許可する。
- paging／swapの成功だけをALSR Schedulerがweightを供給した証拠にしない。
- page fault（取得可能な範囲）、compressed memory、swap、Storage I/O、物理memory pressureを分離記録する。

#### O-02 — 明示的Storage I/O

- 通常mmap／pagingは比較baselineとして許可するが、それだけをALSR供給経路として認定しない。
- M2から明示的range-I/O ALSR modeを必須とする。Model Storeが取得したbytesをPreparation、Ready Queue、有限Ring経由でbackend copyの実sourceにし、既存Runtime sourceへのfallbackは0でなければならない。
- 当初M3で予定したGemma 4 31Bへの適用は、UR-014によりRTX R3Bへ移設した。R3BでもM2と同じ明示的range-I/O構造を維持し、OS mmap／paging controlと比較する。アーキテクチャ実装をR3Bへ先送りしない。
- mmapと明示I/Oのどちらが速いかは事前決定せず実測する。

#### O-03 — M5 acceptance（UR-014前の決定記録）

M0／M2の条件は実施済み。M3行は未実行のままRTX R3Bへ移設されており、M5で再開しない。

| Gate | new tokens | repeats | 最大wall time | 目的 |
|---|---:|---:|---:|---|
| M0 通常12B | 128 | cold 5 + warm 5 | 30分／Run | baselineと分散 |
| M2 ALSR 12B | 128 | 5 | 30分／Run | 正確性と実統合 |
| M3 Gemma 4 31B | 128 | 3 | 60分／Run | `NOT_RUN / RELOCATED_TO_R3B` |

timeoutはfailure／boundaryとして保存し、結果を見た後に生成長を短縮して合格へ変更しない。この原則は移設先R3B／S0でも維持する。

#### O-04 — 出力正当性

- 同一model digest、tokenizer、prompt、Context、seed、決定論的greedy samplingを比較条件とする。
- acceptance生成長についてtoken ID列の完全一致を主基準とする。
- 各stepのtop logitsを診断用に保存する。
- logit数値toleranceはALSR比較前にbaseline自体の再現性から固定し、結果を見て変更しない。
- 不一致Runを性能Claimから除外する。
- perplexityは後続のscale／regression副基準とし、M2の唯一の正当性基準にはしない。

### 15.2 M0 baseline固定記録 — 2026-08-03 JST

`ALSR-M0-M5-BASELINE-01` は、M5 16GiB上の通常Ollama `gemma4:12b`経路でcold 5回、warm 5回を完走し、Gate M0をGREENとした。

- 10/10 RunでHTTP 200、128 token生成、選択token text／byte列の完全一致を確認した。
- Ollama APIの`response.context`にpromptと生成suffixの数値token IDが含まれていたため、M0の保存済み応答から72 prompt token＋128 generated tokenを分離し、M2比較baselineとして固定した。token IDを本文から逆推定していない。
- cold／warm間のselected logprob最大絶対差は`0.009411484003067017`、同一top-20候補の最大絶対差は`0.03801250457763672`だった。
- 後続ALSR比較の診断用logprob絶対toleranceを、結果を見る前のbaseline値として`0.05`に固定する。主基準はtoken列完全一致であり、toleranceはtoken不一致を許容しない。
- coldはOllama process上のunload状態であり、OS page-cache coldではない。
- 最初のcold Runでsystem-wide swap-out約4.15GBを観測した。これはOllama単独への帰属でもALSR成功証拠でもなく、M2および移設先R3BでOS pagingとALSR制御供給を分離する根拠である。
- `ordinary Ollama real inference = true`を確認したが、`ALSR Real Executor inference = false`は未変更である。後者をtrueにするGateはM2である。

証拠正本は `results/m0-gemma4-12b-ordinary-20260803T080813Z/` とする。

### 15.3 M1実graph Manifest固定記録 — 2026-08-03 JST

`ALSR-M1-REAL-GRAPH-MANIFEST-01` は、commit-pinned llama.cppの実scheduler graph callbackとGemma 4 12B実modelを用いて、prefill／decodeの実weight需要順と完全なGGUF byte-range計画を固定し、Gate M1をGREENとした。

- Runtimeはllama.cpp tag `b10091`、commit `b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45`へ固定した。
- model SHA-256は`1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606`、実weight bytesは`7,365,558,464`である。
- 同一promptから2回取得したschema v2 traceはSHA-256 `ba25e7cf7482df7c1892c7ea6d08417b7b6dc1b7ea22be13e3a2081cba9f60ce`でバイト一致した。
- prefillは61 input tokens／1,970 graph nodes、graph signature `393328c730a6beca759a2dcebe6c91f8d68049872ac2607900850a0618791886`、decodeは1 input token／1,970 graph nodes、graph signature `652756ee7ab3140c28778097d0f05d719f0d4e066bfac691a298f33e2bd0a9c3`である。
- GGUF source tensorは667件、Runtime tensor instanceは668件である。同名の`token_embd.weight` 2実体を別destinationとして保持し、両phaseで両実体の使用を確認した。
- storage planは718 Chunk、GGUF data領域`7,365,559,808` byteを、tensor `7,365,558,464` byte＋padding `1,344` byteとして欠落・重複なく説明する。使用Runtime実体668/668をGGUF tensor 667/667へ写像し、unmapped useは0である。
- 全ChunkにSHA-256とCRC32Cを付与した最終Manifestは2回バイト一致し、SHA-256は`c52a0eb47255bfc7f8ba21da99d79e10b57379f6764e5757d962078b45e90f48`、structure SHA-256は`3544cf50fb5e67240387cbad1f90ecb13949761337fd38ae0c8fd616091b40fa`である。
- schema v1の暫定Run 1〜4は、同名の別Runtime実体を名前だけで縮約する欠陥を検出した過程として保存する。M1受入判定からは除外し、schema v2のRun 5／6のみを正式証拠とする。
- default buildとM1 pinned buildの双方で107/107 testを完走した。
- M1は実graph計装とManifest生成の認定であり、ALSR供給経路による推論の認定ではない。`ALSR Real Executor inference = false`は維持し、これをtrueへ変更できるのはGate M2だけである。

証拠正本は `results/m1-real-graph-manifest-20260803T082724Z/` とする。

### 15.4 M2誤判定撤回・修復記録 — 2026-08-04 JST

2026-08-03 JSTに発行した`ALSR-M2-M5-REAL-EXECUTOR-01`のGREEN判定と、そこから導いた`ALSR Real Executor inference = true`を撤回する。保存済みRunのtoken一致やwall timeを否定するものではないが、当該RunはM2が要求する実データ供給経路を実装していなかった。

撤回の直接根拠:

- 全正式Runの`GRAPH_TRACE.json`は`callback_requests_tensor_data: false`を記録している。hookはRuntimeが既にmmapしたtensor dataを観測・照合しただけで、Model Storeへbytesを要求していない。
- backend copyの実sourceは既存Runtime tensor pointerのままであり、Preparation、Ready Queue、有限Ringから供給されたbytesではなかった。
- 旧`SupplyController`が記録したlease、checksum、copy eventは既存copyの追認であり、ALSRがそのcopy sourceを提供した証拠ではない。
- したがって旧Evidence Packが証明する範囲は、pinned llama.cppへのgraph／copy instrumentationとM0出力一致を確認した`INTEGRATION_SPIKE_ONLY`までである。Gate M2のGREEN条件は満たさない。

旧Evidence Pack `results/m2-gemma4-12b-real-executor-20260803T090123Z/` は失敗原因を含む履歴証拠として変更・削除せず保持する。同配下の`M2_RETRACTION.json`を現行判定とし、旧`M2_ACCEPTANCE_SUMMARY.json`および`M2_EVIDENCE_INDEX.json`のGREENは失効済みの自己判定として扱う。

修復版は、HostStoreによる明示range read、checksum Preparation、Ready Queue、固定Scheduler、有限Ring、backendのprepared range request／commitを実装し、Runtime source fallbackを禁止する。短い開発Runの成功をM2 GREENへ昇格させず、128 new tokens × 5 Run、最大1,800秒／Runの正式条件を変更せず再実施する。

### 15.5 M2 Real Supply修復・再認定 — 2026-08-04 JST

`ALSR-M2-M5-REAL-SUPPLY-REPAIR-01`は、旧誤判定を修復し、Gate M2を新EvidenceでGREENへ再認定した。

- HostStoreの明示range read、checksum Preparation、Ready Queue、固定Scheduler、2 slot有限Ring、prepared rangeを実sourceとするMetal backend copy、Executorを接続した。
- Runtime source fallbackを禁止し、5 Run合計fallback 0を確認した。
- 128 new tokens × 5独立Runは、M0 generated token ID列128/128とUTF-8本文が全件完全一致した。
- 各Runは86,912 prepared range、942,791,483,392 byteをModel Storeから読み、同量をbackend weight copyへ供給した。5 Run合計は434,560 range、4,713,957,416,960 byteである。
- 2 slot Ringの3,041,920状態遷移をmutex内の単調sequenceから再構成し、全lifecycle完結、違法遷移0を確認した。
- wall timeは最小307.392秒、中央値324.545秒、最大339.959秒で、5/5が1,800秒未満だった。
- read-only verifierはsummaryの自己申告判定を信用せず、SHA-256、出力、byte会計、bind／commit、Ring sequenceを生Evidenceから導出し、`M2_ACCEPTANCE_VERIFIED: GREEN`を返した。

証拠正本は`results/m2-gemma4-12b-real-supply-repair-20260804T002432Z/`とする。旧Packとsequence導入前の診断Packは正式受入から除外し、削除せず保持する。

この再認定により、12B fixed-path scopeでは`ALSR Real Executor inference = true`である。Gemma 4 31B以上のmemory-pressure実推論、Adaptive効果、性能優位、RTX／CUDAは未証明である。

---

## 16. 次の正式Mission

`ALSR-RQ0-REQ-REBASELINE-01` は2026-08-03 JSTのユーザー承認により完了した。
`ALSR-P0-RUNTIME-INTEGRATION-AUDIT-01` は一次ソース監査とO-01〜O-04のユーザー承認により完了した。
`ALSR-M0-M5-BASELINE-01` は通常Ollama 12B baselineのcold 5 + warm 5実測とEvidence固定により完了した。
`ALSR-M1-REAL-GRAPH-MANIFEST-01` は実graph trace、完全なbyte-range計画、決定論的Manifest、Evidence固定により完了した。
旧`ALSR-M2-M5-REAL-EXECUTOR-01`のGREEN判定は撤回し、`ALSR-M2-M5-REAL-SUPPLY-REPAIR-01`の新EvidenceでM2を再認定した。
ユーザー指示UR-014によりM5はM2到達点で凍結し、M3／M4の高負荷検証をRTX側へ移設した。
次の正式Missionは `ALSR-X0-CORE-FREEZE-WINDOWS-HANDOFF-01` とする。

```text
目的:
M2で実証したModel Store、Preparation、Ready Queue、Scheduler、有限Ring、
prepared range request／commit、Evidence schemaをGit checkpointとして固定し、
Windows RTX機でBackendだけをCUDAへ交換して継続できる状態にする。

許可:
- source、canonical llama.cpp patch、要件・設計、M2小型判定成果物のcommit
- Windows／RTX引継ぎ手順、未決事項、禁止事項、Gate順序の文書化
- portable git bundleの作成
- Windows機で必要な環境情報の項目定義

禁止:
- M5でmodel load、実推論、帯域bench、sustained I/Oを再実行すること
- model weight、Ollama blob、1.7GB生traceを通常Git objectへ入れること
- CUDA移植を理由にcore契約や受入条件を無断変更すること
- ordinary Runtime copyをALSR供給として報告すること
- R0環境固定前にCUDA性能値を仮定すること

成果物:
- Git checkpoint commit／tag
- Windows／RTX handoff文書
- M2小型判定checkpointと生Evidenceの所在・hash
- Windows側の次Mission `ALSR-R0-RTX-ENVIRONMENT-01` の開始条件
```

---

## 17. 状態

本書は2026-08-03 JSTにユーザー承認され、`APPROVED_REQUIREMENTS_BASELINE` として正本化された。
Gate P0は2026-08-03 JSTに `GREEN` で完了し、O-01〜O-04はユーザー承認により決定済みである。
Gate M0は2026-08-03 JSTに通常Ollama 12B baselineの実測・証拠固定により `GREEN` で完了した。
Gate M1は2026-08-03 JSTに実graph由来ManifestとChunk完全性の実測・証拠固定により `GREEN` で完了した。
Gate M2の2026-08-03 JST付GREENは2026-08-04 JSTに撤回した後、明示range-I/O／finite Ringを実装した新Evidenceの128-token実推論5/5により同日 `GREEN` へ再認定した。12B fixed-path scopeで`ALSR Real Executor inference = true`である。
2026-08-04 JSTのUR-014によりM5は凍結した。M3／M4は未実行であり、GREENへ読み替えず、31B memory-pressureとscale envelopeをR3／S0へ移設する。
現在の次GateはX0 Cross-platform Core Freeze、その後はR0 RTX Environmentである。R0前のCUDA性能測定、R3前の実LLM統合、Adaptive実装、S0 scale sweepは未許可である。

旧Rev.1.1は `docs/archive/ALSR_REQ_DESIGN_REV1_1.md` に内容を変更せず保存する。
P0決定前のRev.3.0は `docs/archive/ALSR_REQ_DESIGN_REV3_0.md` に内容を変更せず保存する。
M0決定前のRev.3.1は `docs/archive/ALSR_REQ_DESIGN_REV3_1.md` に内容を変更せず保存する。
M1決定前のRev.3.2は `docs/archive/ALSR_REQ_DESIGN_REV3_2.md` に内容を変更せず保存する。
M2決定前のRev.3.3は `docs/archive/ALSR_REQ_DESIGN_REV3_3.md` に内容を変更せず保存する。
Recovery成果物はRev.1.1時点の履歴監査資料として扱い、Rev.3への適合証拠へ自動昇格させない。
