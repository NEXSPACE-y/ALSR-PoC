# 既知の制約と、まだ言えないこと

更新: 2026-08-04 JST
状態: `M2_GREEN / M5_FROZEN / WINDOWS_RTX_NEXT`

本書はPhase 0-2時点の旧記録から始まったが、A0監査とM2再実装により主張可能範囲が
変更された。§3以降の旧測定値・simulation値は設計履歴として残すが、現行の性能証拠や
RTX予測として引用してはならない。Provenanceの正本は`docs/recovery/`、現在の実装証拠は
`docs/checkpoints/M2_M5_FREEZE_20260804/`、次作業は`docs/WINDOWS_RTX_HANDOFF.md`に従う。

---

## 1. この実装が主張してよいこと

- Gemma 4 12Bを使ったM2正式Packで、128 token × 5 Runの実推論が完了した。
- Model Storeの明示range readからPreparation、Queue、固定Scheduler、有限Ring、Backend、
  llama.cpp Executorまで、設計した供給経路を通過した。
- 正式5 Runではprepared-range欠落時のfallbackが0件だった。
- M2 verifierが正式Packを受理し、撤回済み旧Packを拒否する。
- GGUF解析、Chunk計画、Ring状態機械、供給経路に対応するsourceと軽量testが存在する。

ここでの「M2 GREEN」は構造成立の判定であり、速度優位、VRAM超過、CUDA移植、
scale成立の判定ではない。

## 2. この実装が主張してはいけないこと

- **CUDA / PCIe の性能**。M5 には独立 VRAM への PCIe H2D 経路が無い。
  simulator backend の `measured_on_real_hardware` は常に `false` を返す。
- **Pinned Memory の効果**。M5 では該当経路が存在しない。
- **ALSR が既存方式より速いこと**。設計書 §10 の比較対象 B0〜B5 のうち、
  公平なRTX比較はまだ1つも完了していない。
- **VRAM超過modelが動くこと**。M2の12BはM5構造確認用であり、31B、70B、120B、
  405B級は未検証。
- **CUDA版が機能すること**。Windows I/O、pinned host memory、device Ring、copy／compute
  stream、CUDA event lifetimeは未実装または未検証。
- **出力品質の一般保証**。M2でtoken生成は確認したが、direct baselineとのlogit差、
  perplexity差、長文品質は未認定。
- **M5で追加の高負荷検証を行ってよいこと**。M5は明示的に凍結済みである。

---

## 3. 旧Phase 0記録（現行性能証拠として引用禁止）

以下は当時の記録を追跡可能にするため残す。数値のProvenance、Claim判定、Evidenceの
有無はA0監査成果物で確認し、`MEASURED`、`DERIVED`、`ASSUMED`、`EMULATED`を混同しない。

### 3.1 単一コアの CRC32C が PCIe 4.0 より遅い

`alsr-bw` の M5 実測（中央値、512 MiB バッファ、7 回）:

| 指標 | 実測 [GB/s] |
|---|---|
| RAM memcpy | 55.71 |
| RAM 逐次読み出し | 78.73 |
| **CRC32C（ARMv8 CRC32 命令）** | **11.77** |
| ファイル読み出し (mmap, warm) | 56.39 |
| ファイル読み出し (pread, warm) | 17.06 |

PCIe 4.0 x16 の実効 H2D は 20〜25 GB/s 級である。**単一コアの CRC32C
11.77 GB/s はこれより遅い**。したがって:

- 全 Chunk のホスト側 checksum 検証を線速で回すことはできない。
- Phase 3 以降では (a) 検証を別スレッドへ逃がす、(b) サンプリング検証にする、
  (c) GPU 側で検証する、のいずれかが要る。
- 現在の PoC は checksum を**正確性ゲート専用**として使っており、性能 Run では
  `verify_checksums` を落とせるようにしてある。この切り替えを忘れると
  「ALSR が遅い」という誤った結論が出る。

### 3.2 明示 read（pread）は PCIe 4.0 に対して余裕が薄い

`pread` 17.06 GB/s は PCIe 4.0 の 20〜25 GB/s に対して余裕がない。
設計書 §2.2 の必要条件

```text
min(B_ram, B_prepare, B_h2d) >= S_stream × T_target
```

において、`B_prepare` が律速になり得る。`mmap` 経路（56.39 GB/s）なら余裕が
ある。Phase 3 では **Host Store のモード（mmap / read）を必ず両方測る**こと。
片方だけ測って「RAM は十分速い」と結論しないこと。

なお上表はページキャッシュ warm の値である。cold（初回読み出し）は SSD 帯域に
律速される。cold 値は未測定。

### 3.3 gemma4:12b は RTX 3060 の VRAM に収まる → 検証対象を 31B へ移した

総テンソルバイト 7,365,558,464 = 6.86 GiB < 12 GB。
**このモデルでは ALSR は本来不要**であり、Phase 6 の位置づけどおり
「機構検証」用の対象でしかない。12B の Run で「ALSR が速い / 遅い」を
語ってはいけない。

VRAM 超過領域の検証対象として `gemma4:31b`（テキスト側 18.49 GiB）を
追加した（DR-0009）。ただし **M5 の 16 GiB RAM には収まらない**ので、
31B で取得できるのは解析・Chunk 計画・シミュレーションだけであり、
**Ollama の実推論 baseline は取得できない**。

さらに M5 では 20 GB がページキャッシュに収まらないため Host Store が
SSD 律速になる。これは M5 固有の制約であり、96 GB RAM を積む RTX 検証機の
予測にはならない。混同しないこと。

### 3.3b Chunk サイズは PCIe 律速の領域ではほとんど効かない

`gemma4:31b` / 常駐 10 GB / H2D 25 GB/s でのスイープ（§5.1）:

| Chunk | token/s |
|---|---:|
| 64 MiB | 2.8536 |
| 128 MiB | 2.8604 |
| 256 MiB | 2.8639 |
| 512 MiB | 2.8303 |
| 1024 MiB | **測定不能**（VRAM 不足） |

**幅は 1.2% しかない。** PCIe 律速では律速要因が平均帯域であって粒度では
ないので、これは物理的に妥当な null result である。

**Chunk サイズを最適化しても意味が無い領域がある**ことを、Phase 3 の
スイープ設計に持ち込むこと。逆に、常駐率を上げて PCIe 律速から抜けた
領域では粒度が効き始める可能性があり、そこは未測定。

**1024 MiB が測定不能だったこと自体が重要な制約**である。
ring 4 × 1 GiB = 4.28 GB + 常駐 10 GB = 14.28 GB > 12 GB VRAM。
大きな Chunk と高い常駐率は同じ VRAM を奪い合う。

### 3.4 転送だけで決まる上限（gemma4:12b, S_stream 上限ケース）

| H2D 帯域 [GB/s] | T_max [token/s] | 備考 |
|---:|---:|---|
| 12 | 1.63 | PCIe 3.0 x16 相当 |
| 25 | 3.39 | PCIe 4.0 x16 相当 |
| 113 | 15.34 | M5 実測から逆算した実効値 |

M5 の実測 15.35 token/s と 113 GB/s 行がほぼ一致する（DR-0005）。
**RTX 3060 に PCIe 経由で毎トークン全重みを流す方式では、M5 の 1/9〜1/4.5 の
速度が転送だけで確定する。** これは実装の巧拙で埋まる差ではない。

---

## 4. 実装上の既知の制約

### 4.1 単スレッド

設計書 §8.1 の方針（「初期版で無理に全処理を別スレッド化しない」）に従い、
シミュレータは単スレッドである。時刻はパイプライン漸化式で進めるので、
オーバーラップは正しくモデル化されるが、**スレッド間の競合・偽共有・
ロック競合は 1 つも検証していない**。Phase 4 で実スレッド化する際は
別途テストが要る。

### 4.2 適応 Scheduler は実 plan では Chunk サイズを動かせない

実 plan を使う Run では Chunk サイズが計画で確定しているため、
`allow_chunk_size_change` が自動的に false になる。適応できるのは
先読み距離だけである。この抑制は `scheduler_log.json` に記録される。
合成 Chunk の Run では両方が動く。

Chunk サイズを実行時に変えるには、再計画または「連続する計画 Chunk の併合」が
要る。未実装。

### 4.2b マルチモーダル GGUF の視覚側の扱いは仮置き

`gemma4:31b` は視覚エンコーダ (1,186,547,904 B) と投影器 (12,386,304 B) を
本体に同梱している。本 PoC は「テキスト生成の毎トークンでは読まれない」と
判定して定常ストリーミングから外している（DR-0011）。

**これは『不要』という意味ではない。** 画像入力を扱う経路では必要になる。
未決なのは:

- 視覚エンコーダを VRAM 常駐にするか、画像入力のたびにロードするか
- 画像入力の頻度が高い場合、常駐予算をどう配分するか
- 視覚側を含めた Run を測るときの `exclude_non_per_token_chunks = false` の
  結果はまだ取っていない

### 4.3 GGUF v3 のみ

v1 / v2 は明示的に `Error(Unsupported)` にする。推測で読まない。

### 4.4 ggml 型テーブルは実測で検証されている範囲のみ信頼できる

型テーブル（`src/inspector/ggml_types.cpp`）が正しいことは、
「Σ tensor_bytes + padding == file_size - data_start」の整合チェックで機械的に
検証される。gemma4:12b で実際に現れた **F32 / Q4_K / Q6_K** はこの検査を通って
いる。それ以外の型（IQ 系、TQ 系、MXFP4 等）は**まだ実ファイルで検証していない**。
未検証の型が現れた Run では、整合チェックが落ちるか、落ちなければ正しい。
黙って間違った値を使うことはない設計だが、テーブル値そのものの一次検証は未了。

Q8_1 / Q8_K は中間量子化型として `storable=false` にしてあり、現れたら
明示エラーになる。

### 4.5 12B の別レイヤー mmproj は対象外

`gemma4:12b` には model レイヤー（7,381,382,048 B）とは別に
mmproj レイヤー（175,115,584 B, `mmproj-gemma-4-12B-it-bf16.gguf`）がある。
本 PoC は model レイヤーのみを対象とする。

`gemma4:31b` のように**本体に同梱された**視覚エンコーダ・投影器は
`Subsystem` として分離済み（4.2b / DR-0011）。しかし 12B のように
**別ファイルになっている場合の統合は未対応**である。同じモデルファミリでも
ファイル構成が違うので、対象モデルごとに確認すること。

### 4.6 checksum は CRC-32C（32 bit）

7.37 GB を 30 Chunk に分けた場合、1 Chunk あたり最大 256 MiB。
CRC-32C はバースト誤り検出には強いが、暗号学的な改竄検出には使えない。
証拠管理（§16.4）には SHA-256 を別途使っている。

### 4.7 `materialize_data=false` の Run

実バイトを流さないので正確性の証拠にならない。必ず必須ゲートを落とす
（DR-0007）。長時間のスケジューラ観察専用。

---

## 5. Phase 3 へ進む前の未解決事項

設計書 §20 のうち、Phase 0-2 で解消したもの／残るもの:

| 項目 | 状態 |
|---|---|
| RTX 検証機の OS | **未解決**（実機で採取） |
| RTX 3060 のボードメーカー／型番 | **未解決** |
| 実機の PCIe current link speed / width | **未解決**（§3.2 の x8/x8 リスク） |
| 96GB Mixed DIMM の実効帯域 | **未解決**（STREAM 系で測る） |
| CUDA Driver / Toolkit 版 | **未解決** |
| 対象 Ollama blob の実ファイルパスと digest | **解決**（下記） |
| Gemma 4 12B を扱える llama.cpp commit | **未解決**（Phase 5） |
| 「Opus 5」の正式 CLI model ID | **解決** — `claude-opus-5`（DR-0001） |
| 70B 以降で使う具体的モデルと量子形式 | **未解決** |

解決した blob 情報:

```text
ref     : gemma4:12b
id      : 4eb23ef187e2
blob    : ~/.ollama/models/blobs/sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606
size    : 7,381,382,048 bytes
gguf    : version 3, 667 tensors, 49 kv, alignment 32
arch    : gemma4, block_count 48, embedding_length 3840, context_length 262144
quant   : general.file_type 15 (Q4_K_M 相当)。F32 338 / Q4_K 284 / Q6_K 45
mmproj  : sha256-675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842
          (175,115,584 bytes, 本 PoC の対象外)
```

解決した blob 情報（31B）:

```text
ref     : gemma4:31b
id      : 6316f0629137
size    : 19 GB
gguf    : version 3, 1189 tensors, alignment 32
arch    : gemma4, block_count 60
内訳    : text 18,654,198,000 / vision 1,186,547,904 / projector 12,386,304
実測    : 0.6017 byte/param (Q4_K_M)
```

追加で Phase 3 前に決めること:

- 言語構成（C++ 単一 / C++ + Rust ハイブリッド）— DR-0008 で Phase 3 判定と決定
- ライセンス — `LICENSE` が未確定のまま
- checksum を性能 Run で有効にするかどうかの既定値（3.1 の制約による）
- **§5.1 の「llama.cpp 系実行経路への接続可否調査」が未実施**
  （`docs/CONFORMANCE.md` §5.1 に未達として記載）
- 視覚エンコーダの常駐方針（4.2b）
