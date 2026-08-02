# 実験記録

Phase 0-2（M5 上）。すべての数値は `results/` 以下の生成物から転記している。
手で書き換えない（設計書 §16.4）。

---

## Phase 0 — 基準値と環境固定

### 0.1 環境

| 項目 | 値 |
|---|---|
| 機種 | MacBook Air, Apple M5 |
| ユニファイドメモリ | 17,179,869,184 B (16 GiB) |
| 論理コア | 10 |
| OS | macOS 26.5.2 (build 25F84) |
| コンパイラ | Apple clang 21.0.0 |
| CMake | 4.4.2（本 PoC のために新規導入） |
| Python | 3.14.6 |
| Ollama | 0.32.5 |

再取得: `bash scripts/collect_environment.sh` → `results/environment.json`

### 0.2 対象モデル

| 項目 | 値 |
|---|---|
| ref | `gemma4:12b` |
| ollama id | `4eb23ef187e2` |
| model blob | `sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606` |
| blob サイズ | 7,381,382,048 B (6.87 GiB) |
| GGUF | version 3 / 667 tensors / 49 kv / alignment 32 |
| arch | `gemma4`, block_count 48, embedding_length 3840, ffn 15360 |
| context_length | 262,144（Run は num_ctx 4096 で実施） |
| quantization | `general.file_type` 15（Q4_K_M 相当） |
| mmproj レイヤー | 175,115,584 B（本 PoC の対象外） |

型別内訳:

| 型 | テンソル数 | バイト |
|---|---:|---:|
| F32 | 338 | 3,080,384 |
| Q4_K | 284 | 5,246,484,480 |
| Q6_K | 45 | 2,115,993,600 |
| 合計 | 667 | 7,365,558,464 |

### 0.3 Ollama 基準値（M5, ユニファイドメモリ）

`bash scripts/baseline_m5.sh` → `results/baseline_m5.json`

設定: num_ctx 4096 / num_predict 256 / temperature 0 / seed 20260802 /
warm-up 1 回を除外して 5 サンプル。

| 指標 | 値 [token/s] |
|---|---:|
| 中央値 | **15.97** |
| 平均 | 15.81 |
| 最小 | 14.94 |
| 最大 | 16.22 |

ユーザー提供の一次証拠（`results/baseline_m5_user_provided/provenance.json`）:

| Run | eval_count | eval_duration | token/s |
|---|---:|---:|---:|
| 1024 トークン | 1024 | 66.707 s | **15.35** |
| 256 トークン | 256 | — | **16.49** |

`ollama ps`: SIZE 8.0 GB / PROCESSOR 100% GPU / CONTEXT 4096。
設計書 §3.1 の「実測モデル占有 約 8.0GB」「実測生成速度 約 15.35〜16.49 token/s」
と一致する。

### 0.4 帯域実測

`./build/bin/alsr-bw --buffer-mib 512 --repeats 7 --ollama gemma4:12b --read-mib 2048`
→ `results/bandwidth_m5.json`, `results/bandwidth_report.md`

| 指標 | 中央値 [GB/s] | 備考 |
|---|---:|---|
| ram_memcpy | 55.71 | 単一スレッド |
| ram_sequential_read | 78.73 | 単一スレッド、u64 総和 |
| crc32c | 11.77 | ARMv8 CRC32 命令使用 |
| file_read_mmap | 56.39 | ページキャッシュ warm |
| file_read_read (pread) | 17.06 | ページキャッシュ warm |

**この Run で判明した設計上の制約は `KNOWN_LIMITATIONS.md` §3 に書いた。**
要点は 2 つ:

1. 単一コアの CRC32C（11.77 GB/s）は PCIe 4.0 x16 の実効 H2D（20〜25 GB/s）より
   遅い。全 Chunk のホスト側 checksum を線速で回せない。
2. `pread`（17.06 GB/s）は PCIe 4.0 に対して余裕が薄い。`B_prepare` が律速に
   なり得る。`mmap`（56.39 GB/s）なら余裕がある。

### 0.5 M5 では測れないもの

Pageable / Pinned H2D、PCIe 実効帯域、CUDA コピーエンジン、
copy/kernel 重複率、VRAM 枯渇時の実 GPU 待機。すべて Phase 3（RTX 機）。

---

## Phase 1 — Model Inspector

`./build/bin/alsr-inspect --ollama gemma4:12b --manifest-out results/manifest_gemma4_12b.json`

### 1.1 合格条件の判定

| Phase 1 合格条件 | 結果 |
|---|---|
| 解析対象 byte 範囲の重複／欠落が説明可能 | **OK** |
| 同じモデルで manifest が決定論的 | **OK**（単体テストで固定） |
| 未対応形式を明示エラーにできる | **OK**（単体テストで固定） |

レイアウト整合:

```text
data bytes         : 7,365,559,808
tensor total       : 7,365,558,464
padding            :         1,344   (alignment 32 で説明可能)
unexplained        :             0
issues             :             0
```

### 1.2 実行順序マニフェスト

- block_count 48（メタデータとテンソル名から得た層数が一致）
- entries 667、**unknown 0**（19 種のテンソル名パターンをすべて分類できた）
- 常駐候補（1 MiB 未満）: 3,080,384 B = 2.94 MiB。全 F32 の norm / scale / rope

人間の確認を要求した項目が 1 件:

> `output.weight` が無い。tied embedding を lm_head として全読みする実装なら
> 毎トークン +825,753,600 bytes。実行経路(Phase 5)で要確認

### 1.3 S_stream と T_max（設計書 §2.3）

| 量 | 値 |
|---|---:|
| S_stream 下限（埋め込み行列を除く） | 6,539,804,864 B/token |
| S_stream 上限（tied lm_head で全読み） | 7,365,558,464 B/token |

| H2D [GB/s] | T_max 上限ケース [tok/s] | T_max 下限ケース [tok/s] |
|---:|---:|---:|
| 5 | 0.6788 | 0.7645 |
| 10 | 1.3577 | 1.5291 |
| 12 | 1.6292 | 1.8349 |
| 16 | 2.1723 | 2.4466 |
| 25 | 3.3942 | 3.8227 |
| 28 | 3.8015 | 4.2815 |
| 113 | **15.3417** | 17.2788 |

### 1.4 この Phase で得た最大の知見

M5 実測 15.35 token/s（ユーザー提供の 1024 トークン Run）を S_stream 上限で
割り戻すと **113.06 GB/s**。上表の 113 GB/s 行が 15.3417 tok/s を返す。
自前スクリプトの中央値 15.97 token/s では **117.59 GB/s**。

つまり:

- gemma4:12b は毎トークン **tied embedding を含む全重み 6.86 GiB を読んでいる**
  （上限ケースが実態に近い）。設計書 §20 の未確定事項が 1 つ埋まった（DR-0005）。
- M5 のユニファイドメモリは重み供給に **113〜118 GB/s** を出している。
- **RTX 3060 に PCIe 経由で毎トークン全重みを流す方式は、PCIe 3.0 で 1.63 tok/s、
  PCIe 4.0 で 3.39 tok/s が転送だけで確定する。M5 の 1/9〜1/4.5 である。**
  これは実装の巧拙で埋まる差ではない（設計書 §2.3 / §14 の最上段リスク）。

なお **gemma4:12b (6.86 GiB) は RTX 3060 の 12 GB VRAM に収まる**ので、
このモデル自体は ALSR を必要としない。Phase 6 の位置づけどおり機構検証専用。

---

## Phase 2 — Chunk Planner と帯域シミュレータ

### 2.1 Chunk 計画（256 MiB 固定）

`./build/bin/alsr-plan --ollama gemma4:12b --chunk-mib 256 --out results/plan_gemma4_256mib.json`

| 指標 | 値 |
|---|---:|
| chunks | 30 |
| covered bytes | 7,365,559,808（= data_start から末尾まで完全被覆） |
| tensor bytes | 7,365,558,464 |
| padding bytes | 1,344 |
| min / max chunk | 16,384 / 268,433,760 |
| split tensors | 1（`token_embd.weight`, Q6_K 825 MB） |
| exec-order 逆転 | 1 / 29 |
| structure sha256 | `170474fdb8a782b95449c0c3dd9909585dae63d75a9571395c0852830dab2154` |

`verify_plan()` は毎回走り、被覆・重複・境界をすべて検査して通っている。

分割粒度: Q6_K の量子ブロック 210 B と alignment 32 B の最小公倍数 = 3,360 B。
`token_embd.weight` の切れ目は 268,433,760 B（= 79,891 × 3,360）刻み。

**exec-order 逆転が 1 箇所しかない**のは重要な観測である。`output_norm.weight`
がファイル先頭にあるだけで、それ以外はファイル順 ≒ 実行順。つまり
**逐次先読みがほぼそのまま効く**。MoE ではこうならないと予想される（§2.5）。

### 2.1.1 粒度を細かくすると実行順とファイル順のズレが露出する

同じモデルを可変 Chunk（min 16 MiB / max 512 MiB）で刻むと:

| モード | chunks | exec-order 逆転 |
|---|---:|---:|
| 固定 256 MiB | 30 | **1 / 29** |
| 可変 16–512 MiB | 308 | **153 / 307** |

粒度を細かくするとブロック内のテンソル順（ファイル上は `attn_k`, `attn_k_norm`,
`attn_norm`, `attn_output`, `attn_q`, … とアルファベット順に近い並び。実行順は
`attn_norm` → `attn_q` → `attn_q_norm` → `attn_k` → …）のズレが Chunk 境界を
またいで露出する。

**これは Chunk サイズ選択のトレードオフに直結する。**

- 大きい Chunk: ファイル順の逐次先読みがそのまま効くが、粒度が粗く
  待機とVRAM浪費が増える（§14「大きすぎる Chunk」）
- 小さい Chunk: 制御は細かくできるが、実行順の先読みが飛び飛びのファイル
  アクセスになる。mmap のページフォールト局所性と `readahead` が効かなくなる

Phase 3 の Chunk サイズスイープでは、**転送効率だけでなくこの逆転数も併記する**
こと。片方だけ見て最適サイズを決めない。

### 2.2 サンプル Run 3 本

`bash scripts/run_samples.sh` → `results/<run_id>/`

| Run | scheduler | H2D 設定 | 実効 H2D | token/s | GPU 待機率 | 終盤待機率 | 重複率 | starvation | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| run01 gemma4 fixed | fixed | 25 GB/s | 25.05 | **3.361** | 91.76 % | 91.60 % | 8.04 % | 25 | PASS |
| run02 gemma4 adaptive | adaptive | 12 GB/s | 12.04 | **1.625** | 96.02 % | 95.96 % | 3.86 % | 25 | PASS |
| run03 synthetic healthy | adaptive | 25 GB/s | 24.63 | — | 0.63 % | 0.00 % | 98.42 % | 0 | PASS |

**数理モデルとシミュレータの自己整合**:

| Run | 式による T_max | シミュレータの token/s | 差 |
|---|---:|---:|---:|
| run01 (25 GB/s) | 3.3942 | 3.3607 | −1.0 % |
| run02 (12 GB/s) | 1.6292 | 1.6249 | −0.3 % |

差はレイテンシ・jitter・パイプライン立ち上がりで説明できる範囲。
**式と実装が独立に同じ答えを出している**ことの確認になる（§15.3 の
「数理モデルと実測が 20% 以上乖離」には遠く及ばない）。

### 2.3 スケジューラの挙動

run02（PCIe 3.0 相当、帯域不足）で適応 Scheduler は 2 回変更した。
`results/<run_id>/scheduler_log.json` に理由が残る。実 plan なので Chunk サイズは
動かせず、先読み距離のみ。**それでも終盤待機率は 95.96% のままで、
平均帯域不足は制御で消えないことが実測で確認できた**（設計書 §2.2）。

run03（供給十分）では starvation 0、重複率 98.42%。制御則そのものは健全側で
正しく動いている。

### 2.4 再現性

同一設定で run01 を 2 回実行し、`summary.json` を比較:

- `wall_seconds` 以外は**完全一致**（7.367 s → 7.387 s）
- `metrics.jsonl` の SHA-256 も一致

「仮想時間は再現する / 壁時計は再現しない」が分離できている（DR-0004、NFR-05）。

### 2.5 ストレステストで固定した物理的主張

| 主張 | 検証方法 | 結果 |
|---|---|---|
| 平均供給 < 平均消費 なら必ず枯渇する | 深度 2/3/4/8/16 で供給 5 GB/s vs 消費 10 GB/s | 全深度で starvation > 0、終盤待機率 > 10% |
| 深度を増やしても平均不足は隠せない | 深度 2 と 16 を比較 | 両方とも終盤待機率 > 30% |
| 供給十分なら warm-up 後は枯渇しない | 供給 40 GB/s vs 消費 10 GB/s、prefetch 3 | starvation 0、重複率 > 90% |
| jitter がバッファ余裕内なら枯渇しない | 上記に jitter ±20% | starvation 0 |
| jitter が余裕を超えると枯渇が出る | 供給 11 vs 消費 10、prefetch 1、jitter ±50% | starvation > 0 |
| 仮想時間 600 秒の定常運転で進行性の枯渇がない | repeat 10 万周 / 上限 600 s | starvation 0、終盤待機率 0 |
| 小 Chunk 大量投入でも壊れない | 64 KiB × 8192 chunk、深度 4 | wrap-around 8188、順序違反 0、欠落 0 |
| 適応が振動しない | 帯域不足下で min_interval 2 | 変更回数 ≤ chunk 数 / 2 |

### 2.6 テスト結果

```text
unit        : 73 test cases, 3853 assertions — 全通過
integration : 12 test cases,  114 assertions — 全通過
stress      : 12 test cases,   78 assertions — 全通過
```

---

## 次に測るべきこと（Phase 3 の入口）

1. RTX 機の環境証拠（OS / Driver / CUDA / **PCIe current link gen & width**）。
   §3.2 の x8/x8 リスクがあるので `nvidia-smi` 表示だけを信じない。
2. RAM 帯域（96GB Mixed DIMM、STREAM 系）。理論 51.2 GB/s を実効値として扱わない。
3. Pageable vs Pinned H2D、Sync vs Async、Stream 1〜4、Chunk 16 MiB〜1 GiB。
4. 仮想時間モデルと CUDA Event 実測の乖離。20% 以上なら DR-0004 の反証条件。
5. checksum を性能 Run で有効にするかの方針（`KNOWN_LIMITATIONS.md` §3.1）。
