# A0_INVENTORY — Recovery Gate A0 棚卸し

| 項目 | 内容 |
|---|---|
| Mission | `ALSR-A0-RECOVERY-AUDIT-01`（改訂: `ALSR-A0-REM-01` — §12 参照） |
| 準拠設計書 | Adaptive Layer Streaming Runtime（ALSR）PoC 要件定義書・基本設計書 — Rev.02 Recovery Baseline（`ALSR-REQ-DESIGN-REV02-2026-08-02` / 文書版 2.0-RC1） |
| 実施日 | 2026-08-02 JST |
| 実施者 | Claude Code（Claude Opus 5, model ID `claude-opus-5`） |
| 本 Mission の性質 | **実装ではない。既存成果物の棚卸しのみ。** |
| コード変更 | **0**（Rev.02 §8 Gate A0 GREEN条件） |
| 状態 | `COMPLETED_NOT_AUDITED`（Rev.02 §10.2 / 固定7） |

> **この文書は監査結果ではない。** 実装担当による棚卸しの提出物である。
> GREEN / YELLOW / RED の裁定は豆虎Gate が行う（Rev.02 §2.2）。

---

## 0. 本 Mission で実施しなかったこと（Rev.02 §14 禁止事項の遵守）

| 禁止事項 | 遵守状況 |
|---|---|
| コード修正 | 実施していない。`src/` `include/` `backends/` `tests/` `tools/` `scripts/` `configs/` `CMakeLists.txt` に変更なし |
| build | 実施していない。既存の `build/` は本 Mission 開始前（2026-08-02 14:32）の生成物 |
| benchmark 再実行 | 実施していない。`results/` 配下のファイルは一切更新していない |
| model download | 実施していない |
| model 削除 | 実施していない |
| requirement 変更 | 実施していない。`docs/ALSR_REQ_DESIGN.md` を含む既存文書を 1 行も変更していない |
| Gate 進行 | 実施していない。Gate P0 / M1 / M2 / M3 / R0 のいずれにも着手していない |

追加は `docs/recovery/` 配下の成果物 5 点のみ。

---

## 1. 前提条件（ユーザー宣言）

| ID | 宣言内容 | 出所 | 監査上の扱い |
|---|---|---|---|
| P-01 | 「まだ着手したばかりの研究なので Git 管理は行なっていません。検証内容如何によってはお蔵入りも有り得るので。」 | ユーザー（2026-08-02, 本セッション） | remote 不在・`results/` 非追跡・branch 戦略不在は**意図的な運用方針**であり、逸脱として判定しない。Rev.02 §14 実施1/2 が要求する Git 情報の記録自体は実施した |

この宣言により、Deviation Matrix の D-13 / D-14（`git_status.txt` / `git_diff.patch` 不在）は severity を LOW に留めている。

---

## 2. Repository 情報（Rev.02 §14 実施1）

| 項目 | 値 |
|---|---|
| path | `/Users/yoshiki/Desktop/ALSR-PoC` |
| branch | `main`（唯一のブランチ） |
| HEAD | `0ce74bda24843ead02ec2ee992febd65ab8882ae` |
| remote | **無し**（P-01 により意図的） |
| upstream | 無し |
| working tree（監査開始時） | **clean**（変更・未追跡ファイルともに 0） |
| stash | 無し |
| tag | 無し |
| 追跡ファイル数 | 71 |
| 索引対象ファイル総数 | 150（`build/` `third_party/_fetch/` `.git/` `__pycache__/` `.claude/` を除く） |

> 初版は索引対象を 148 としていた。`third_party/` をディレクトリごと除外した際、取得物ではない
> 追跡ファイル 2 件（`third_party/.gitkeep` / `third_party/README.md`）まで落としていたためである。
> `ALSR-A0-REM-01` で 2 件を索引へ追加し、追跡 71 件と索引の `git_tracked` 件数を一致させた（D-43 / §7.2）。

### 2.1 リポジトリ実体パスの移行（`ALSR-A0-REM-02` item 6 / D-45）

**本監査の作業中に、リポジトリの実体パスが移動した。** 記録しておく。

| 項目 | 値 |
|---|---|
| 移行前 | `/Users/yoshiki/Library/Mobile Documents/com~apple~CloudDocs/Desktop/デスクトップ - m-nexspaceのMacBook Air/ALSR-PoC` |
| 移行後（現在） | `/Users/yoshiki/Desktop/ALSR-PoC` |
| 性質 | iCloud Drive のデスクトップ同期配下 → ローカル `~/Desktop` 実体 |
| 検出 | `ALSR-A0-REM-01` の作業中。shell が「Working directory was deleted」を返し、旧パスの `ALSR-PoC` が消失 |
| 発生源 | **本監査の操作ではない。** 環境側で発生した（iCloud デスクトップ同期の解除と推測されるが、**A0 は推測で埋めない**ため原因は未確定） |
| 記録 | `A0_EVIDENCE_INDEX.json:path_migration` |

移行前後の同一性確認:

| 項目 | 移行前 | 移行後 |
|---|---|---|
| HEAD | `0ce74bda…8882ae` | `0ce74bda…8882ae` |
| `git status --porcelain` | `?? docs/recovery/` | `?? docs/recovery/` |
| 追跡ファイル数 | 71 | 71 |
| `git fsck --full --strict` | （未実施） | 異常報告なし（exit 0） |

**注意**: §2 の `path` は初版から `/Users/yoshiki/Desktop/ALSR-PoC` と記録されていた。
これは移行後のパスと一致するが、初版の記録時点で実体があったのは旧パスである。
初版はユーザー環境上の論理パス（`~/Desktop`）を記録しており、iCloud 同期による
実体の所在までは記録していなかった。**結果として値は正しいが、根拠は一致していなかった。**

---

## 3. Git 解析（Rev.02 §14 実施2）

### 3.1 commit 一覧（全 7 件、すべて 2026-08-02）

| # | commit | 時刻 | 件名 | 変更規模 |
|---|---|---|---|---|
| 1 | `b29e9ce` | 13:49:50 | ALSR PoC Phase 0-2: 解析器・Chunk計画・仮想時間シミュレータ | 66 files, +11458 |
| 2 | `027c0aa` | 13:50:32 | run03 の per_token_bytes を 0 にする（合成データに実モデルの S_stream を適用しない） | 2 files, +8/−4 |
| 3 | `23424ae` | 14:03:12 | VRAM 常駐率のモデル化を追加 (設計書 §2.4) | 8 files, +266/−3 |
| 4 | `b45c990` | 14:05:01 | 常駐率のテストを追加し、境界の誤読を修正 | 3 files, +91/−3 |
| 5 | `8b8a2e6` | 14:31:16 | 設計書適合監査と逸脱の是正、31B での VRAM 超過領域の検証 | 27 files, +1899/−94 |
| 6 | `af35cae` | 14:32:07 | KNOWN_LIMITATIONS / README を 31B の実測結果で更新 | 2 files, +76/−3 |
| 7 | `0ce74bd` | 14:32:25 | README / KNOWN_LIMITATIONS の記述を実態に合わせる | 2 files, +9/−4 |

すべての commit の author は `Claude Opus 5 <noreply@anthropic.com>`。

### 3.2 未追跡・除外ファイル

| 対象 | 状態 | 理由 |
|---|---|---|
| `results/**`（80 ファイル中 79） | `.gitignore:22 results/*` により除外 | Evidence Pack を明示的に `git add -f` する運用の想定。Pack 本体は 1 件も追跡下に無く、**証拠は Git 管理外にディスク上のみ存在する** |
| `results/.gitkeep`（1 ファイル） | `.gitignore:23 !results/.gitkeep` により**除外解除・追跡下** | ディレクトリ保持用の空ファイル。証拠ではない |
| `build/**` | `.gitignore` により除外 | ビルド生成物 |
| `third_party/_fetch/**` | `.gitignore` により除外 | Catch2 v3.15.3 の FetchContent 取得物 |
| `.claude/settings.local.json` | `.gitignore` 対象外だが未追跡 | ツール設定 |

**P-01 により、これは運用方針上の選択であり逸脱ではない。** ただし「お蔵入りの可能性」がある以上、`results/` がバックアップ対象外である事実は記録しておく。

> `ALSR-A0-REM-03` までは「`results/` 80 ファイル」を一括で非追跡と記述していたが、
> 正しくは **追跡 1（`results/.gitkeep`）+ 非追跡 79** である（D-55）。
> 追跡されている 1 件はディレクトリ保持用の空ファイルであり証拠ではないため、
> 「Evidence 本体が Git のアンカーを持たない」という結論自体は変わらない。

### 3.3 diff

working tree は監査開始時 clean であり、`git diff` / `git diff --cached` はともに空。
本 Mission 完了時点の差分は `docs/recovery/` 配下の新規 5 ファイルのみ（§9 参照）。

---

## 4. Requirement Mapping（Rev.02 §14 実施3）

Rev.02 の各条項に対する実装状況。詳細は `A0_DEVIATION_MATRIX.csv`（60 行）を参照。

### 4.1 分類集計

| 分類 | 件数 |
|---|---|
| 実装済 | 4 |
| 部分実装 | 7 |
| 未実装 | 21 |
| 範囲外 | 1 |
| 誤認 | 25 |
| 記録済逸脱 | 2 |
| **合計** | **60** |

severity: HIGH 19 / MEDIUM 21 / LOW 14 / INFO 6

| 版 | 行数 | 追加 |
|---|---:|---|
| 初版 | 42 | — |
| `ALSR-A0-REM-01` | 44 | D-43（Evidence Index の追跡ファイル欠落 2 件）/ D-44（整合検証の結論が検証範囲を超えていた件）。D-34 の severity を MEDIUM → HIGH |
| `ALSR-A0-REM-02` | 47 | D-45（リポジトリ実体パス移行の未記録）/ D-46（「113〜118 GB/s」が 2 母集団の混合）/ D-47（A0-007 の 3 主張同居と A0-004 の `evidence_verified` 誤り） |
| `ALSR-A0-REM-03` | 52 | D-48（A0-018 の主判定矛盾）/ D-49（MP-118 の Provenance 誤分類）/ D-50（MP-125・MP-121 の依存関係誤記）/ D-51（改訂履歴と Git 状態の記録欠落）/ D-52（パス連続性のスコープ超過＝ D-44 の再発） |
| `ALSR-A0-REM-04` | 57 | D-53（A0-017 の粒度・判定誤り）/ D-54（時間スコープ超過の残存）/ D-55（`results/` の Git 追跡件数誤記）/ D-56（Claim summary の過大表現）/ D-57（類型集計の不整合） |
| `ALSR-A0-REM-05` | 60 | D-58（A0-017B の未立証 runtime・OS 断定）/ D-59（改訂履歴の回数・見出し不整合）/ D-60（不変とする数値のスコープ未限定） |

D-46 は既存文書（`README.md`）側の問題であり、文書の書き換えは禁止事項のため記録のみ行った。

#### A0 成果物側の欠陥（`ALSR-A0-REM-05` / D-59 で全 ID を再集計）

`ALSR-A0-REM-03` までは類型別件数と列挙 ID が一致していなかった（「分類・記録の誤り 2 件」としながら 3 ID を列挙）。
件数を伏せるのではなく、**類型ごとに全 ID を書き出して合計が一致することを確認できる形**へ改めた。

| 類型 | 件数 | Deviation ID（全件） |
|---|---:|---|
| A. 証拠が支える範囲より広く書いた（スコープ超過） | 7 | D-44 / D-48 / D-52 / D-54 / D-56 / D-58 / D-60 |
| B. 主張の粒度が証拠の粒度と合っていない | 2 | D-47 / D-53 |
| C. 分類・出自・依存関係の誤り | 2 | D-49 / D-50 |
| D. 記録の欠落・件数・履歴の誤り | 6 | D-43 / D-45 / D-51 / D-55 / D-57 / D-59 |
| **合計** | **17** | — |

合計 17 件は、Deviation Matrix 60 行のうち A0 成果物自身の欠陥として記録した全件と一致する。
類型 A / B の分け方には判断が入るため、根拠は各 ID の note に記載している。

**同型の誤りは 2 類型で、3 つの再発関係として確認されている。** D-52 は D-44 の再発（事象と状態の混同）、
D-54 は D-52 の残存（同類型の時間軸での再発）、D-53 は D-47 の再発（粒度不整合）である。
**1 度の是正では同じ型の誤りが消えない。**

### 4.2 Rev.02 の Gate 体系に対する現在地

| Gate | 内容 | 状態 |
|---|---|---|
| **A0** | Recovery Audit | **本 Mission で実施。裁定待ち** |
| P0 | Prior Art（llama.cpp / Ollama / CUDA 調査） | **未着手**（D-03 / D-26） |
| M1 | M5 Baseline 固定 | 内容は実施済。**Gate 認定は未実施**（D-29） |
| M2 | Model Inspector | 内容は実施済。**Gate 認定は未実施**（D-30） |
| M3 | 数理モデル固定（PHYSICAL_MODEL.md + 3 スキーマ） | **未着手**（D-04 / D-01 / D-31） |
| R0 | RTX 環境取得 | **未着手**。RTX 機に一切アクセスしていない（D-39） |

**重要な構造的問題**: Rev.02 §8 Gate M3 は「コードより先に式を固定する」と定めるが、
現状は実装（`src/` 11,458 行超）が先行し、数理モデル文書とスキーマが存在しない。
既存実装は旧設計書 Rev.1.1（`docs/ALSR_REQ_DESIGN.md` 文書版 1.1）の Phase 0-2 体系に沿って
作られており、Rev.02 の Gate 体系へは自動的には対応しない。

### 4.3 Rev.02 §9.2 実装順序に対する現在地

| # | 項目 | 状態 |
|---|---|---|
| 1 | Model Resolver | 部分実装（`src/cli/cli_common.hpp:109` の関数のみ。独立モジュール不在。D-06） |
| 2 | GGUF Inspector | 実装済（`src/inspector/`） |
| 3 | Metric / Provenance Schema | **未実装**（D-01 / D-21） |
| 4 | Deterministic Queue | 実装済（`src/buffer/ring_buffer.cpp`。ディレクトリ名は `queue` でなく `buffer`。D-07） |
| 5 | Static Chunk Planner | 実装済（`src/planner/`） |
| 6 | RTX H2D Benchmark | **未実装**（RTX 機未アクセス） |
| 7 | CUDA Overlap Test | **未実装**（`backends/cuda/` は README のみ） |
| 8 | VRAM Budget Calculator | **未実装**（D-24。常駐量は手計算の設定値） |
| 9 | Baseline 比較 | 部分実装（M5 の Ollama 基準値のみ。既存方式との比較は未実施） |
| 10 | Fixed ALSR | 実装済（`src/sim/`、fixed scheduler） |
| 11 | Adaptive Scheduler | **実装済かつ使用済み**（run02 / run05）。**Rev.02 固定4「Adaptive Scheduler は固定方式・実GPU計測後」に抵触**（D-27） |

### 4.4 Rev.02 §13 リポジトリ構成との差分

| 要求 | 現状 |
|---|---|
| `CMakeLists.txt` | ✅ 存在 |
| `README.md` | ✅ 存在 |
| `docs/ALSR_REQ_DESIGN.md` | ⚠️ 存在するが **Rev.1.1**。Rev.02 は未配置（D-38） |
| `docs/PRIOR_ART.md` | ❌ 不在（D-03） |
| `docs/PHYSICAL_MODEL.md` | ❌ 不在（D-04） |
| `docs/DECISIONS.md` | ✅ 存在（DR-0001〜DR-0011） |
| `docs/CLAIMS.md` | ❌ 不在（D-05） |
| `schemas/` 3 ファイル | ❌ ディレクトリごと不在（D-01） |
| `include/` `src/` `backends/` `tests/` `tools/` `configs/` | ✅ 存在 |
| `src/resolver` | ❌ 不在（D-06） |
| `src/inspector` `src/planner` | ✅ 存在 |
| `src/queue` | ⚠️ `src/buffer` として存在（D-07） |
| `src/metrics` | ⚠️ `src/telemetry` として存在（D-08） |
| `evidence/` | ⚠️ `results/` として存在。`<gate>` 階層なし（D-02） |

リポジトリ内に存在するが Rev.02 §13 に記載の無いもの:
`docs/CONFORMANCE.md` / `docs/EXPERIMENTS.md` / `docs/KNOWN_LIMITATIONS.md` / `src/cli` /
`src/sim` / `src/buffer` / `src/scheduler` / `scripts/` / `LICENSE`。
いずれも旧 Rev.1.1 の構成に由来する。Rev.02 §13 は「予定リポジトリ構成」であり
追加物を明示的に禁じてはいないため、逸脱としては計上していない。

---

## 5. Metric Provenance 分類（Rev.02 §14 実施4）

詳細は `A0_METRIC_PROVENANCE.json`（125 項目）を参照。

| 種類 | 件数 | 主な内容 |
|---|---:|---|
| USER_OBSERVED | 6 | 15.35 / 16.49 token/s、`ollama ps` の 8.0 GB・100% GPU |
| MEASURED_M5 | 25 | baseline 中央値 15.965 token/s、帯域 5 指標、GGUF 静的解析値、Chunk 計画値、wall_seconds |
| **MEASURED_RTX** | **0** | **RTX 検証機に一切アクセスしていない** |
| EMULATED | 36 | 全 Run の token/s・実効 H2D・GPU 待機率・starvation、Chunk スイープ |
| DERIVED | 29 | S_stream 上限/下限、T_max 表、逆算帯域 113.06 / 117.59 / 121.46 GB/s、405B 外挿（202.5 GB を含む）、常駐 2.16 倍 |
| ASSUMED | 26 | H2D 12/25 GB/s、prepare 100 GB/s、consume 300 GB/s、device_memory 12 GiB、resident 10 GB |
| DECLARED_SPEC | 3 | RTX3060 VRAM 12 GB、PCIe 5.0 x16 63 GB/s、DDR4-3200 51.2 GB/s |
| UNKNOWN | 13 | Gate R0 の全取得項目、31B の実推論 baseline、cold 読み出し帯域、S_stream 上限/下限の確定 |

初版は 124 項目（DERIVED 27 / DECLARED_SPEC 4）。`ALSR-A0-REM-02` で MP-125（117.59 GB/s）を追加した。
`README.md:203` の「113〜118 GB/s」の上端 118 はこの値の丸めでありながら、初版の台帳に項目が無かった。

`ALSR-A0-REM-03` で **MP-118（405B 理想 4bit 202.5 GB）を DECLARED_SPEC → DERIVED へ訂正**した（D-49）。
202.5 GB はいかなる機器・規格の仕様値でもなく、405e9（公称パラメータ数 / ASSUMED）× 0.5 byte/param
（理想 4bit という理想化 / ASSUMED）の算術結果である。この誤分類により、A0-011 の
**「設計書の見積りより 20% 大きい」が DERIVED どうしの比較である**ことが隠れていた。比較の基準側も仮定に依存する。

### 5.0 値の所在と、入力側の出自（`ALSR-A0-REM-02`）

初版の `_method` は「値は `results/` 配下の実ファイルから直接読み出しており、手転記していない」と
一括して述べていたが、これは全体の一部にしか当てはまらなかった。値の所在で 3 系統に分離した。

| 系統 | 件数 | 内容 |
|---|---:|---|
| (A) results 由来 | 114 | `results/` 配下の実ファイルから直接読み出した。手転記なし |
| (B) docs 由来 | 11 | Markdown 内にのみ存在（MP-115〜125）。DECLARED_SPEC 3 + DERIVED 8。うち DERIVED 8 件は Rev.02 §10.3 の evidence 要求を未充足 |
| (C) 再計算値 | — | 本監査が計算した値。metrics 配列には計上せず、注記フィールド（`alternative_basis` / `series_separation`）として保持 |

さらに `provenance` を **出力側**と**入力側**に分離した（`_provenance_model`）。

| フィールド | 意味 |
|---|---|
| `provenance_final` | その値そのものの出自。従来の `provenance` と同義 |
| `provenance_inputs` | その値を得るのに使った入力の出自。一次値は `(一次値・入力なし)` |
| `value_origin` | `RESULTS_FILE` / `DOCS_ONLY` |

これにより次が明示的になった。

- **入力に ASSUMED を含む値は 65 項目**（自身が ASSUMED のものを除く）。仮定が変われば値も変わる。
- **EMULATED 36 項目は例外なく入力に ASSUMED を含む。** 「シミュレータが出した」ことは
  「入力が確からしい」ことを一切含意しない。
- **入力に MEASURED_RTX を持つ項目は 0。**

Rev.02 §5.2 が禁じる誤読（EMULATED を実測と読ませる）は、final だけを示して input を隠すことで起きる。

### 5.1 Simulation / Measurement の分離状況

Rev.02 §8 Gate A0 GREEN 条件「Simulation/Measurement 分離」に対して:

- **分離は成立している。** `results/*/config.json` の `resolved_backend.measured_on_real_hardware`
  が全 Run で `false` を返し、`caveat` に「M5 には独立 VRAM への PCIe H2D 経路が無いため、
  この結果で CUDA の成立を主張しない」と明記されている。
- ただし**機械可読な provenance フィールドは存在しない**（D-21）。分離は文書とフラグに依存しており、
  Rev.02 §5.2 が示す `{"metric":..., "provenance":"EMULATED"}` 形式の出力にはなっていない。

### 5.2 最も重い出自の問題

1. **EMULATED 値の入力が ASSUMED である**（D-23 / HIGH）
   全 Run の H2D 帯域 12 / 25 GB/s は `configs/*.json` のファイル名で `pcie3` / `pcie4` と
   命名され、文書でも「PCIe 3.0 / 4.0 相当」と呼ばれている。Rev.02 §4.2 は
   「RTX3060 だから PCIe4.0 x16」という推定を**明示的に禁止**し、実機取得値のみを許可する。
   MEG X570 UNIFY は GPU 配置により x16 / x8+x8 が変わり得るため、この仮定は
   Gate R0 の実測で置き換える必要がある。

2. **resident 10 GB が §6.5 の予算式による導出でない**（D-24 / HIGH）
   Rev.02 §6.5 は `VRAM総量 = runtime + KV cache + activation + ring buffer + resident weight + 安全余白`
   での計算を求め、「resident 10GB だけ決める」ことを名指しで禁じている。
   現状の 10 GB は `docs/EXPERIMENTS.md` 2b.3 の手計算（ring 1 GB / KV+activation+context 1 GB の
   「想定」）に基づく設定値であり、コード上に予算計算器は存在しない
   （`kv_cache` / `activation` / `safety_margin` の識別子が `include/` `src/` に 1 つも無い）。

3. **DERIVED 値が実測と表記されている箇所がある**（D-34 / HIGH）
   `docs/KNOWN_LIMITATIONS.md:237` および `README.md:221` の「実測 0.6017 byte/param」。
   除数の公称パラメータ数はモデル名由来の ASSUMED であり、正しい分類は DERIVED。

   さらに `ALSR-A0-REM-01` で、**分子の取り違え**が判明した。A0 初版（MP-119）は本値の scope を
   「`total_tensor_bytes` ÷ 公称パラメータ数」と記載していたが、
   19,853,132,208 ÷ 31e9 = **0.640424** であり 0.6017 にならない。
   0.6017 を与えるのは `text_bytes`（18,654,198,000 ÷ 31e9 = 0.6017483）である。

   | 分子 | byte/param | 405B 外挿 | 設計書 202.5 GB 比 | 5 tok/s 必要帯域 |
   |---|---:|---:|---:|---:|
   | `text_bytes`（現行の実体） | 0.6017 | 243.7 GB | +20% | 1168.5 GB/s |
   | `total_tensor_bytes` | 0.640424 | 259.4 GB | +28% | 1246.9 GB/s |

   vision tower（1,186,547,904 B）と projector（12,386,304 B）を含めるか否かで結論の数値が動く。
   その選択は文書側で一度も明示されていない。値そのものは変更せず、
   MP-119 の scope を `text_bytes` 基準へ訂正し、MP-120〜123 に `depends_on` と代替基準値を併記した。

---

## 6. Claim 抽出（Rev.02 §14 実施5）

Rev.02 §14 が指定する 8 語で全文検索した結果（`docs/ALSR_REQ_DESIGN.md` 原文内のヒットを除く）:

| 検索語 | ヒット行数 | （うち設計書 Rev.1.1 原文内） |
|---|---:|---:|
| 実測 | 71 | 13 |
| token/s | 38 | 7 |
| RTX | 37 | 17 |
| PCIe | 54 | 13 |
| resident | 117 | 0 |
| GREEN | 3 | 7 |
| 完了 | 10 | 7 |
| 改善 | 2 | 3 |

ここから性能・成立性に関する主張を抽出し `A0_CLAIM_LEDGER.json` に台帳化した。初版 18 件、
`ALSR-A0-REM-02` の A0-007 分割と `ALSR-A0-REM-04` の A0-017 分割により現在 **21 件**
（配列要素は 23。分割元 A0-007 / A0-017 は `status = SPLIT` で集計外）。件数は現物から再集計している。

`ALSR-A0-REM-01` で status を 4 つ追加し 6 件を再判定・1 件変更・3 件再分類。
`ALSR-A0-REM-02` で status を 1 つ追加し、1 件を再判定・1 件を 3 分割・1 件を系列分離した。
`ALSR-A0-REM-03` で A0-018 を再判定した（当該 RED 裁定の指摘 1 件目）。
`ALSR-A0-REM-04` で A0-017 を 2 分割し、summary の過大表現を訂正した。
`ALSR-A0-REM-05` で A0-017B の note から Evidence / Provenance を持たない runtime・OS 挙動の断定を削除した。
status は 1 件につき 1 つの主判定であり、同時に当てはまる判定は `status_secondary` に列挙する。

| status | 主判定 | 副次を含む | 該当 |
|---|---:|---:|---|
| SUPPORTED（証拠あり・provenance 整合） | **0** | **0** | — |
| SUPPORTED_WITH_SCOPE_NOTE（証拠はあるが表現が scope を超えて読める） | 3 | 3 | A0-005 / 007C / **018** |
| **UNSUPPORTED_AS_WORDED**（文言どおりの主張を裏付ける証拠が無い） | **8** | 8 | A0-002 / 003 / 006 / **007B** / 008 / 009 / 010 / **017B** |
| **CONDITIONAL_DERIVATION_ONLY**（明示された仮定の下でのみ成立する導出） | 0 | 9 | 副次のみ |
| **EVIDENCE_PRESENT_NOT_REEXECUTED**（証拠は在るが監査は再実行していない） | 4 | **11** | 主判定は A0-001 / **007A** / 015 / **017A** |
| **INTERNAL_HASH_CONSISTENT_ONLY**（hash 突合が同一 Pack 内で閉じている） | 0 | 1 | A0-015 の副次 |
| **TEST_SOURCE_PRESENT_NOT_EXECUTED**（テストコードは在るが実行結果が存在しない） | **1** | 1 | A0-004 |
| PROVENANCE_MISLABELED（出自の表記が実際と異なる） | 1 | 1 | A0-011 |
| NO_EVIDENCE_FILE（対応する証拠ファイルが `results/` に無い） | 2 | 6 | A0-012 / 016 |
| SELF_DECLARED_NOT_AUDITED（実装者の自己申告） | 2 | 2 | A0-013 / 014 |
| **主判定 合計** | **21** | — | — |

severity: HIGH 4 / MEDIUM 10 / LOW 1 / 未付与 6。

**無条件の SUPPORTED は 0 件である。** ただしこれを「証拠が主張を支えているものが 1 件も無い」と
要約してはならない（D-56）。`EVIDENCE_PRESENT_NOT_REEXECUTED` は *証拠が支えていない* という判定ではなく、
*証拠ファイルの存在と内容の一致は確認したが、本監査は再実行していない* という判定である。正しくは:

- 無条件の `SUPPORTED` は **0 件**。
- 監査によって**再現された**主張も **0 件**（本 Gate は benchmark 再実行を禁じられている）。
- `EVIDENCE_PRESENT_NOT_REEXECUTED` の **4 件**は、保存済み証拠の**内容確認まで**が済んでいる。
- 文言どおりに成立しないと判定したのは `UNSUPPORTED_AS_WORDED` の **8 件**である。

**MEASURED_RTX を要するが持たない主張**: A0-003 / A0-006 / **A0-007B** / A0-008 / A0-009 / A0-010 の 6 件。

`EVIDENCE_PRESENT_NOT_REEXECUTED` が副次込みで 11 件になるのは、本 Gate が benchmark 再実行を
禁じられており（Rev.02 §14）、`results/` 配下を evidence に持つ全ての主張が「存在と内容の確認のみ」の
状態にあるためである。**本台帳のどの判定も「監査が再現した」という意味を持たない。**

### 6.1 特に重い主張

| ID | 主張 | 判定 | 問題 |
|---|---|---|---|
| A0-003 | 「PCIe 3.0 で 1.63 tok/s、PCIe 4.0 で 3.39 tok/s が転送だけで**確定する**」 | UNSUPPORTED_AS_WORDED / HIGH | PCIe の実測は 0 件。入力 12 / 25 GB/s は ASSUMED。Rev.02 §4.2 の禁止推定に該当し、§5.2 の禁止表現に該当する断定形 |
| A0-008 | 「**常駐率が効く。**31B で 1.329 → 2.864 tok/s（2.16 倍）」 | UNSUPPORTED_AS_WORDED / HIGH | 分子・分母とも EMULATED。2.16 倍はシミュレータ上ほぼ `per_token_bytes` の比（18.654/8.654 = 2.155）で決まる算術的帰結。Rev.02 §11.2 が禁じる実性能認定として読める |
| A0-011 | 「**実測** 0.6017 byte/param から外挿すると 405B は 243.7 GB」 | PROVENANCE_MISLABELED / HIGH | 出自の誤表記（実測ではなく DERIVED）+ 分子の未定義（`text_bytes` か総テンソルか）+ 証拠ファイル不在 |
| A0-013 | 「Phase 0-2 **完了**」 | SELF_DECLARED_NOT_AUDITED / HIGH | Rev.02 固定7 / §2.3 により自己発行の完了判定。§0 で既存成果物は一括 UNVERIFIED |

### 6.2 再判定の内容（`ALSR-A0-REM-01`）

| ID | 初版 | 改訂後 | 理由の要点 |
|---|---|---|---|
| A0-002 | SUPPORTED_WITH_SCOPE_NOTE | **UNSUPPORTED_AS_WORDED** | 「113〜118 GB/s を出している」は帯域の測定結果として読めるが、帯域は一度も測定されていない。token/s を S_stream 上限で割った商であり、下限ケース仮定では 100.38 GB/s になる |
| A0-003 | SUPPORTED_WITH_SCOPE_NOTE | **UNSUPPORTED_AS_WORDED** | 「PCIe 3.0 / 4.0 で〜が確定する」は実在の PCIe 世代についての断定。MEASURED_RTX = 0 |
| A0-006 | SUPPORTED_WITH_SCOPE_NOTE | **UNSUPPORTED_AS_WORDED** | 「線速で回せない」は実行環境（Ryzen 7 5700X）の結論だが、当該機の CRC32C も PCIe 実効帯域も未測定。左辺 11.77 GB/s のみが確定 |
| A0-008 | SUPPORTED_WITH_SCOPE_NOTE | **UNSUPPORTED_AS_WORDED** | 上表参照。severity も MEDIUM → HIGH |
| A0-009 | SUPPORTED_WITH_SCOPE_NOTE | **UNSUPPORTED_AS_WORDED** | 「PCIe 律速の領域では効かない」は物理的状況一般への一般化。確認されたのは、Chunk サイズが平均帯域に影響しない構造のシミュレータが構造どおりの null result を返したことのみ。同じ sweep の壁時計は 1.395 s → 8.318 s と 6 倍近く動いている |
| A0-010 | SUPPORTED_WITH_SCOPE_NOTE | **UNSUPPORTED_AS_WORDED** | 「VRAM 不足で測定不能」は観測ではなく、シミュレータの事前チェックが `ERROR[invalid]` を返して停止した結果。GPU は関与しておらず測定は試みられていない |
| A0-011 | NO_EVIDENCE_FILE | **PROVENANCE_MISLABELED** | 出自の誤表記が最も重い欠落であるため主判定を移した。証拠不在は `status_secondary` に残置 |
| A0-001 | SUPPORTED | **EVIDENCE_PRESENT_NOT_REEXECUTED** | GGUF 解析器を再実行していない。確認したのは記録されたファイルの内容 |
| A0-015 | SUPPORTED | **EVIDENCE_PRESENT_NOT_REEXECUTED** | Run を再実行していない。副次 INTERNAL_HASH_CONSISTENT_ONLY（§7.1） |
| A0-017 | SUPPORTED | **EVIDENCE_PRESENT_NOT_REEXECUTED** | 環境収集を再実行していない。当該ファイルは `git.commit` が未解決のまま（D-36） |

### 6.2.1 再判定・分割の内容（`ALSR-A0-REM-02`）

3 件はいずれも **証拠の粒度と主張の粒度が合っていなかった** ことによる誤りである。

**A0-002 — 数値系列の分離**

`README.md:203` の「113〜118 GB/s」は、**2 つの異なる母集団の値を 1 つの区間に混ぜていた**。

| 系列 | 入力の出自 | 母集団 | n | token/s | 逆算帯域 [GB/s] | 台帳 |
|---|---|---|---:|---|---|---|
| A | USER_OBSERVED | ターミナル画面からの転記 | 2 | 15.35 / 16.49 | **113.06 / 121.46** | MP-094 / 095 |
| B | MEASURED_M5 | `scripts/baseline_m5.sh` warm-up 除外 | 5 | 14.9408 / 15.9653 / 16.2179 | 110.05 / **117.59** / 119.45 | MP-125 |

- 区間の下端 **113 は系列 A の最小値**、上端 **118 は系列 B の中央値の丸め**。
- 系列 A の最大値 **121.46** と系列 B の最小値 **110.05** は、どちらも区間の外に落ちている。
- すなわち 113〜118 は **どちらの系列の範囲でもない**。両系列を合わせた包絡は 110.05〜121.46。
- **いずれも `provenance_final` は DERIVED** であり、帯域計を当てた測定は 1 件も存在しない。
- S_stream 下限を仮定すると包絡は **97.71〜107.84 GB/s** へ移り、上限仮定の区間と**重ならない**（DR-0005 / U-06）。
- 系列 B の 117.59 は初版の台帳に項目が無かった（`ALSR-A0-REM-02` で MP-125 として追加）。

**A0-004 — `SUPPORTED_WITH_SCOPE_NOTE` → `TEST_SOURCE_PRESENT_NOT_EXECUTED`**

初版は `evidence_verified = true` としていたが、これは誤りである。確認できたのは
`tests/stress/test_stress.cpp` が存在し、そこに 8 つの物理的主張がテストとして書かれていることだけで、
**そのテストが通ることは確認していない**。A0 / REM-01 〜 REM-05 はいずれも test を実行しておらず、
過去の実行結果も `results/` に Pack として残っていない（D-42）。`evidence_verified` を **false** に訂正した。
主張の内容自体は数理的に自明だが、**A0 は自明さを証拠の代わりにしない**。

`EVIDENCE_PRESENT_NOT_REEXECUTED` を使わないのは、同ステータスが「参照できる実行結果が存在する」ことを
前提にするのに対し、本件はその実行結果自体が存在しないためである。

**A0-007 — 3 主張への分割**

必要な証拠の種類が異なる 3 つを 1 件にまとめていたため、**算術で足りる A の成立が、
MEASURED_RTX を要する B の成立のように読めていた**。

| ID | 主張 | 必要な証拠 | 判定 |
|---|---|---|---|
| **A0-007A** | 総テンソル 6.86 GiB は公称 VRAM 12 GB より小さい | MEASURED_M5 + DECLARED_SPEC（算術） | EVIDENCE_PRESENT_NOT_REEXECUTED |
| **A0-007B** | 実行時に VRAM へ**収まる** | **MEASURED_RTX**（実効利用可能容量） | **UNSUPPORTED_AS_WORDED** / MEDIUM |
| **A0-007C** | このモデルで ALSR の優位は語れない。検証対象は 31B へ | 設計判断（DR-0009） | SUPPORTED_WITH_SCOPE_NOTE |

B が未支持なのは、公称 12 GB から差し引くべき CUDA コンテキスト・ドライバ予約・KV キャッシュ・
アクティベーション・ring buffer が**いずれも未取得**であり、§6.5 の予算計算器も未実装（D-24）だからである。
6.86 GiB と 12 GB の差が大きく反転の見込みは小さいが、**見込みは証拠ではない**。

C は B を理由に挙げているため理由は未確定だが、C は「語ってはいけない」という**制限**であるため、
B が誤っていた場合に生じるのは過剰に保守的だったという結果にとどまり、誤った主張は生じない。

### 6.2.2 再判定の内容（`ALSR-A0-REM-03`）— RED 裁定の指摘 1 件目

**A0-018 — `SUPPORTED` → `SUPPORTED_WITH_SCOPE_NOTE`**（D-48）

A0-018 は `docs/KNOWN_LIMITATIONS.md:18-27` の自己制限宣言
（CUDA / PCIe の性能・Pinned Memory の効果・対既存方式の優位・出力品質は主張しない）を扱う。
初版の note は「**この宣言に正面から違反する記述は見つからなかった**」と書いていた。

これは**同じ台帳の他の判定と矛盾している**。

| 台帳の判定 | 内容 |
|---|---|
| A0-003 | 「PCIe 3.0 で 1.63 tok/s、PCIe 4.0 で 3.39 tok/s が転送だけで確定する」（`README.md:205-207`） |
| A0-003 の status | **UNSUPPORTED_AS_WORDED / HIGH** |
| D-22 | Rev.02 §5.2 の**禁止表現に該当**すると記録 |

A0-003 は、宣言が名指しで禁じている「**CUDA / PCIe の性能**」の主張そのものである。
すなわち **宣言は存在するが、遵守されていない**。成立するのは「当該の自己制限が明記されている」
ところまでであり、「その制限が守られている」ことではない。両者を混同していた。

`compliance` フィールドで宣言の存在（`declaration_present: true`）と遵守（`declaration_observed: false`）を
分離し、矛盾する記述を削除した。

**この再判定により、本台帳に無条件の `SUPPORTED` は 1 件も残らない。**

### 6.2.3 再判定・分割の内容（`ALSR-A0-REM-04` / `ALSR-A0-REM-05`）

**A0-017 — 2 主張への分割**（D-53）

`A0-007` と**同型**の粒度不整合である。静的なバイト数比較と実行可否を 1 件にまとめていたため、
**算術で確定する A の成立が、実行系の観測を要する B の成立のように読めていた**。

| ID | 主張 | 必要な証拠 | 判定 |
|---|---|---|---|
| **A0-017A** | 総テンソル 19,853,132,208 B > M5 物理 RAM 17,179,869,184 B | MEASURED_M5 どうしの算術 | EVIDENCE_PRESENT_NOT_REEXECUTED |
| **A0-017B** | Ollama の実推論 baseline を**取得できない** | 実行の試行記録 / runtime 挙動 | **UNSUPPORTED_AS_WORDED** + NO_EVIDENCE_FILE / MEDIUM |

B が未支持である理由は、`results/` に実行を試みた記録（失敗ログ・エラー出力・中断記録）がなく、
runtime 挙動も観測していないためである。A0-017A の静的な容量比較は、実行不能を示す実行証拠ではない。
`ALSR-A0-REM-04` の note は、Evidence / Provenance を持たない runtime・OS 挙動まで事実として記載していたため、
`ALSR-A0-REM-05` で削除した（D-58）。

**「取得していない」は事実だが、「取得できない」は示されていない。**

A0-017A の evidence には、初版に欠けていた `manifest_gemma4_31b.json`（比較の左辺）を追加した。

### 6.3 適切に自己制限されている点

以下は Rev.02 の方針と整合しており、A0 として問題を認めなかった。

- `docs/KNOWN_LIMITATIONS.md` §2 が「CUDA / PCIe 性能」「Pinned Memory の効果」
  「ALSR が既存方式より速いこと」「出力品質」を**主張してはいけないもの**として明示している。
- `gemma4:12b` が RTX 3060 の VRAM に収まるため ALSR の検証対象として不適格である、という
  自己に不利な事実を DR-0009 で正式に記録し、検証対象を 31B へ移している。
- 合成 Run（run03）の `per_token_bytes` を 0 にし、実モデルの S_stream を合成データへ
  適用しない措置がとられている（commit `027c0aa`）。
- `docs/CONFORMANCE.md` が自己監査で 8 件の未達を自ら検出・記録している。

---

## 7. 成果物の所在確認（Rev.02 §8 Gate A0 GREEN条件）

`A0_EVIDENCE_INDEX.json` に全 150 ファイルのパス・カテゴリ・バイト数・SHA-256・Git 追跡状態を記録した。

| カテゴリ | ファイル数 |
|---|---:|
| EVIDENCE（`results/`） | 80 |
| SOURCE（`src/` `include/` `backends/` `tests/` `CMakeLists.txt`） | 48 |
| TOOLING（`scripts/` `tools/`） | 6 |
| CONFIG（`configs/`） | 6 |
| DOCUMENT（`docs/`） | 5 |
| OTHER（`README.md` `LICENSE` `.gitignore` `third_party/.gitkeep` `third_party/README.md`） | 5 |

Git 追跡 71 / 追跡外 79。追跡 71 件と索引の `git_tracked=true` 71 件は集合として完全に一致する
（片側のみに存在するパスは 0 件）。突合の記録は `A0_EVIDENCE_INDEX.json:tracked_file_reconciliation`。

### 7.1 整合検証（Rev.02 §10.2 integrity_failures 相当）

6 Run すべてについて `metrics.jsonl` の SHA-256 を再計算し、`summary.json` に記録された値と突合した。

| Run | 結果 |
|---|---|
| `run01-gemma4-fixed-pcie4-29057ced221a` | 一致 |
| `run02-gemma4-adaptive-pcie3-fd058c7ee687` | 一致 |
| `run03-synthetic-adaptive-healthy-cc444f432f2d` | 一致 |
| `run04-31b-resident10-pcie4-596595facb61` | 一致 |
| `run05-31b-resident10-pcie3-c851259f505a` | 一致 |
| `run06-31b-resident0-pcie4-ef64f863d9ba` | 一致 |

**integrity failures = 0。すなわち、現在の Pack 内で内部 hash 不一致は 0 件である。**

この結果が示す範囲はそこまでである。突合した 2 つの値 — `summary.json` の記録値と
`metrics.jsonl` の再計算値 — はどちらも同じ Pack の中にあり、**外部アンカーが存在しない**。

- remote 無し・署名無し・独立したタイムスタンプ無し・第三者による保管無し。
- `results/` は `.gitignore:22` により非追跡であり、**Git の履歴もアンカーにならない**。
- 両者が揃って差し替えられた場合、この突合では検出できない。

したがって本結果から「事後に書き換えられていない」を導くことはできない。
初版はここを「事後に書き換えられた形跡は見つからなかった」と記載しており、
検証範囲を超えた結論であった（D-44。`ALSR-A0-REM-01` で是正）。
台帳上の扱いは `INTERNAL_HASH_CONSISTENT_ONLY`（A0-015 の副次判定）。

また 6 Run はいずれも**再実行していない**（Rev.02 §14 が benchmark 再実行を禁じている）。
確認したのは保存済みファイルの内容であり、ゲートの PASS を監査が再現したわけではない。

### 7.1.1 `results/` の連続性 — 証明できる範囲と未証明範囲（`ALSR-A0-REM-02` item 7）

§2.1 のパス移行をまたいで `results/` の内容が保たれているかを確認した。
本索引は**移行前**（2026-08-02 18:18 JST、旧パス）に全ファイルの SHA-256 を採取しており、
これを**移行後**のパスで再計算して全件突合できる。

#### 証明できる範囲

| 対象 | 結果 |
|---|---|
| 索引 150 件の SHA-256 再計算 | **一致 150 / 不一致 0 / 欠落 0** |
| うち移行前に採取済みの 148 件 | 全件 byte-identical |
| `results/` 配下 80 件 | **全件 byte-identical** |
| `results/` にディスク上のみ存在するファイル | **0 件** |
| 索引にあってディスクに無いファイル | **0 件** |
| HEAD / 追跡ファイル集合 | 移行前後で一致（§2.1） |
| `git fsck --full --strict` | 異常報告なし |

**以上が本突合の示す範囲である。**

> **時間の境界について（`ALSR-A0-REM-04` / D-54）**
> `ALSR-A0-REM-03` までは「証明できるのは 18:18 以降に限られる」と書いていたが、これは
> **18:18 以降なら連続性が証明できる**かのように読める。実際に証明できるのは
> 「採取時点の記録値と再計算時点の内容が一致した」という **2 点の事実だけ**である。
> その 2 点の間についても、18:18 以前についても、何も証明していない。
> 18:18 が持つ意味は「これより前には突合の相手すら存在しない」という**下限**であって、
> 「これより後は保証される」という**保証ではない**。

#### この結果から言えないこと（`ALSR-A0-REM-03` / D-52）

`ALSR-A0-REM-02` はここに
**「パス移行によって `results/` の内容が失われた・混ざったことは起きていない」** と書いていた。
これは 2 点で検証範囲を超えていたため削除した。

1. **事象についての主張になっている。** 観測したのは移行前後の 2 つの**状態**だけであり、
   移行そのものは観測していない。移行の最中に何が起きたかは、この突合からは言えない。
2. **「失われていない」は索引に載る範囲でしか言えない。** 18:18 以前に消えたファイルは索引に無く、
   したがって突合にも現れない。索引の網羅性が検証済みなのは**追跡ファイルについてのみ**であり（§7.2）、
   非追跡の `results/` は「18:18 時点で存在した 80 件」以上の母集団を確定する手段が無い。

これは **D-44（「事後に書き換えられた形跡なし」）と同型のスコープ超過を、同じ Mission 群の中で
再発させたもの**である。突合の結果として言えるのは、上表に書いた内容だけである。

#### 未証明範囲

| # | 証明できないこと | 理由 |
|---|---|---|
| 1 | 18:18 より前の状態 | Run Pack が作られた 13:49〜14:32 の区間は索引の外側。索引がこの区間の**最も古いアンカー**であり、それ以前を遡る手段が無い |
| 2 | 18:18 と再計算時点の**間**に変更が無かったこと | 2 点で hash が一致することは、その間の不変を意味しない。書き換えて元へ戻した場合は区別できない |
| 3 | 旧パスと新パスが一時的に**別実体として併存**していなかったこと | 本監査は常に 1 つのファイルシステムビューしか観測していない。片方だけが変化していた可能性を排除できない |
| 4 | 索引自体の真正性 | 索引は実装担当が同一セッションで生成した**未追跡ファイル**であり、署名も独立タイムスタンプも remote も無い（§7.1 / D-44 と同じ限界） |
| 5 | iCloud の evict / re-materialize が内容を変えていないこと | **18:18 以降についても否定できない。** 2 時点の hash 一致は、その間の変更・復元の不在を意味しない（#2 と同じ理由）。時間帯を切っても解消しない（D-54） |
| 6 | `results/` の母集団そのもの | 非追跡であるため、18:18 時点の 80 件が「あるべき全部」だったのかを確認する手段が無い |

**残存リスク**: `results/` 80 件のうち追跡下にあるのは `.gitkeep` の 1 件のみで、
Evidence 本体 79 件は `.gitignore` により非追跡であり、Git のアンカーを持たない（D-55）。
今回のパス移行は、この脆弱性が実際に顕在化し得ることを示した事例である（U-07 / U-11）。
全件一致したという事実の記録であって、仕組みで守られていることの記録ではない。

### 7.2 索引の網羅性（追跡 71 件との突合）

初版の索引は追跡 71 件のうち **69 件**しか収録しておらず、差 2 件があった。
特定した欠落は以下である。

| パス | バイト数 | SHA-256 | 欠落の原因 |
|---|---:|---|---|
| `third_party/.gitkeep` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 索引除外規則が `third_party/` をディレクトリごと対象外にしていた。除外の意図は FetchContent 取得物（`third_party/_fetch/` の Catch2 v3.15.3）であり、直下の追跡ファイルは対象外にすべきではなかった |
| `third_party/README.md` | 690 | `533e91012779256a1a1f8a479bf3155a301d83589d602a2320a8c96004d3c3cc` | 同上。`third_party/` の運用方針を記述した文書であり取得物ではない |

`ALSR-A0-REM-01` で 2 件を索引へ追加し、除外規則を `third_party/_fetch/` へ限定した。
索引 148 → **150**、`git_tracked` 69 → **71**、差 **0**（D-43）。

なお本突合が示すのは索引の網羅性のみである。索引 150 件のうち追跡外は 79 件で、
その **79 件はすべて `results/` 配下**である（`results/` 索引 80 件 − 追跡下の `.gitkeep` 1 件）。
`ALSR-A0-REM-03` までは追跡外 79 件に `.claude/settings.local.json` を含めて記述していたが、
同ファイルは索引対象外であり 79 件には入らない（D-55）。
これらが Git のアンカーを持たずディスク上にのみ存在する状態は変わっていない（§3.2 / U-07）。

### 7.3 参照が切れている証拠

`docs/DECISIONS.md:173` および `:224` は run01 の Run ID を
`run01-gemma4-fixed-pcie4-cc2b89940ac3` と記載しているが、実在するディレクトリは
`run01-gemma4-fixed-pcie4-29057ced221a` である。設定変更により `run_id` が変わった結果と
考えられるが、**A0 では推測で埋めない**（D-41）。

---

## 8. Scope 逸脱の記録（Rev.02 §8 Gate A0 GREEN条件）

本 Mission における scope 逸脱: **無し**。

§0 に記載のとおり、Rev.02 §14 の禁止事項 7 項目をすべて遵守し、
追加したファイルは `docs/recovery/` 配下の成果物 5 点のみである。

既存成果物側の scope 逸脱は Deviation Matrix に 60 件として記録した
（うち 17 件は A0 成果物自身の欠陥であり、REM-01 〜 REM-05 で是正済み。§4.1 に全 ID）。
そのうち Rev.02 の固定事項に正面から抵触するものは以下 3 件。

| 固定事項 | 抵触内容 | Deviation |
|---|---|---|
| 固定4「Adaptive Scheduler は固定方式・実GPU計測後」 | adaptive Scheduler が実装済みで run02 / run05 に使用されている | D-27 |
| 固定7「実装者の完了報告は監査結果ではない」 | README が「Phase 0-2 完了」と記載、CONFORMANCE が多数の「適合」を自己判定 | D-37 |
| 固定6「全性能Claimは 値→Provenance→Evidence→Scope を持つ」 | 405B 外挿値に Evidence が無い | D-32 |

いずれも **Rev.02 発行前に行われた作業の結果**であり、A0 では是正せず記録のみ行った
（是正は Gate 進行にあたり §14 で禁止されている）。

---

## 9. 本 Mission による変更ファイル一覧

| ファイル | 種別 | 規模 |
|---|---|---|
| `docs/recovery/A0_INVENTORY.md` | 新規 | （本ファイル） |
| `docs/recovery/A0_DEVIATION_MATRIX.csv` | 新規 | 60 行 + ヘッダ |
| `docs/recovery/A0_METRIC_PROVENANCE.json` | 新規 | 125 metrics + 13 unknown |
| `docs/recovery/A0_EVIDENCE_INDEX.json` | 新規 | 150 files |
| `docs/recovery/A0_CLAIM_LEDGER.json` | 新規 | 21 claims（配列要素 23。うち 2 件は `SPLIT` で集計外） |

`docs/recovery/` の外にあるファイルの変更: **0 件**。削除: **0 件**。
上表の 5 点は `ALSR-A0-REM-01` 〜 `ALSR-A0-REM-05` で改訂されている（§12）。改訂もこの 5 点の中で閉じており、
コード・設定・既存文書・`results/` はいずれも変更していない。

各改訂の統一 diff はリポジトリ外の成果物として提出しており、その SHA-256 と適用検証の結果は
§12.4 に記録している（`docs/recovery/` を A0 成果物 5 点のみに保つため、リポジトリには置いていない）。

---

## 10. 未確認事項（A0 で確定できなかったもの）

| # | 事項 | 理由 |
|---|---|---|
| U-01 | Rev.02 をリポジトリへ配置するか、配置先はどこか | A0 の成果物に含まれず、`docs/ALSR_REQ_DESIGN.md`（Rev.1.1）の扱いも未指示（D-38） |
| U-02 | 既に実装済みの adaptive Scheduler を今後の Run で使ってよいか | 固定4 に抵触するが、既存実装の撤去は Gate 進行にあたる（D-27） |
| U-03 | `results/` を `evidence/<gate>/<run_id>/` へ再編するか、現状維持か | 構造変更は A0 の範囲外（D-02） |
| U-04 | 旧 Rev.1.1 由来の `docs/CONFORMANCE.md` `EXPERIMENTS.md` `KNOWN_LIMITATIONS.md` の扱い | Rev.02 §13 に記載が無い。廃止・改訂・併存のいずれかは裁定を要する |
| U-05 | `run01` の Run ID 不一致（`cc2b89940ac3` vs `29057ced221a`）の原因 | 推測で埋めないため未確定（D-41） |
| U-06 | S_stream の上限 / 下限のどちらが実態か | 実行経路の計装が必要（DR-0005 の反証条件が未解消） |
| U-07 | `results/` の Evidence 本体 79 ファイルが Git 管理外である状態を継続するか | P-01 により現時点では意図的。追跡下にあるのは `.gitkeep` の 1 件のみ（D-55）。「お蔵入り」判断時の証拠保全方針は未定 |
| U-08 | Gate R0 の全 13 項目 | RTX 検証機に未アクセス |
| U-09 | 規模外挿の分子を `text_bytes` と総テンソルバイトのどちらに固定するか | 選択により 405B 外挿が 243.7 GB / 259.4 GB、設計書比が +20% / +28% と変わる。vision tower を Dense 外挿に含めるかは設計判断であり、Gate M3（`PHYSICAL_MODEL.md`）で式として固定すべき事項（D-34 / MP-119） |
| U-10 | `README.md` / `docs/KNOWN_LIMITATIONS.md` の「実測 0.6017 byte/param」等の誤表記を文書側で是正するか | 既存文書の書き換えは Gate 進行にあたり §14 で禁止されている。A0 / REM-01 では台帳への記録のみ行った（D-34 / D-37） |
| U-11 | Evidence Pack に外部アンカー（署名 / 独立タイムスタンプ / 追跡下への移動）を導入するか | 現状の hash 突合は同一 Pack 内で閉じており、改変の不在を示せない（§7.1 / D-44）。P-01 の運用方針と両立する形は未定。§2.1 のパス移行はこの脆弱性が顕在化し得ることを示した |
| U-12 | S_stream 上限 / 下限のどちらを採るかを固定しないまま逆算帯域を引用してよいか | 仮定の選択で包絡が 110.05〜121.46 GB/s と 97.71〜107.84 GB/s に分かれ、**重ならない**（§6.2.1 / D-46）。Gate M3 の `PHYSICAL_MODEL.md` で式として固定すべき事項。U-06 の未解消が数値の提示にまで及んでいる |
| U-13 | `tests/stress/` を実行して Evidence Pack を残すか | A0-004 の根拠はテストコードの存在のみで、実行結果は 1 件も存在しない（D-42 / TEST_SOURCE_PRESENT_NOT_EXECUTED）。test 実行は A0 の禁止事項であり、どの Gate で実施するかは裁定を要する |
| U-14 | リポジトリを iCloud 同期配下へ戻すか、ローカル実体のまま運用するか | §2.1 の移行は環境側で発生した。`results/` が非追跡である以上、同期の有無は証拠保全の方針そのものである（U-07 / U-11 と連動） |

---

## 11. 結論（実装担当としての報告。裁定ではない）

1. Rev.02 §14 の Mission `ALSR-A0-RECOVERY-AUDIT-01` を、禁止事項 7 項目を遵守して実行した。
   コード変更 0、build 無し、benchmark 再実行無し。改訂 Mission `ALSR-A0-REM-01` 〜 `ALSR-A0-REM-05` も
   同じ禁止事項を遵守した（REM-02 以降は `README.md` / `docs/KNOWN_LIMITATIONS.md` の修正禁止も明示的に遵守）。
2. 既存成果物は **commit 7 件 / 追跡 71 ファイル / 証拠 80 ファイル**。索引は 150 ファイル。
   `metrics.jsonl` の SHA-256 は 6 Run すべて `summary.json` の記録値と一致し、
   **現在の Pack 内で内部 hash 不一致は 0 件**である。外部アンカーが無いため、
   これは改変の不在を意味しない（§7.1 / D-44）。
3. 作業中にリポジトリ実体パスが iCloud 同期配下からローカルへ移行した（§2.1 / D-45）。
   移行前に採取した索引 148 件の SHA-256 を移行後に再計算し、**`results/` 80 件を含め全件一致**した。
   ただし証明できるのは「採取時点と再計算時点の内容が一致した」という 2 点の事実のみで、
   その間についても 18:18 以前についても何も証明していない（§7.1.1 / D-54）。
4. 全数値 125 項目に Provenance を付与し、**出力側（`provenance_final`）と入力側（`provenance_inputs`）を分離**した。
   **MEASURED_RTX は 0 件**であり、入力に MEASURED_RTX を持つ項目も 0 件である。
   **入力に ASSUMED を含む値は 65 項目**、**EMULATED 36 項目は例外なく入力に ASSUMED を含む**（§5.0）。
   設計書の 405B 見積り 202.5 GB も仕様値ではなく DERIVED である（D-49）。**比較の不動点は存在しない。**
5. 性能主張 21 件を台帳化した。**無条件の `SUPPORTED` は 0 件、監査が再現した主張も 0 件。**
   文言どおりには成立しない主張が 8 件（UNSUPPORTED_AS_WORDED）、出自の誤表記が 1 件、
   テストコードのみで実行結果が無いものが 1 件、証拠ファイル不在が 2 件、実装者の自己申告が 2 件、
   scope 注記付きが 3 件、**保存済み証拠の内容確認まで済んでいるもの**が 4 件（EVIDENCE_PRESENT_NOT_REEXECUTED）。
   最後の 4 件を「証拠が支えていない」と要約してはならない（D-56）。
6. Rev.02 に対する逸脱 60 件を記録した。HIGH 19 件の中心は
   **§5 Provenance のコード実装欠落 / §6.5 VRAM 予算計算器の欠落 / §6.6 Prefill-Decode 分離の欠落 /
   §7 Prior Art 未実施 / §8 Gate M3（数理モデル先行固定）未実施**である。
   すなわち **Rev.02 が「コードより先に固定せよ」と定めた層が、まだ存在しない**。
7. A0 成果物自身の欠陥を 5 回の改訂で計 17 件是正した（§4.1 に類型別の全 ID / §12）。
   REM-01 は「証拠が支える範囲より広く書いていた」型（過大な判定 6 件 / 過大な結論 1 件 / 索引欠落 2 ファイル）、
   REM-02 は「証拠の粒度と主張の粒度が合っていない」型（2 母集団の混合 / 未実行テストの検証済み扱い /
   必要証拠の異なる 3 主張の同居）、REM-03 は **RED 裁定を受けた残存 5 件**（主判定の自己矛盾 / Provenance 誤分類 /
   依存関係誤記 / 記録の欠落 / スコープ超過の**再発**）である。
   REM-04 は RED 裁定を受けた残存 5 件（粒度の再発 / 時間スコープの残存 / 件数誤記 /
   判定名の意味を超えた要約 / 類型集計の不整合）である。
   REM-05 は REM-04 の残存 3 件（未立証の runtime・OS 断定 / 改訂履歴の不整合 / 不変範囲の未限定）を是正した。
   **同型の誤りは 2 類型・3 再発関係として確認されている**（D-52 ← D-44、D-54 ← D-52、D-53 ← D-47）。
   1 度の是正では同じ型の誤りが消えない。本報告が **5 度目の改訂**であること自体が、
   成果物の信頼度に対する情報である。
8. 本報告は `COMPLETED_NOT_AUDITED` である。次の作業は実装担当から独立した現物監査と
   豆虎Gate の裁定であり、実装担当は次 Gate へ進まない（Rev.02 §2.3 / 固定7 / 固定8）。

---

## 12. 改訂記録

### 12.0 `ALSR-A0-REM-01`

| 項目 | 内容 |
|---|---|
| Mission | `ALSR-A0-REM-01` |
| 実施日 | 2026-08-02 JST |
| 対象 | `docs/recovery/` の A0 成果物 5 点のみ |
| 性質 | **A0 成果物自身の是正。再監査でも次 Gate 進行でもない。** |
| コード変更 | **0**。build / test / benchmark / model 操作もすべて実施していない |
| 状態 | `COMPLETED_NOT_AUDITED`（裁定は豆虎Gate） |

#### 12.1 変更内容（`ALSR-A0-REM-01`）

| # | 変更 | 対象ファイル | 参照 |
|---|---|---|---|
| 1 | status に `UNSUPPORTED_AS_WORDED` / `CONDITIONAL_DERIVATION_ONLY` / `EVIDENCE_PRESENT_NOT_REEXECUTED` / `INTERNAL_HASH_CONSISTENT_ONLY` を追加。主判定 1 つ + `status_secondary` の運用規則を明文化 | `A0_CLAIM_LEDGER.json` | §6 |
| 2 | A0-002 / 003 / 006 / 008 / 009 / 010 を再判定（いずれも SUPPORTED_WITH_SCOPE_NOTE → UNSUPPORTED_AS_WORDED）。A0-008 は severity も MEDIUM → HIGH | `A0_CLAIM_LEDGER.json` | §6.2 |
| 3 | A0-011 を NO_EVIDENCE_FILE → `PROVENANCE_MISLABELED` へ変更。summary の同項目は 0 → **1** | `A0_CLAIM_LEDGER.json` | §6.2 |
| 4 | MP-119 の scope を訂正（分子は `total_tensor_bytes` ではなく `text_bytes`）。分子・分母・代替基準（0.640424 byte/param）を構造化フィールドで明示 | `A0_METRIC_PROVENANCE.json` | §5.2-3 |
| 5 | MP-120 / 121 / 122 / 123 に `depends_on` を追加し scope を連動。総テンソル基準での代替値（259.4 GB / 1246.9 GB/s / 44.8 GB / 76.9 GB）を併記 | `A0_METRIC_PROVENANCE.json` | §5.2-3 |
| 6 | 追跡 71 件と索引 69 件の差 2 件を特定・追加。索引 148 → 150 / `git_tracked` 69 → 71 | `A0_EVIDENCE_INDEX.json` | §7.2 / D-43 |
| 7 | 「事後に書き換えられた形跡なし」を削除し「現在の Pack 内で内部 hash 不一致 0 件」へ変更。外部アンカー不在の限定を明記 | `A0_INVENTORY.md` / `A0_EVIDENCE_INDEX.json` | §7.1 / D-44 |
| 8 | 再実行していない主張（A0-001 / 015 / 017）を `EVIDENCE_PRESENT_NOT_REEXECUTED` へ再分類。副次を含め 11 件に付与 | `A0_CLAIM_LEDGER.json` | §6 |
| 9 | D-34 の severity を MEDIUM → HIGH。D-43 / D-44 を追加（42 → 44 行） | `A0_DEVIATION_MATRIX.csv` | §4.1 |

**既存の `A0_METRIC_PROVENANCE.metrics[].value` と参照元の性能・容量値は変更していない。**
0.6017 / 243.7 / 1168.5 / 42.1 / 72.2 / 113〜118 / 1.63 / 3.39 / 1.329 / 2.864 / 2.16 は
いずれも初版のままである。Claim / Deviation の件数・構造上の数値は、この不変範囲に含めない（D-60）。
変更したのは判定・scope・依存関係・集計・網羅性の記述である。

#### 12.2 本改訂で行っていないこと（`ALSR-A0-REM-01`）

| 禁止事項 | 遵守状況 |
|---|---|
| コード修正 | 実施していない |
| build | 実施していない |
| test | 実施していない（`tests/stress/` を含め 1 件も実行していない） |
| benchmark | 実施していない。`results/` 配下は 1 バイトも変更していない |
| model 操作 | 実施していない |
| 次 Gate 進行 | 実施していない。P0 / M1 / M2 / M3 / R0 のいずれにも着手していない |
| 既存文書の修正 | 実施していない。`README.md` / `docs/KNOWN_LIMITATIONS.md` / `docs/EXPERIMENTS.md` の記述は誤表記を指摘しただけで**書き換えていない**（是正は Gate 進行にあたる） |

すなわち **`docs/KNOWN_LIMITATIONS.md:237` の「実測 0.6017 byte/param」は現在も残っている。**
本改訂が変えたのは、それを台帳側で `PROVENANCE_MISLABELED` と判定し、分子が `text_bytes` であることを
記録したところまでである。文書側の是正には裁定を要する（U-04 / U-09）。

---

### 12.3 `ALSR-A0-REM-02`

| 項目 | 内容 |
|---|---|
| Mission | `ALSR-A0-REM-02` |
| 実施日 | 2026-08-02 JST |
| 対象 | `docs/recovery/` の A0 成果物 5 点のみ |
| 性質 | **A0 成果物自身の是正。再監査でも次 Gate 進行でもない。** |
| コード変更 | **0**。build / test / benchmark / model 操作 / `README.md`・`KNOWN_LIMITATIONS.md` 修正もすべて実施していない |
| 状態 | `COMPLETED_NOT_AUDITED`（裁定は豆虎Gate） |

#### 12.3.1 変更内容（`ALSR-A0-REM-02`）

| # | 指示 | 変更 | 対象ファイル | 参照 |
|---|---|---|---|---|
| 1 | A0-002 の系列分離 | `series_separation` を追加。系列 A（USER_OBSERVED / n=2）113.06・121.46 GB/s、系列 B（MEASURED_M5 / n=5）110.05〜119.45（中央値 117.59）。両系列とも `provenance_final` は DERIVED | `A0_CLAIM_LEDGER.json` / `A0_METRIC_PROVENANCE.json` | §6.2.1 |
| 2 | A0-004 の再判定 | `SUPPORTED_WITH_SCOPE_NOTE` → **`TEST_SOURCE_PRESENT_NOT_EXECUTED`**。`evidence_verified` を true → **false** に訂正。status に同名の値を追加 | `A0_CLAIM_LEDGER.json` | §6.2.1 |
| 3 | A0-007 の 3 分割 | **A0-007A**（公称容量比較 / EVIDENCE_PRESENT_NOT_REEXECUTED）、**A0-007B**（実 runtime 収容可否 / **UNSUPPORTED_AS_WORDED**）、**A0-007C**（検証対象としての適否 / SUPPORTED_WITH_SCOPE_NOTE）。分割元は `status = SPLIT` で集計外 | `A0_CLAIM_LEDGER.json` | §6.2.1 |
| 4 | `_method` の修正 | 値の所在で 3 系統に分離。(A) results 由来 114 / (B) docs 由来 11 / (C) 再計算値。初版の一括記述は (A) にしか当てはまらなかった | `A0_METRIC_PROVENANCE.json` | §5.0 |
| 5 | final / input provenance の分離 | 全 125 項目に `provenance_final` / `provenance_inputs` / `value_origin` を付与し `_provenance_model` で定義 | `A0_METRIC_PROVENANCE.json` | §5.0 |
| 6 | パス移行の記録 | 移行前後のパス・検出経緯・同一性確認を記録 | `A0_INVENTORY.md` / `A0_EVIDENCE_INDEX.json` | §2.1 / D-45 |
| 7 | `results/` 連続性の明記 | 索引 150 件を再計算し全件一致（`results/` 80 件を含む）。証明できる範囲と未証明範囲 5 項目を分離して記載 | `A0_INVENTORY.md` / `A0_EVIDENCE_INDEX.json` | §7.1.1 |
| 8 | diff の成果物化 | diff の SHA-256・改訂前後の 5 ファイル SHA-256・patch 往復検証結果を記録 | `A0_INVENTORY.md` | §12.4 |
| 付随 | 整合 | MP-125 追加（124 → 125 項目）、D-45 / D-46 / D-47 追加（44 → 47 行）、U-12 / U-13 / U-14 追加、集計の再計算 | 全 5 点 | §4.1 / §5 / §6 / §10 |

**既存 124 項目の `metrics[].value` / `unit` は変更していない。** `provenance` / `scope` はこの不変範囲に
含めない。MP-125（117.59 GB/s）は新規追加だが、これは
`docs/EXPERIMENTS.md:155` に既に存在する値を台帳へ収録したものであり、新しく計算した値ではない。

#### 12.3.2 本改訂で行っていないこと（`ALSR-A0-REM-02`）

| 禁止事項 | 遵守状況 |
|---|---|
| コード修正 | 実施していない |
| build | 実施していない |
| test | 実施していない。**A0-004 の再判定は `tests/stress/` を実行せずに行った**（実行していないこと自体が判定根拠である） |
| benchmark | 実施していない。`results/` 配下は 1 バイトも変更していない（§7.1.1 の再計算は読み取りのみ） |
| model 操作 | 実施していない |
| `README.md` / `KNOWN_LIMITATIONS.md` 修正 | 実施していない。D-46（「113〜118 GB/s」）も指摘のみで**書き換えていない** |
| 次 Gate 進行 | 実施していない。P0 / M1 / M2 / M3 / R0 のいずれにも着手していない |

`README.md:203` の「113〜118 GB/s」、`docs/KNOWN_LIMITATIONS.md:237` の「実測 0.6017 byte/param」は
**いずれも現在も文書に残っている。** 本改訂が行ったのは台帳側での判定と記録までである。

### 12.4 改訂 diff の記録（`ALSR-A0-REM-02` item 8）

統一 diff は各改訂の成果物として提出している。`docs/recovery/` を A0 成果物 5 点のみに保つため
（各 Mission の「対象」指定）、リポジトリ内には配置していない。

#### 12.4.1 5 ファイルの SHA-256

| ファイル | 初版 | REM-01 後 | REM-02 後 | REM-03 後（= REM-04 の改訂前） | REM-04 後（= REM-05 の改訂前） | REM-05 後 |
|---|---|---|---|---|---|---|
| `A0_CLAIM_LEDGER.json` | `e4f1c60b83ae…f11392` | `ac13f746d62e…f398d9` | `b706c447fd31934d309f151a23157dd9fd671250d57e18fd778f8f036433babc` | `aa5752251de14b4d3cadd8c2d1530387d37f76866b54265920c621afb19e8cfb` | `65a7e24d717ca62ccbf41e55d36349cd813b4092e6c2447e7b810f2f296e7791` | `08f8a1a33f7dc080c1e0dc30873b392112dcd547fb6da87d9c37c4ad55e8fc93` |
| `A0_METRIC_PROVENANCE.json` | `d1cff2605e80…7bc05a` | `a1c4acc82108…9edb5a` | `ff144670594f0f4e986bf8054e975952e202ed10127eb39d19d5514b345b9872` | `48c55dffb694255ab2089db3b60a4473e069d5fef4c6229ba31720e86f5aed51` | `48c55dffb694255ab2089db3b60a4473e069d5fef4c6229ba31720e86f5aed51` | `48c55dffb694255ab2089db3b60a4473e069d5fef4c6229ba31720e86f5aed51`（未変更） |
| `A0_EVIDENCE_INDEX.json` | `63fc653cfd89…097934` | `1303a2853c64…9544c8` | `cebc5d104d31e832fa2e6dc3de90e1b9a8bc3df09ae6fdcd56fd1b0f34bfbb87` | `0781e68db66f69eae5593de118b7f35a0d3f66c459bec7c49e41608165805af6` | `0d374d5369d68d38e026ef4d653ac8cead4a34e430074362aaa404baa025184d` | `0d374d5369d68d38e026ef4d653ac8cead4a34e430074362aaa404baa025184d`（未変更） |
| `A0_DEVIATION_MATRIX.csv` | `4e546df481c4…f48531` | `ced731dbc1aa…0cc715` | `8b6f4affd406606f940359ea1baf355f1ee0e5a7245a083ff93601e01ec51e9e` | `7b7db1f7d899ff0e2451117432f36069d445d025bb151238fad0253e21931973` | `d1115e1fb139d293622da59327249bce7187c43ab66a0e0a2c8c45cf52fe759e` | `994cfb581b596e6522d415b1c61801ff62c0c2d65a591e9cca2429de041bc91c` |
| `A0_INVENTORY.md` | `0bd8ac653732…f2f25` | `17e61bd1e7f3…c311062` | `f76ddecf21c0…8a2adf47` | `a96b26f95d597bfa1c1abfd36359820dea9eddb39911ec7cf9e563d1aeae1dd3` | `5dbf17068c389af41107dfb10810fd347e0b1de6b29f3287f0677dbb30e6e735` | **自己参照のため記載できない**（下記） |

初版・REM-01 後の 64 桁は §12.4 の初出（REM-02 報告）に記載済み。ここでは可読性のため短縮している。

**自己参照の制約**: 本ファイル（`A0_INVENTORY.md`）自身の**最新**の SHA-256 と、
その改訂の diff の SHA-256 は、**この文書の中に書けない**。値を書き込めば内容が変わり、
書いた値が古くなるためである（diff も本ファイルの差分を含むため同じ制約を受ける）。
両者は **各改訂の提出報告に記載する**。検証する側は、受け取った `A0_INVENTORY.md` を自分で
`shasum -a 256` にかけ、報告値と突き合わせること。
1 つ前の版までは確定しているため、上表の「REM-04 後」列には本ファイルの 64 桁も記載している。
他の 4 ファイルはこの制約を受けない。

各 diff の SHA-256 は、それぞれの改訂時点の報告に属する。

| 改訂 | diff ファイル | SHA-256 |
|---|---|---|
| `ALSR-A0-REM-01` | `ALSR-A0-REM-01.diff` | `497521557f358461f279fb694343035c85a878663237847a59f903cbb7aad89d` |
| `ALSR-A0-REM-02` | `ALSR-A0-REM-02.diff` | `dcc8594b9d83f4cd46c0373397583328b52d6e3c4fdaec841ac24da672c68b4e` |
| `ALSR-A0-REM-03` | `ALSR-A0-REM-03.diff` | `c2d0559504af9f4bc9b97d595c144d517f47eec9ffae73bb487201fc86404d6e`（**監査未受領 / NOT_VERIFIED** — §12.6.2） |
| `ALSR-A0-REM-04` | `ALSR-A0-REM-04.diff` | 本ファイルの差分を含むため記載不可。提出報告に記載 |
| `ALSR-A0-REM-05` | `ALSR-A0-REM-05.diff` | 本ファイルの差分を含むため記載不可。完了報告に記載 |

#### 12.4.2 patch 往復検証

各 diff について、改訂前の 5 ファイルへ `patch -p1` を適用すると改訂後の 5 ファイルと
SHA-256 が一致することを確認した。

| 改訂 | 検証手順 | 結果 |
|---|---|---|
| `ALSR-A0-REM-01` | 改訂前を編集の逆適用で再構成 → 作業開始前に採取した SHA-256 と 5 件一致 → diff を適用 → 改訂後と 5 件一致 | **往復一致** |
| `ALSR-A0-REM-02` | 改訂前（= REM-01 後）を作業前にコピー保存 → SHA-256 が REM-01 報告値と 5 件一致 → diff を適用 → 改訂後と 5 件一致 | **往復一致** |
| `ALSR-A0-REM-03` | 改訂前（= REM-02 後）を作業前にコピー保存 → SHA-256 が REM-02 報告値と 5 件一致 → diff を適用 → 改訂後と 5 件一致 | **往復一致**（ただし diff は監査未受領 / NOT_VERIFIED） |
| `ALSR-A0-REM-04` | 改訂前（= REM-03 後）を作業前にコピー保存 → SHA-256 が REM-03 報告値と 5 件一致 → diff を適用 → 改訂後と 5 件一致 | **往復一致** |
| `ALSR-A0-REM-05` | 改訂前（= REM-04 後）を作業前にコピー保存 → SHA-256 が REM-04 報告値と 5 件一致。diff 適用後の照合結果は完了報告に記載 | **完了報告に記載** |

この検証が示すのは diff の完全性（diff が改訂の全内容を含み、過不足が無いこと）である。
改訂内容の正しさを示すものではない。

---

### 12.5 `ALSR-A0-REM-03`

| 項目 | 内容 |
|---|---|
| Mission | `ALSR-A0-REM-03` |
| 実施日 | 2026-08-02 JST |
| 契機 | **`ALSR-A0-REM-02` が RED 裁定**。指摘 5 件の是正 |
| 対象 | `docs/recovery/` の A0 成果物 5 点のみ |
| 性質 | **A0 成果物自身の是正。再監査でも次 Gate 進行でもない。** |
| コード変更 | **0**。build / test / benchmark / model 操作 / `README.md`・`KNOWN_LIMITATIONS.md` 修正もすべて実施していない |
| 状態 | `COMPLETED_NOT_AUDITED`（裁定は豆虎Gate） |

#### 12.5.1 指摘と是正の対応

| # | 指摘 | 特定した対象 | 是正 | Deviation |
|---|---|---|---|---|
| 1 | Claim 主判定の残存誤り 1 件 | **A0-018**（`SUPPORTED`）。note の「この宣言に正面から違反する記述は見つからなかった」が、同台帳の A0-003 判定（PCIe 性能の主張 / UNSUPPORTED_AS_WORDED / HIGH / D-22）と矛盾 | `SUPPORTED_WITH_SCOPE_NOTE` へ再判定。`compliance` で宣言の存在と遵守を分離。**無条件の SUPPORTED は 0 件に** | D-48 |
| 2 | Provenance 誤分類 1 件 | **MP-118**（405B 理想 4bit 202.5 GB）が `DECLARED_SPEC`。仕様値ではなく 405e9（ASSUMED）× 0.5 byte/param（ASSUMED）の算術結果 | `DERIVED` へ訂正（DECLARED_SPEC 4 → 3 / DERIVED 28 → 29）。A0-011 の「20% 大きい」が **DERIVED どうしの比較**であることを明記 | D-49 |
| 3 | 依存関係誤記 1 件 | **MP-125** の `depends_on` が `MP-008`（`decode_speed_mean` 15.808）を指していた。117.59 GB/s の分子は中央値 15.9653 = **MP-007** | `MP-007` へ訂正。あわせて MP-121 から `MP-084`（同値なだけの run04 config 値）参照を除去 | D-50 |
| 4 | 集計・Git 状態の内部不整合 | `A0_EVIDENCE_INDEX.json` のみ `_amendments` に REM-02 を欠いていた。`repository` に観測時刻が無く `path_migration.git_status_before` と矛盾して読めた | REM-02 エントリを遡って記録（`recorded_late_by`）。`_path_note` / `_working_tree_note` で時点を明示。#1 / #2 に伴う集計を全件再計算 | D-51 |
| 5 | パス連続性のスコープ超過 | §7.1.1 と `path_migration.proven_range` の「**パス移行によって `results/` の内容が失われた・混ざったことは起きていない**」 | 削除し、`scope_limit_of_proven_range` として範囲外の理由 2 点を明記。未証明範囲に #6（母集団の不確定）を追加 | D-52 |

**既存の `A0_METRIC_PROVENANCE.metrics[].value` は変更していない。** 202.5 / 117.59 / 1168.5 を含め、
いずれの `value` も当該改訂前のままである。Claim / Deviation の件数・構造上の数値はこの不変範囲に含めない（D-60）。
訂正したのは判定・分類・依存関係・記録・表現である。

#### 12.5.2 D-52 について（同型の再発）

指摘 5 は、`ALSR-A0-REM-01` が D-44 として是正した
「**事後に書き換えられた形跡なし**」と**同じ型**の誤りである。
検証したのは 2 つの状態であるのに、その間で起きた**事象**について述べていた。
`ALSR-A0-REM-01` で 1 度是正した型の誤りを、`ALSR-A0-REM-02` で自ら再発させたことになる。

この事実は記録に残す。**1 度の是正では同じ型の誤りが消えない。**
本報告が 3 度目の改訂であることは、成果物の信頼度に対する情報そのものである。

#### 12.5.3 本改訂で行っていないこと

| 禁止事項 | 遵守状況 |
|---|---|
| コード修正 | 実施していない |
| build / test / benchmark | 実施していない。`results/` 配下は 1 バイトも変更していない（読み取りのみ） |
| model 操作 | 実施していない |
| `README.md` / `KNOWN_LIMITATIONS.md` 修正 | 実施していない。D-46 / D-48 が指す文書側の記述はいずれも**残したまま**である |
| 次 Gate 進行 | 実施していない |

`README.md:203`（「113〜118 GB/s」）、`README.md:205-207`（PCIe 性能の断定）、
`docs/KNOWN_LIMITATIONS.md:237`（「実測 0.6017 byte/param」）は**いずれも現在も文書に残っている**。
本改訂が行ったのは台帳側での判定と記録までである（U-10 / D-46 / D-48）。

---

### 12.6 `ALSR-A0-REM-04`

| 項目 | 内容 |
|---|---|
| Mission | `ALSR-A0-REM-04` |
| 実施日 | 2026-08-02 JST |
| 契機 | **`ALSR-A0-REM-03` が RED 裁定**。残存指摘 5 件の是正 |
| 対象 | `docs/recovery/` の A0 成果物 5 点のみ |
| 性質 | **A0 成果物自身の是正。再監査でも次 Gate 進行でもない。** |
| コード / テスト / ベンチマーク / `results/` / `README.md` / `KNOWN_LIMITATIONS.md` | **変更・実行していない** |
| 状態 | `COMPLETED_NOT_AUDITED`（裁定は豆虎Gate） |

#### 12.6.1 指摘と是正の対応

| # | 指摘 | 是正 | Deviation |
|---|---|---|---|
| 1 | A0-017 の Claim 粒度・判定誤り | A0-017A（静的比較 / EVIDENCE_PRESENT_NOT_REEXECUTED / evidence に manifest と environment の両方）と A0-017B（実行可否 / UNSUPPORTED_AS_WORDED + NO_EVIDENCE_FILE）へ分割。分割元は `SPLIT` で集計外（§6.2.3） | D-53 |
| 2 | D-52 型の時間スコープ超過の残存 | 「18:18 以降についてのみ否定できる」「証明できるのは 18:18 以降に限られる」を削除。2 時点の hash 一致からは間の変更・復元の不在を**18:18 以降についても**証明できない。18:18 は下限であって保証ではない（§7.1.1 / §11） | D-54 |
| 3 | `results/` の Git 追跡件数誤記 | 正しい内訳 **索引 80 = 追跡 1（`.gitkeep`）+ 非追跡 79** を明記。索引の追跡外 79 件は**すべて `results/` 配下**であり、索引対象外の `.claude/settings.local.json` は含まない（§3.2 / §7.1.1 / §7.2） | D-55 |
| 4 | Claim summary の過大表現 | 「20 件のうち、証拠が主張を文言どおりに支えているものは 1 件も無い」を削除。`EVIDENCE_PRESENT_NOT_REEXECUTED` は*証拠が支えていない*ではなく*内容確認まで済んでいる*判定である（§6） | D-56 |
| 5 | 類型集計の不整合 | 「分類・記録の誤り 2 件」としながら 3 ID を列挙していた。**類型ごとに全 ID を明示**し、合計 14 件が A0 成果物側の欠陥の総数と一致することを確認できる形へ（§4.1） | D-57 |

**変更していない値の範囲**は `A0_METRIC_PROVENANCE.metrics[].value`、既存 Evidence に記録された値、
参照元文書の測定値である。Claim / Deviation の件数や構造上の数値は改訂に伴って追加・再集計している（D-60）。

#### 12.6.2 `ALSR-A0-REM-03` の diff と自己検証ログの扱い

監査側の受領物に含まれていないため、以下は **`NOT_VERIFIED`** である。

| 対象 | 状態 |
|---|---|
| `ALSR-A0-REM-03.diff` | `NOT_VERIFIED`（監査未受領） |
| REM-03 報告中の「自己検証 46 項目」のログ | `NOT_VERIFIED`（監査未受領） |

**これらを REM-03 の証拠として遡及生成しない。** 生成すれば、事後に作った資料を
当時の検証記録として提示することになり、Rev.02 §10 の趣旨に反する。
REM-03 の SHA-256 突合が監査側で 5/5 一致した事実は §12.4.1 の表に残るが、
それは成果物の一致であって、検証プロセスが記録として存在することを意味しない。

---

### 12.7 `ALSR-A0-REM-05`

| 項目 | 内容 |
|---|---|
| Mission | `ALSR-A0-REM-05` |
| 実施日 | 2026-08-02 JST |
| 契機 | `ALSR-A0-REM-04` の読取監査で確認された残存指摘 3 件の是正 |
| 対象 | `docs/recovery/` の A0 成果物 5 点のみ |
| 性質 | **A0 成果物自身の是正。再監査でも次 Gate 進行でもない。** |
| コード / テスト / ベンチマーク / `results/` / `README.md` / `KNOWN_LIMITATIONS.md` | **変更・実行していない** |
| 状態 | `COMPLETED_NOT_AUDITED`（実装担当は GREEN を自己発行しない） |

#### 12.7.1 指摘と是正の対応

| # | 指摘 | 是正 | Deviation |
|---|---|---|---|
| 1 | A0-017B の note に Evidence / Provenance を持たない runtime・OS 挙動の断定が残存 | 当該断定を削除し、`results/` に実行試行・失敗記録が無いこと、runtime を観測していないこと、静的な容量比較は実行不能を示す実行証拠ではないことに判定根拠を限定（§6.2.3） | D-58 |
| 2 | 改訂履歴の回数・類型数・見出しが内部不整合 | 再発記録を **2 類型・3 再発関係**へ訂正。RED 裁定の根拠のない序数を削除し、§12 見出しを版列挙のない一般見出しへ変更 | D-59 |
| 3 | 「数値そのものは 1 つも変更していない」の範囲が未限定 | 不変範囲を `A0_METRIC_PROVENANCE.metrics[].value`、既存 Evidence に記録された値、参照元文書の測定値へ限定。Claim / Deviation の件数・構造上の数値は改訂に伴って変化し得ることを明記 | D-60 |

#### 12.7.2 再集計と不変範囲

- Claim は集計対象 21 件（配列 23）のまま。status / severity の件数も変更していない。
- Metric は 125 件のまま。`A0_METRIC_PROVENANCE.json` は変更していない。
- Evidence Index は 150 件のまま。`A0_EVIDENCE_INDEX.json` は変更していない。
- Deviation Matrix は D-58〜D-60 を追加して 57 → **60 行**。分類は誤認 22 → **25**、severity は MEDIUM 20 → **21** / LOW 12 → **14**。
- A0 成果物自身の欠陥は 14 → **17 件**。§4.1 に全 ID を列挙した。

#### 12.7.3 本改訂で行っていないこと

コード修正、build、test、benchmark、model 操作、`results/`・`README.md`・
`docs/KNOWN_LIMITATIONS.md`・`docs/EXPERIMENTS.md` の変更は行っていない。
読取確認と `docs/recovery/` の A0 成果物の是正だけを実施した。
