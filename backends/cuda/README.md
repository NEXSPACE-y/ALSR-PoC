# CUDA Transfer Backend — Phase 3（未実装）

このディレクトリは設計書 §18「Phase 3 用 Mission」で実装する。
**Phase 0-2 の Mission には含まれない**ので、意図的に空にしてある。

## 実装するインタフェース

`include/alsr/transfer_backend.hpp` の `TransferBackend` を実装する。
simulator backend と同じ抽象を使うことが NFR-04（移植性）の要件である。

```cpp
class TransferBackend {
  virtual std::string          name() const = 0;
  virtual TransferCapabilities capabilities() const = 0;
  virtual double h2d_time_s(Bytes n, double now_s) = 0;
  virtual double prepare_time_s(Bytes n, double now_s) = 0;
  virtual double compute_time_s(Bytes n, double now_s) = 0;
  virtual u32    consume(const u8* host_data, Bytes n) = 0;
  virtual json::Value config_json() const = 0;
};
```

simulator は仮想時間を返す。CUDA backend は **CUDA Event の実測値**を返す。
`capabilities().measured_on_real_hardware` を `true` にしてよいのは
CUDA backend だけである。

## Phase 3 で満たすこと（設計書 FR-08 / §18）

- `cudaMemcpyAsync` を非デフォルト Stream で使う
- H2D と kernel を異なる非デフォルト Stream で実行できる
- CUDA Event で処理区間を計測する
- `deviceOverlap` / `asyncEngineCount` 等の能力を取得する
- 同期範囲を最小化し、全 Device 同期をホットパスで乱用しない
- Pageable / Pinned、Sync / Async、Stream 1〜4、Chunk 16 MiB〜1 GiB のスイープ
- Pinned Memory 上限を設け、OS / Driver を不安定にしない（NFR-03）
- 10 分の stress test
- GREEN / YELLOW / RED 判定レポート

**実 LLM 統合は Phase 3 に含めない。**

## 着手前に必ず採取すること（設計書 §3.2 / §3.3）

```bash
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
           --format=csv
nvidia-smi -q | grep -A4 "GPU Link Info"
nvcc --version
cat /proc/driver/nvidia/version
lspci -vv | grep -A20 VGA        # LnkCap / LnkSta を確認
```

`nvidia-smi` の表示だけを信じないこと。CUDA sample の `bandwidthTest`、
PCIe トポロジ、自前の H2D microbenchmark で裏を取る。

MEG X570 UNIFY は対応 CPU で 1 枚時 x16、2 枚時 x8/x8 になる。
将来の 2 GPU 構成では GPU ごとの PCIe 上限が半減し得るので、
**current link width を Run ごとに保存する**（§14 のリスク表）。

## Phase 0-2 から持ち込む前提

`docs/EXPERIMENTS.md` と `docs/KNOWN_LIMITATIONS.md` に測定値がある。特に:

- `gemma4:12b` の S_stream 上限 = 7,365,558,464 byte/token
- 転送だけの上限: PCIe 3.0 (12 GB/s) で 1.63 tok/s、PCIe 4.0 (25 GB/s) で 3.39 tok/s
- M5 のユニファイドメモリ実効値 113〜118 GB/s、実測 15.35〜15.97 token/s
- **単一コアの CRC32C は 11.77 GB/s。PCIe 4.0 より遅い。**
  全 Chunk のホスト側 checksum を線速で回せない前提で設計すること
- `gemma4:12b` は 12 GB VRAM に収まる。ALSR の優位を測る対象ではない

## 判定基準（DR-0004 の反証条件）

仮想時間モデルと CUDA Event の実測が **20% 以上乖離**したら、
モデル（特に直列エンジン仮定と固定レイテンシ）を見直す。
設計書 §15.3 のエスカレーション条件に該当する。

## 言語構成（DR-0008）

Phase 3 は独自カーネルを必要とせず `cudaMemcpyAsync` / Stream / Event /
device attribute の取得だけで済むため、**Rust バインディングでの実装を試す
場所として適している**。失敗しても損害が 1 モジュールに閉じる。
判定基準は `docs/DECISIONS.md` DR-0008 を参照。
