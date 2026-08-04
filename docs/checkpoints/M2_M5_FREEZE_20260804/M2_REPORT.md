# ALSR M2 Real Supply Repair 正式完了報告

Mission: `ALSR-M2-M5-REAL-SUPPLY-REPAIR-01`  
正式判定: `GREEN`  
判定時刻: `2026-08-04T10:00:24+09:00`

## 1. 結論

M5 16 GiB上のGemma 4 12B実推論で、次の実データ経路が機能することを確認した。

```text
GGUF Model Store
  -> 明示range read / checksum Preparation
  -> Ready Queue
  -> 固定Scheduler
  -> 有限Ring（2 slot）
  -> prepared rangeを実sourceとするMetal backend copy
  -> llama.cpp Executor
  -> token生成
```

128 new tokens × 5独立Runは、M0 baselineのgenerated token ID列128/128およびUTF-8本文と全件完全一致した。通常Runtime tensor sourceへのfallbackは5 Run合計0だった。

したがって、このscopeでは`ALSR Real Executor inference = true`と判定する。

## 2. 旧M2判定の撤回

2026-08-03 JSTの旧M2 GREENは誤判定として撤回した。旧Runの`callback_requests_tensor_data`は全件`false`で、hookは既存mmap tensorとbackend copyを観測・検査しただけだった。ALSRがcopy source bytesを供給した証拠ではなかった。

旧Evidence Packは削除せず、`results/m2-gemma4-12b-real-executor-20260803T090123Z/M2_RETRACTION.json`に撤回理由を固定した。旧Runが示す範囲は`INTEGRATION_SPIKE_ONLY`である。

## 3. 修復内容

- pinned llama.cpp scheduler ABIにprepared range request／commitを追加した。
- `HostStoreMode::Read`でGGUFの必要rangeを明示的に読み込む`RealSupplyPipeline`を接続した。
- PreparationでCRC32Cを毎回、SHA-256を各unique Chunkの初回に検証した。
- Ready Queue、固定Scheduler、2 slotの有限Ringを実Executorへ接続した。
- Ring状態を`EMPTY -> PREPARING -> READY -> COPYING -> RESIDENT -> IN_USE -> RETIRED -> EMPTY`として強制した。
- backendがprepared bytesを同期copyした後にのみslotを回収した。
- prepared rangeが欠落した場合の通常Runtime source fallbackを禁止した。
- 並行ログの行順と実状態順を混同しないよう、mutex内で単調な`transition_sequence`を採番した。

Runtimeの通常mmapはmodel metadataとgraph構築のため残る。ただし、host-backed weightをMetal splitへcopyする実sourceはModel Storeから準備したRing bytesであり、mmap tensor pointerへのfallbackは0である。

## 4. 正式受入結果

| Run | token ID | 本文 | wall time秒 | peak RSS bytes | Model Store read bytes | prepared ranges | fallback |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 128/128一致 | 一致 | 324.545 | 4,255,825,920 | 942,791,483,392 | 86,912 | 0 |
| 2 | 128/128一致 | 一致 | 307.392 | 4,441,735,168 | 942,791,483,392 | 86,912 | 0 |
| 3 | 128/128一致 | 一致 | 318.034 | 5,097,439,232 | 942,791,483,392 | 86,912 | 0 |
| 4 | 128/128一致 | 一致 | 332.274 | 4,895,244,288 | 942,791,483,392 | 86,912 | 0 |
| 5 | 128/128一致 | 一致 | 339.959 | 4,260,069,376 | 942,791,483,392 | 86,912 | 0 |

wall timeは最小`307.392`秒、中央値`324.545`秒、最大`339.959`秒で、5/5が事前上限1,800秒未満だった。swap operationは5 Runとも0だった。

5 Run合計:

- Model Store明示read／prepared／backend weight copy: `4,713,957,416,960` byte
- prepared range: `434,560`
- Ring状態遷移: `3,041,920`
- 違法状態遷移: `0`
- Runtime source fallback: `0`

各RunでModel Store read bytes、pipeline prepared bytes、pipeline supplied bytes、backend weight copy bytesが完全一致した。

## 5. 再現性・実装固定

- model SHA-256: `1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606`
- llama.cpp: tag `b10091` / commit `b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45`
- Runtime patch SHA-256: `33f7bf1348c755e4a59c54d17d43ff963726e54fa26bf5212278d874542682e7`
- Executor binary SHA-256: `50cda0a2412f5065bb9b4eede525e1c6bd8ef4b44a87d5682c6eec9f1ed3e771`
- 独立graph capture 2本: SHA-256 `2168aa94351524e35130e23ec660bffacf95e5950996e7e8ea8af5c7eab5d626`でバイト一致
- 独立Manifest 2本: SHA-256 `871a5592265fae4512ba10d3ae809acdc8c8f476deb7f971e497c1764ad9354a`でバイト一致
- Manifest structure SHA-256: `14e1c39b2f305b66d6711386c5e4e6c91337410fca0b074be6203ed9f1426511`
- 単体・統合test: `109/109 PASS`

実行に使用したsource、canonical Runtime patch、verifier、Executor binaryは`implementation/`へ固定した。

## 6. 判定方法

検証器はsummaryの自己申告GREENを合格根拠にしない。SHA-256を再計算した後、5本のRESULT、GRAPH_TRACE、supply_traceを読み、出力、I/O byte会計、range bind／commit、Ring transition sequence、時間上限を生Evidenceから導出した。

```text
M2_ACCEPTANCE_VERIFIED: GREEN
(derived: 5/5 exact, explicit Model Store -> finite Ring -> Backend, fallback 0)
```

旧撤回Packを同じ検証器へ渡すと、`M2_RETRACTION.json`を検出して失敗する。

## 7. 除外した診断Run

`results/m2-gemma4-12b-real-supply-repair-20260803T235054Z/`の5 Runは推論自体には成功したが、並行telemetryに実状態順のsequenceが無かったため正式受入から除外した。結果を流用せず、修正後に5 Runすべて取り直した。

## 8. Scope境界

今回GREENになったのは、Gemma 4 12Bを用いたM5 fixed-path Real Supply Integrationである。

未証明:

- Gemma 4 31Bまたは16 GiB制約を実質的に超えるmodel
- 「重量級LLMを少ないメモリでどこまで動かせるか」という研究上限
- Adaptive Schedulerの効果
- Ollama／通常mmapより速いという性能優位
- RTX／CUDA／PCIe経路

次のGate M3では、この同じ明示range-I/O／有限Ring構造をGemma 4 31Bへ適用し、M5 16 GiB上のmemory-pressure実推論を検証する。アーキテクチャを別方式へ縮約してはならない。
