# ALSR Windows／RTX 引継ぎ手順

文書ID: `ALSR-X0-WINDOWS-RTX-HANDOFF-01`
作成日: 2026-08-04 JST
状態: `LOCAL_GIT_CHECKPOINT_READY / REMOTE_NOT_CONFIGURED`

## 1. 現在地

M5上のGate M2は、Gemma 4 12Bを用いた実データ供給経路の128 token × 5 RunでGREENへ再認定された。

実証済みの経路:

```text
GGUF Model Store
  -> explicit range read
  -> checksum Preparation
  -> Ready Queue
  -> fixed Scheduler
  -> finite Ring
  -> prepared range request / commit
  -> Backend copy
  -> real llama.cpp Executor
  -> token generation
```

M5はこの到達点で凍結する。今後の31B、scale、CUDA、PCIe、VRAM検証はWindows RTX機で行う。

これはM3／M4を完了扱いにする変更ではない。両Gateは`NOT_RUN`のままRTX側のR3／S0へ移設された。

## 2. M5凍結規則

ユーザーの新たな明示許可がない限り、M5で次を実行しない。

- GGUF／Ollama modelのload
- 実LLM推論
- 帯域benchmark
- sustained file I/O
- M2 acceptanceの再実行
- 31B以上のmodel取得・変換・実行

M5で許可されるのは、Git操作、文書作業、read-only監査、軽量なsource検査までとする。テスト実行も原則Windows側へ移す。

## 3. Gitに含めるもの／含めないもの

Git checkpointへ含める:

- ALSR sourceとtest
- pinned llama.cpp canonical patch
- 要件定義・設計・監査履歴
- M2の小型summary／report／verification／retraction
- Windows／RTX引継ぎ手順

通常Gitへ含めない:

- GGUF、Ollama blob、model weight
- build directory
- `work/`の開発中間物
- `results/`の生Evidence Pack
- 1 Runあたり100 MiBを超える` supply_trace.jsonl`

正式M2 raw EvidenceはM5ローカルの次の場所にある。

```text
/Users/yoshiki/Desktop/ALSR-PoC/results/
  m2-gemma4-12b-real-supply-repair-20260804T002432Z/
```

容量は約1.7 GB。`SHA256SUMS`自体のSHA-256は次のとおり。

```text
95c4a64c8cac7f9e16eeea24c6a8055589ef5a59c14aaf947c3bec2ac47ead48
```

小型の判定成果物と上記SHA一覧は`docs/checkpoints/M2_M5_FREEZE_20260804/`へ複製した。これによりWindows側は判定、実装identity、raw Evidenceの期待hashを参照できる。ただしraw trace本体は別途バックアップしない限りWindows側には存在しない。

### 3.1 Git checkpointの復元

checkpoint tagは`m5-freeze-20260804`、作業branchは`agent/m5-freeze-windows-handoff`とする。

GitHub remote設定後:

```powershell
git clone <repository-url> ALSR-PoC
Set-Location ALSR-PoC
git checkout m5-freeze-20260804
git switch -c windows/rtx-r0
```

portable bundleから復元する場合:

```powershell
git clone ALSR-PoC-m5-freeze-20260804.bundle ALSR-PoC
Set-Location ALSR-PoC
git switch agent/m5-freeze-windows-handoff
git show --no-patch m5-freeze-20260804
git switch -c windows/rtx-r0
```

bundleはGit repositoryの外に生成し、同梱の`.sha256`と`git bundle verify`で確認する。
bundleにはGit管理外のraw Evidence、model、`results/`、`work/`は含まれない。

## 4. 次のGate順序

順序を入れ替えない。

| 順序 | Mission | 目的 | 完了条件の要点 |
|---:|---|---|---|
| 1 | `ALSR-X0-CORE-FREEZE-WINDOWS-HANDOFF-01` | Git checkpointとBackend境界の固定 | commit／tag、handoff、Evidence所在 |
| 2 | `ALSR-R0-RTX-ENVIRONMENT-01` | Windows／RTX実環境の取得 | 仮定0、driver／CUDA／PCIe／RAM／storage固定 |
| 3 | `ALSR-R1-CUDA-TRANSFER-MICROBENCH-01` | pageable／pinned、sync／async、stream、Chunkの実測 | 生ログ、反復、cold／warm区別 |
| 4 | `ALSR-R2-CUDA-SYNTHETIC-STREAMING-01` | CUDA bufferとconsumer kernelで順序・完全性・overlapを確認 | finite VRAM Ring、fallback 0、checksum |
| 5 | `ALSR-R3A-CUDA-12B-PORT-REGRESSION-01` | M2と同一modelでCUDA移植の正確性を確認 | direct baseline一致、実ALSR source、短い固定受入 |
| 6 | `ALSR-R3B-CUDA-31B-MEMORY-PRESSURE-01` | 最初のVRAM超過実modelを推論 | 31B実推論、memory／I/O／正確性Evidence |
| 7 | `ALSR-S0-SCALE-RESEARCH-01` | 20〜31Bから70B、120B、405B級へ境界探索 | 成立／不成立frontier |

R3Aの12Bはport regressionであり、研究上限の証明ではない。R3Bの31Bも最初の測定点であり、研究完了ではない。

## 5. Windows側で最初に採取する情報（R0）

実装やbenchmarkの前に、次をそのままEvidenceへ保存する。

- Windows edition／build
- CPU正式名、core／thread数
- RAM総量、DIMMごとの容量／速度／メーカー
- RTX 3060のboard名、VRAM、VBIOS
- NVIDIA driver版
- CUDA driver capabilityとToolkit版
- PCIe current／maximum generationとlink width
- modelを置くstorageの種類、容量、空き、filesystem
- Visual Studio／MSVC、CMake、Ninja、Git、Pythonの版
- power limit、temperature limit、background GPU process
- Ollama／llama.cppのmodel実体path、size、SHA-256

PowerShell採取例:

```powershell
systeminfo
nvidia-smi -q
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pci.bus_id,pstate,temperature.gpu,power.limit --format=csv
Get-CimInstance Win32_Processor | Format-List *
Get-CimInstance Win32_PhysicalMemory | Format-List Manufacturer,PartNumber,Capacity,Speed,ConfiguredClockSpeed
Get-PhysicalDisk | Format-Table FriendlyName,MediaType,BusType,Size,HealthStatus
git --version
cmake --version
python --version
nvcc --version
```

取得できない項目は推測で埋めず`NOT_OBSERVED`とする。

## 6. 現sourceをWindowsでそのまま実行してはいけない理由

このcheckpointはWindowsで即実行可能な完成品ではない。次のport作業が必要である。

1. `HostStore`はPOSIXの`open`／`pread`／`mmap`／`madvise`へ直接依存している。Windows用read-only range I/O実装が必要。
2. Executor名と一部説明は`alsr-m5-executor`／Metal固定のままである。Backend-neutral interfaceへ分離する。
3. pinned llama.cpp patchのprepared range ABIはBackend共通だが、CUDAでのcopy完了・event lifetimeは未検証。
4. CUDA pinned host memory、device buffer、copy stream、compute streamは未接続。
5. Windows environment／resource telemetryは未実装または未検証。

これらを直さず通常llama.cpp／Ollamaが動いたことをALSR成功として報告してはならない。

## 7. pinned llama.cppの復元

固定値:

```text
tag: b10091
commit: b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45
patch: patches/llama.cpp-b10091-alsr-m2.patch
patch SHA-256: 33f7bf1348c755e4a59c54d17d43ff963726e54fa26bf5212278d874542682e7
```

PowerShell例:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git work/upstream/llama.cpp-b10091
git -C work/upstream/llama.cpp-b10091 checkout b4d6c7d8ff69c2e05e4e8ee7e6e710a08abd7b45
$Patch = (Resolve-Path patches/llama.cpp-b10091-alsr-m2.patch).Path
git -C work/upstream/llama.cpp-b10091 apply --check $Patch
git -C work/upstream/llama.cpp-b10091 apply $Patch
git -C work/upstream/llama.cpp-b10091 status --short
```

期待される差分SHAがcanonical patchと一致しない場合はbuildを止める。

## 8. CUDA buildの入口

これはR0完了後の入口例であり、現checkpointのまま成功を保証するものではない。

```powershell
cmake -S . -B build-rtx -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DALSR_BUILD_TESTS=ON `
  -DALSR_ENABLE_M1_LLAMA_TRACE=ON `
  -DALSR_ENABLE_M2_EXECUTOR=ON `
  -DALSR_LLAMA_CPP_SOURCE_DIR="$PWD/work/upstream/llama.cpp-b10091" `
  -DALSR_LLAMA_CPP_PATCH_FILE="$PWD/patches/llama.cpp-b10091-alsr-m2.patch" `
  -DGGML_CUDA=ON `
  -DGGML_METAL=OFF
cmake --build build-rtx --parallel
ctest --test-dir build-rtx --output-on-failure
```

最初の失敗はport境界のEvidenceとして保存し、都合のよいplatform branchへ黙ってfallbackしない。

## 9. R1／R2で固定すべき比較軸

- pageable vs pinned host memory
- synchronous vs asynchronous H2D
- single stream vs copy／compute separate streams
- Chunk sizeとRing depthの組合せ
- cold storage read vs warm page cache／RAM source
- CRC32C／SHA検証あり・なしを別Runとして記録
- resident VRAM bytesとstreamed bytesの分離
- transfer-only、compute-only、overlapの分離
- current PCIe link speed／widthの各Run記録

結果を見てから受入条件や母集団を変更しない。

## 10. 禁止事項

- 12Bが通常CUDAで動いたことをR3B成功としない。
- ordinary llama.cpp offload／OS pagingをALSR供給として扱わない。
- prepared range欠落時にRuntime tensor pointerへfallbackしない。
- model全体をRAMまたはVRAMへcopyしてstreaming成功と報告しない。
- M2のQueue／Ring／range lifecycleを削除・縮約しない。
- benchmark値、trace、summaryを事後修正しない。
- failure Runを削除・上書きしない。
- model weightをGitへ追加しない。
- 31Bだけで研究全体を終了しない。

## 11. Windows作業開始時の読書順

1. `docs/ALSR_REQ_DESIGN.md`
2. 本書
3. `docs/checkpoints/M2_M5_FREEZE_20260804/CHECKPOINT.md`
4. `docs/checkpoints/M2_M5_FREEZE_20260804/M2_REPORT.md`
5. `docs/KNOWN_LIMITATIONS.md`
6. `patches/llama.cpp-b10091-alsr-m2.patch`

この順序より先にコードを変更しない。

## 12. 未解決事項

- GitHub remoteのrepository名／可視性は未確定。
- raw M2 Evidence 1.7 GBのMac外バックアップ先は未確定。
- Windows native MSVCとWSL2のどちらを正式環境にするか未決定。
- CUDA Toolkit版、driver版、PCIe current link、storage帯域はR0で取得する。
- Gemma 4 31BのWindows側実ファイルpath／digest／量子形式は未固定。

未解決事項をAIが推測で確定してはならない。
