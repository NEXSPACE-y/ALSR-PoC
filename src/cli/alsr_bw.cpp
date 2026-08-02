// alsr-bw — Phase 0 の帯域実測 (設計書 §11 Phase 0)。
//
// M5 で測れるのは以下だけである。PCIe / Pinned H2D は RTX 機 (Phase 3)。
//   - RAM memcpy 帯域
//   - RAM 逐次読み出し帯域
//   - モデルファイルの読み出し帯域 (mmap / pread)
//   - CRC32C スループット（checksum が律速でないことの確認）
//
// ページキャッシュの影響を受けるため、cold/warm を区別せずに一括で「実測値」と
// 呼ばない。warm 前提であることを明示して記録する。
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/env.hpp"
#include "alsr/host_store.hpp"
#include "cli_common.hpp"

using namespace alsr;

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Measurement {
  std::string name;
  std::string unit = "GB/s";
  std::vector<double> samples;
  std::string note;

  [[nodiscard]] json::Value to_json() const {
    json::Value j = json::Value::make_object();
    j.set("name", name);
    j.set("unit", unit);
    j.set("median", median(samples));
    j.set("min", samples.empty() ? 0.0
                                 : *std::min_element(samples.begin(), samples.end()));
    j.set("max", samples.empty() ? 0.0
                                 : *std::max_element(samples.begin(), samples.end()));
    json::Value arr = json::Value::make_array();
    for (const double s : samples) arr.push_back(json::Value(s));
    j.set("samples", std::move(arr));
    j.set("note", note);
    return j;
  }
};

void usage() {
  std::cout << R"(alsr-bw — Phase 0 帯域実測

使い方:
  alsr-bw [options]

オプション:
  --buffer-mib <n>   ベンチ用バッファサイズ [MiB] (既定 512)
  --repeats <n>      反復回数 (既定 7、warm-up 後の中央値を採る)
  --model <path>     ファイル読み出し帯域を測る対象
  --ollama <ref>     Ollama blob を対象にする
  --read-mib <n>     ファイルから読むバイト数 [MiB] (既定 2048)
  --out <path>       JSON 出力
  --md-out <path>    Markdown 出力 (bandwidth_report.md)
)";
}

}  // namespace

int main(int argc, char** argv) {
  return cli::run_main([&]() -> int {
    cli::Args args(argc, argv);
    if (args.flag("help")) {
      usage();
      return 0;
    }
    args.reject_unknown({"buffer-mib", "repeats", "model", "ollama", "read-mib",
                         "out", "md-out", "help"});

    const std::size_t buf_bytes =
        static_cast<std::size_t>(args.get_u64("buffer-mib", 512) * MiB);
    const u64 repeats = std::max<u64>(args.get_u64("repeats", 7), 1);

    std::vector<Measurement> results;

    // --- 1. memcpy 帯域 --------------------------------------------------
    {
      std::vector<u8> src(buf_bytes);
      std::vector<u8> dst(buf_bytes);
      for (std::size_t i = 0; i < buf_bytes; ++i) {
        src[i] = static_cast<u8>(i & 0xFF);
      }
      Measurement m;
      m.name = "ram_memcpy";
      m.note = "memcpy の read+write 合計ではなく、コピーしたバイト数 / 秒";
      std::memcpy(dst.data(), src.data(), buf_bytes);  // warm-up
      for (u64 r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        std::memcpy(dst.data(), src.data(), buf_bytes);
        const double s = seconds_since(t0);
        m.samples.push_back(static_cast<double>(buf_bytes) / s / 1e9);
      }
      // 最適化で消えないように結果を使う
      volatile u8 sink = dst[buf_bytes - 1];
      (void)sink;
      results.push_back(std::move(m));
    }

    // --- 2. 逐次読み出し帯域 ---------------------------------------------
    {
      std::vector<u64> buf(buf_bytes / sizeof(u64));
      std::iota(buf.begin(), buf.end(), 0ull);
      Measurement m;
      m.name = "ram_sequential_read";
      m.note = "u64 の総和。ストア無しの純粋な読み出し";
      u64 acc = 0;
      for (u64 r = 0; r < repeats + 1; ++r) {
        const auto t0 = Clock::now();
        u64 local = 0;
        for (const u64 v : buf) local += v;
        const double s = seconds_since(t0);
        acc += local;
        if (r > 0) {  // 1 回目は warm-up
          m.samples.push_back(static_cast<double>(buf.size() * sizeof(u64)) / s / 1e9);
        }
      }
      volatile u64 sink = acc;
      (void)sink;
      results.push_back(std::move(m));
    }

    // --- 3. CRC32C スループット ------------------------------------------
    {
      std::vector<u8> buf(buf_bytes);
      for (std::size_t i = 0; i < buf_bytes; ++i) {
        buf[i] = static_cast<u8>((i * 31) & 0xFF);
      }
      Measurement m;
      m.name = "crc32c";
      m.note = crc32c_hardware_enabled() ? "ARMv8 CRC32 命令を使用"
                                         : "ソフトウェア slicing-by-8";
      volatile u32 sink = 0;
      sink = crc32c(buf.data(), buf.size());  // warm-up
      for (u64 r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        const u32 v = crc32c(buf.data(), buf.size());
        const double s = seconds_since(t0);
        sink = v;
        m.samples.push_back(static_cast<double>(buf_bytes) / s / 1e9);
      }
      (void)sink;
      results.push_back(std::move(m));
    }

    // --- 4. ファイル読み出し ---------------------------------------------
    std::string model_path;
    if (args.has("model") || args.has("ollama")) {
      model_path = cli::resolve_model_path(args);
      const Bytes want = args.get_u64("read-mib", 2048) * MiB;

      for (const HostStoreMode mode : {HostStoreMode::Mmap, HostStoreMode::Read}) {
        HostStore store(model_path, mode);
        const Bytes n = std::min<Bytes>(want, store.file_size());
        std::vector<u8> dst(static_cast<std::size_t>(std::min<Bytes>(n, 256 * MiB)));

        Measurement m;
        m.name = std::string("file_read_") + std::string(to_string(mode));
        m.note = "ページキャッシュ warm 前提。cold 値ではない";
        for (u64 r = 0; r < repeats; ++r) {
          const auto t0 = Clock::now();
          Bytes done = 0;
          while (done < n) {
            const Bytes take = std::min<Bytes>(dst.size(), n - done);
            store.read_into(done, take, dst.data());
            done += take;
          }
          const double s = seconds_since(t0);
          m.samples.push_back(static_cast<double>(n) / s / 1e9);
        }
        results.push_back(std::move(m));
      }
    }

    // --- 出力 ------------------------------------------------------------
    json::Value root = json::Value::make_object();
    root.set("buffer_bytes", buf_bytes);
    root.set("repeats", repeats);
    root.set("model_path", model_path);
    json::Value arr = json::Value::make_array();
    for (const auto& m : results) arr.push_back(m.to_json());
    root.set("measurements", std::move(arr));
    root.set("environment", collect_environment());
    root.set("caveat",
             "M5 には独立 VRAM への PCIe H2D 経路が無い。ここに Pageable/Pinned "
             "H2D は含まれない (設計書 §3.1 / Phase 3 で RTX 機にて実測)");

    std::cout << "\n=== 帯域実測 (中央値, " << repeats << " 回) ===\n";
    for (const auto& m : results) {
      std::cout << "  " << std::left << std::setw(26) << m.name << std::right
                << std::fixed << std::setprecision(2) << std::setw(9)
                << median(m.samples) << " " << m.unit << "   " << m.note << "\n";
    }
    std::cout << "\n  ※ M5 では PCIe H2D を測れない。Phase 3 で RTX 機にて実測する。\n";

    if (args.has("out")) {
      std::ofstream out(args.get("out"), std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + args.get("out"));
      out << root.dump(2) << "\n";
      std::cout << "\nwrote " << args.get("out") << "\n";
    }
    if (args.has("md-out")) {
      std::ostringstream md;
      md << "# 帯域実測レポート (Phase 0)\n\n";
      md << "- バッファ: " << buf_bytes << " bytes (" << human_bytes(buf_bytes)
         << ")\n";
      md << "- 反復: " << repeats << " 回（中央値を採用）\n";
      if (!model_path.empty()) md << "- モデル: `" << model_path << "`\n";
      md << "\n| 指標 | 中央値 [GB/s] | 最小 | 最大 | 備考 |\n";
      md << "|---|---:|---:|---:|---|\n";
      for (const auto& m : results) {
        const double lo = m.samples.empty()
                              ? 0.0
                              : *std::min_element(m.samples.begin(), m.samples.end());
        const double hi = m.samples.empty()
                              ? 0.0
                              : *std::max_element(m.samples.begin(), m.samples.end());
        md << "| " << m.name << " | " << std::fixed << std::setprecision(2)
           << median(m.samples) << " | " << lo << " | " << hi << " | " << m.note
           << " |\n";
      }
      md << "\n> M5 には独立 VRAM への PCIe H2D 経路が無い。Pageable / Pinned "
            "H2D は Phase 3 で RTX 3060 上にて実測する（設計書 §3.1）。\n";
      std::ofstream out(args.get("md-out"), std::ios::trunc);
      if (!out) fail(ErrorKind::Io, "cannot write " + args.get("md-out"));
      out << md.str();
      std::cout << "wrote " << args.get("md-out") << "\n";
    }
    return 0;
  });
}
