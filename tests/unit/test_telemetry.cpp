#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "alsr/bandwidth.hpp"
#include "alsr/checksum.hpp"
#include "alsr/host_store.hpp"
#include "alsr/telemetry.hpp"
#include "test_helpers.hpp"

using namespace alsr;
using namespace alsr::test;

TEST_CASE("Percentiles: nearest-rank", "[unit][telemetry]") {
  const Percentiles p = compute_percentiles({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  REQUIRE(p.p50 == 5.0);
  REQUIRE(p.p95 == 10.0);
  REQUIRE(p.p99 == 10.0);
  REQUIRE(p.max == 10.0);
  REQUIRE(p.mean == 5.5);

  const Percentiles empty = compute_percentiles({});
  REQUIRE(empty.p50 == 0.0);
  REQUIRE(empty.mean == 0.0);

  const Percentiles one = compute_percentiles({42.0});
  REQUIRE(one.p50 == 42.0);
  REQUIRE(one.p99 == 42.0);
}

TEST_CASE("MetricsWriter: JSONL は 1 行 1 レコードの追記", "[unit][telemetry]") {
  TempDir tmp("metrics");
  const std::string path = tmp.file("metrics.jsonl");

  {
    MetricsWriter w(path);
    for (u64 i = 0; i < 5; ++i) {
      ChunkMetric m;
      m.chunk_id = i;
      m.bytes = 1024 * (i + 1);
      m.h2d_ms = 1.5 * static_cast<double>(i);
      m.checksum_ok = true;
      m.scheduler_action = "keep";
      w.write("run-x", m);
    }
    REQUIRE(w.lines_written() == 5);
  }

  std::ifstream in(path);
  std::string line;
  std::size_t n = 0;
  while (std::getline(in, line)) {
    const json::Value v = json::Value::parse(line);
    REQUIRE(v.at("run_id").as_string() == "run-x");
    REQUIRE(v.at("chunk_id").as_int() == static_cast<i64>(n));
    ++n;
  }
  REQUIRE(n == 5);

  // 追記であること
  {
    MetricsWriter w(path);
    ChunkMetric m;
    m.chunk_id = 99;
    w.write("run-x", m);
  }
  std::ifstream in2(path);
  std::size_t n2 = 0;
  while (std::getline(in2, line)) ++n2;
  REQUIRE(n2 == 6);
}

TEST_CASE("必須ゲート: §12.1 の条件を機械判定する", "[unit][telemetry]") {
  RunSummary s;
  s.chunk_count = 10;
  s.environment_recorded = true;
  evaluate_mandatory_gates(s);
  REQUIRE(s.gate_pass);
  REQUIRE(s.gate_failures.empty());

  SECTION("checksum 失敗") {
    s.checksum_failures = 1;
    evaluate_mandatory_gates(s);
    REQUIRE_FALSE(s.gate_pass);
  }
  SECTION("Chunk 欠落") {
    s.missing_chunks = 3;
    evaluate_mandatory_gates(s);
    REQUIRE_FALSE(s.gate_pass);
  }
  SECTION("順序違反") {
    s.order_violations = 1;
    evaluate_mandatory_gates(s);
    REQUIRE_FALSE(s.gate_pass);
  }
  SECTION("environment 未記録") {
    s.environment_recorded = false;
    evaluate_mandatory_gates(s);
    REQUIRE_FALSE(s.gate_pass);
  }
  SECTION("Chunk がゼロ") {
    s.chunk_count = 0;
    evaluate_mandatory_gates(s);
    REQUIRE_FALSE(s.gate_pass);
  }
}

TEST_CASE("run_id は設定内容から決定論的に決まる", "[unit][telemetry]") {
  json::Value a = json::Value::make_object();
  a.set("x", 1);
  a.set("y", "z");
  json::Value b = json::Value::make_object();
  b.set("x", 1);
  b.set("y", "z");
  json::Value c = json::Value::make_object();
  c.set("x", 2);
  c.set("y", "z");

  REQUIRE(make_run_id("t", a) == make_run_id("t", b));
  REQUIRE(make_run_id("t", a) != make_run_id("t", c));
  REQUIRE(make_run_id("t", a).rfind("t-", 0) == 0);
}

TEST_CASE("EvidencePack: 必要なファイルを書ける", "[unit][telemetry]") {
  TempDir tmp("evidence");
  EvidencePack pack(tmp.dir(), "run-abc");

  json::Value cfg = json::Value::make_object();
  cfg.set("k", 1);
  pack.write_json("config.json", cfg);
  pack.write_text("report.md", "# hi\n");

  REQUIRE(std::filesystem::exists(pack.file("config.json")));
  REQUIRE(std::filesystem::exists(pack.file("report.md")));
  const json::Value back = json::Value::parse_file(pack.file("config.json"));
  REQUIRE(back.at("k").as_int() == 1);
}

TEST_CASE("HostStore: mmap と read が同じデータを返す", "[unit][hoststore]") {
  TempDir tmp("hoststore");
  const std::string p = tmp.file("model.gguf");
  write_test_gguf(p, {{"a.weight", 0, {4096}}, {"b.weight", 12, {3840, 64}}});

  HostStore m(p, HostStoreMode::Mmap);
  HostStore r(p, HostStoreMode::Read);
  REQUIRE(m.file_size() == r.file_size());

  std::vector<u8> bm(4096), br(4096);
  for (const Bytes off : {Bytes{0}, Bytes{1}, Bytes{4095}, m.file_size() - 4096}) {
    m.read_into(off, 4096, bm.data());
    r.read_into(off, 4096, br.data());
    INFO("offset=" << off);
    REQUIRE(bm == br);
  }
  REQUIRE(m.bytes_read() == r.bytes_read());
}

TEST_CASE("HostStore: 範囲外アクセスは拒否する", "[unit][hoststore]") {
  TempDir tmp("hoststore-range");
  const std::string p = tmp.file("model.gguf");
  write_test_gguf(p, {{"a.weight", 0, {1024}}});

  HostStore s(p, HostStoreMode::Mmap);
  std::vector<u8> buf(64);
  REQUIRE_THROWS_AS(s.read_into(s.file_size(), 1, buf.data()), Error);
  REQUIRE_THROWS_AS(s.read_into(0, s.file_size() + 1, buf.data()), Error);
  REQUIRE_THROWS_AS(s.read_into(0, 1, nullptr), Error);
  REQUIRE_NOTHROW(s.read_into(0, 0, nullptr));  // 長さ 0 は無害
}

TEST_CASE("HostStore: synthetic は決定論的", "[unit][hoststore]") {
  HostStore a("", HostStoreMode::Synthetic, 1 << 20);
  HostStore b("", HostStoreMode::Synthetic, 1 << 20);
  std::vector<u8> x(4096), y(4096);
  a.read_into(12345, 4096, x.data());
  b.read_into(12345, 4096, y.data());
  REQUIRE(x == y);
  // オフセット依存であり、読み順に依存しない
  std::vector<u8> z(4096);
  b.read_into(0, 4096, z.data());
  b.read_into(12345, 4096, z.data());
  REQUIRE(z == x);
}

TEST_CASE("BandwidthModel: 決定論と設定検証", "[unit][bandwidth]") {
  BandwidthProfile p;
  p.bytes_per_s = 25e9;
  p.fixed_latency_s = 1e-5;
  p.jitter_fraction = 0.2;

  BandwidthModel a(p, 42);
  BandwidthModel b(p, 42);
  BandwidthModel c(p, 43);

  std::vector<double> ta, tb, tc;
  double now = 0.0;
  for (int i = 0; i < 50; ++i) {
    ta.push_back(a.transfer_time_s(256 * MiB, now));
    tb.push_back(b.transfer_time_s(256 * MiB, now));
    tc.push_back(c.transfer_time_s(256 * MiB, now));
    now += 0.01;
  }
  REQUIRE(ta == tb);      // 同一 seed → 同一系列
  REQUIRE(ta != tc);      // seed が違えば違う
  for (const double t : ta) REQUIRE(t > 0.0);

  // jitter 無しなら厳密に理想時間 + レイテンシ
  BandwidthProfile q;
  q.bytes_per_s = 10e9;
  q.fixed_latency_s = 0.0;
  q.jitter_fraction = 0.0;
  BandwidthModel d(q, 1);
  REQUIRE(d.transfer_time_s(10'000'000'000ull, 0.0) == 1.0);
}

TEST_CASE("BandwidthProfile: 不正値を拒否する", "[unit][bandwidth]") {
  BandwidthProfile p;
  p.bytes_per_s = 0;
  REQUIRE_THROWS_AS(p.validate(), Error);

  BandwidthProfile q;
  q.jitter_fraction = 1.5;
  REQUIRE_THROWS_AS(q.validate(), Error);

  BandwidthProfile r;
  r.pause_period_s = 1.0;
  r.pause_duration_s = 2.0;
  REQUIRE_THROWS_AS(r.validate(), Error);
}

TEST_CASE("BandwidthModel: 一時停止が転送時間へ乗る", "[unit][bandwidth]") {
  BandwidthProfile p;
  p.bytes_per_s = 1e9;
  p.fixed_latency_s = 0.0;
  p.jitter_fraction = 0.0;
  p.pause_period_s = 0.5;
  p.pause_duration_s = 0.1;

  BandwidthModel m(p, 7);
  // 1GB = 1.0 秒。0.5 秒ごとの停止に 2 回当たる。
  const double t = m.transfer_time_s(1'000'000'000ull, 0.0);
  REQUIRE(t > 1.0);
  REQUIRE(m.pause_hits() >= 2);
}
