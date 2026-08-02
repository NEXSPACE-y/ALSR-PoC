#include "alsr/sim.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <sstream>

#include "alsr/checksum.hpp"
#include "alsr/ring_buffer.hpp"

namespace alsr {
namespace {

double safe_div(double a, double b) noexcept {
  if (!(b > 0.0) || !std::isfinite(b) || !std::isfinite(a)) return 0.0;
  const double r = a / b;
  return std::isfinite(r) ? r : 0.0;
}

/// シミュレーションが 1 単位として扱う Chunk。
struct SimChunk {
  u64   chunk_id = 0;
  Bytes source_offset = 0;
  Bytes bytes = 0;
  u32   expected_crc = 0;
  bool  crc_known = false;
  bool  per_token = true;  ///< 定常状態で毎トークン流すか
};

/// 直列エンジンが作る互いに素な区間列どうしの重なりを測る。
double overlap_measure(const std::vector<std::pair<double, double>>& a,
                       const std::vector<std::pair<double, double>>& b) {
  double total = 0.0;
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() && j < b.size()) {
    const double lo = std::max(a[i].first, b[j].first);
    const double hi = std::min(a[i].second, b[j].second);
    if (hi > lo) total += hi - lo;
    if (a[i].second < b[j].second) ++i;
    else ++j;
  }
  return total;
}

/// 直近 N 件の統計。scheduler 入力を作る。
class Window {
public:
  explicit Window(std::size_t n) : n_(n == 0 ? 1 : n) {}

  void push(double copy_ms, double compute_ms, double wait_ms, double span_ms) {
    copy_.push_back(copy_ms);
    compute_.push_back(compute_ms);
    wait_sum_ += wait_ms;
    span_sum_ += span_ms;
    wait_.push_back(wait_ms);
    span_.push_back(span_ms);
    while (copy_.size() > n_) {
      copy_.pop_front();
      compute_.pop_front();
      wait_sum_ -= wait_.front();
      wait_.pop_front();
      span_sum_ -= span_.front();
      span_.pop_front();
    }
  }

  [[nodiscard]] double p50_compute() const { return quantile(compute_, 0.50); }
  [[nodiscard]] double p50_copy() const { return quantile(copy_, 0.50); }
  [[nodiscard]] double p95_copy() const { return quantile(copy_, 0.95); }
  [[nodiscard]] double wait_ratio() const { return safe_div(wait_sum_, span_sum_); }

  /// copy 時間の変動係数。分散の指標として scheduler へ渡す。
  [[nodiscard]] double copy_cv() const {
    if (copy_.size() < 2) return 0.0;
    double sum = 0.0;
    for (const double v : copy_) sum += v;
    const double mean = sum / static_cast<double>(copy_.size());
    if (!(mean > 0.0)) return 0.0;
    double ss = 0.0;
    for (const double v : copy_) ss += (v - mean) * (v - mean);
    return std::sqrt(ss / static_cast<double>(copy_.size())) / mean;
  }

private:
  static double quantile(const std::deque<double>& d, double q) {
    if (d.empty()) return 0.0;
    std::vector<double> v(d.begin(), d.end());
    std::sort(v.begin(), v.end());
    auto idx = static_cast<std::size_t>(std::ceil(q * static_cast<double>(v.size())));
    if (idx == 0) idx = 1;
    if (idx > v.size()) idx = v.size();
    return v[idx - 1];
  }

  std::size_t        n_;
  std::deque<double> copy_, compute_, wait_, span_;
  double             wait_sum_ = 0.0;
  double             span_sum_ = 0.0;
};

std::vector<SimChunk> load_plan_chunks(const std::string& plan_path,
                                       std::string& model_path_out) {
  const json::Value pj = json::Value::parse_file(plan_path);
  const ChunkPlan   plan = plan_from_json(pj);
  if (model_path_out.empty()) model_path_out = plan.model_path;

  std::vector<SimChunk> out;
  out.reserve(plan.chunks.size());
  // 実行順（exec_sequence）で流す。ALSR の要点は「実行順で先読みする」こと。
  for (const std::size_t idx : plan.exec_sequence) {
    const Chunk& c = plan.chunks[idx];
    SimChunk s;
    s.chunk_id      = c.chunk_id;
    s.source_offset = c.source_offset;
    s.bytes         = c.bytes;
    s.expected_crc  = c.crc32c;
    s.crc_known     = c.checksum_computed;
    s.per_token     = c.is_per_token();
    out.push_back(s);
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// SimConfig
// ---------------------------------------------------------------------------
void SimConfig::validate() const {
  if (ring_depth == 0) fail(ErrorKind::Invalid, "ring_depth must be >= 1");
  if (prefetch_depth == 0) fail(ErrorKind::Invalid, "prefetch_depth must be >= 1");
  if (prefetch_depth > ring_depth) {
    fail(ErrorKind::Invalid, "prefetch_depth must be <= ring_depth");
  }
  if (repeat_passes == 0) fail(ErrorKind::Invalid, "repeat_passes must be >= 1");
  if (scheduler != "fixed" && scheduler != "adaptive") {
    fail(ErrorKind::Invalid, "scheduler must be 'fixed' or 'adaptive'");
  }
  if (plan_path.empty()) {
    if (synthetic_chunk_bytes == 0 || synthetic_total_bytes == 0) {
      fail(ErrorKind::Invalid,
           "synthetic mode requires synthetic_total_bytes > 0 and "
           "synthetic_chunk_bytes > 0");
    }
  }
}

json::Value SimConfig::to_json() const {
  json::Value j = json::Value::make_object();
  j.set("label", label);
  j.set("seed", seed);

  json::Value src = json::Value::make_object();
  src.set("plan_path", plan_path);
  src.set("model_path", model_path);
  src.set("host_store_mode", std::string(to_string(host_store_mode)));
  src.set("synthetic_total_bytes", synthetic_total_bytes);
  src.set("synthetic_chunk_bytes", synthetic_chunk_bytes);
  j.set("source", std::move(src));

  json::Value ring = json::Value::make_object();
  ring.set("ring_depth", ring_depth);
  ring.set("prefetch_depth", prefetch_depth);
  j.set("ring", std::move(ring));

  json::Value sch = json::Value::make_object();
  sch.set("type", scheduler);
  sch.set("stats_window", stats_window);
  sch.set("fixed_chunk_size", fixed.chunk_size);
  sch.set("fixed_prefetch_depth", fixed.prefetch_depth);
  sch.set("adaptive_initial_chunk_size", adaptive.initial_chunk_size);
  sch.set("adaptive_initial_prefetch_depth", adaptive.initial_prefetch_depth);
  sch.set("adaptive_min_prefetch_depth", adaptive.min_prefetch_depth);
  sch.set("adaptive_max_prefetch_depth", adaptive.max_prefetch_depth);
  sch.set("adaptive_min_chunk_size", adaptive.min_chunk_size);
  sch.set("adaptive_max_chunk_size", adaptive.max_chunk_size);
  sch.set("adaptive_min_interval_chunks", adaptive.min_interval_chunks);
  sch.set("adaptive_starvation_wait_ratio", adaptive.starvation_wait_ratio);
  sch.set("adaptive_grow_ratio", adaptive.grow_ratio);
  sch.set("adaptive_pressure_ratio", adaptive.pressure_ratio);
  j.set("scheduler", std::move(sch));

  json::Value be = json::Value::make_object();
  be.set("h2d", backend.h2d.to_json());
  be.set("prepare", backend.prepare.to_json());
  be.set("consume_bytes_per_s", backend.consume_bytes_per_s);
  be.set("consume_jitter_fraction", backend.consume_jitter_fraction);
  be.set("device_memory_bytes", backend.device_memory_bytes);
  be.set("pinned_limit_bytes", backend.pinned_limit_bytes);
  be.set("verify_checksums", backend.verify_checksums);
  j.set("backend", std::move(be));

  json::Value run = json::Value::make_object();
  run.set("repeat_passes", repeat_passes);
  run.set("max_virtual_seconds", max_virtual_seconds);
  run.set("max_chunks", max_chunks);
  run.set("per_token_bytes", per_token_bytes);
  run.set("resident_bytes", resident_bytes);
  run.set("exclude_non_per_token_chunks", exclude_non_per_token_chunks);
  run.set("cancel_at_chunk", cancel_at_chunk);
  run.set("materialize_data", materialize_data);
  j.set("run", std::move(run));

  return j;
}

SimConfig SimConfig::from_json(const json::Value& v) {
  SimConfig c;
  c.label = v.get_string("label", c.label);
  c.seed  = static_cast<u64>(v.get_int("seed", static_cast<i64>(c.seed)));

  if (const json::Value* s = v.find("source")) {
    c.plan_path  = s->get_string("plan_path", "");
    c.model_path = s->get_string("model_path", "");
    c.host_store_mode =
        host_store_mode_from_string(s->get_string("host_store_mode", "mmap"));
    c.synthetic_total_bytes = static_cast<Bytes>(
        s->get_int("synthetic_total_bytes",
                   static_cast<i64>(c.synthetic_total_bytes)));
    c.synthetic_chunk_bytes = static_cast<Bytes>(
        s->get_int("synthetic_chunk_bytes",
                   static_cast<i64>(c.synthetic_chunk_bytes)));
  }
  if (const json::Value* r = v.find("ring")) {
    c.ring_depth = static_cast<std::size_t>(
        r->get_int("ring_depth", static_cast<i64>(c.ring_depth)));
    c.prefetch_depth = static_cast<std::size_t>(
        r->get_int("prefetch_depth", static_cast<i64>(c.prefetch_depth)));
  }
  if (const json::Value* s = v.find("scheduler")) {
    c.scheduler    = s->get_string("type", c.scheduler);
    c.stats_window = static_cast<std::size_t>(
        s->get_int("stats_window", static_cast<i64>(c.stats_window)));
    c.fixed.chunk_size = static_cast<Bytes>(
        s->get_int("fixed_chunk_size", static_cast<i64>(c.fixed.chunk_size)));
    c.fixed.prefetch_depth = static_cast<std::size_t>(s->get_int(
        "fixed_prefetch_depth", static_cast<i64>(c.fixed.prefetch_depth)));
    auto& a = c.adaptive;
    a.initial_chunk_size = static_cast<Bytes>(s->get_int(
        "adaptive_initial_chunk_size", static_cast<i64>(a.initial_chunk_size)));
    a.initial_prefetch_depth = static_cast<std::size_t>(
        s->get_int("adaptive_initial_prefetch_depth",
                   static_cast<i64>(a.initial_prefetch_depth)));
    a.min_prefetch_depth = static_cast<std::size_t>(
        s->get_int("adaptive_min_prefetch_depth",
                   static_cast<i64>(a.min_prefetch_depth)));
    a.max_prefetch_depth = static_cast<std::size_t>(
        s->get_int("adaptive_max_prefetch_depth",
                   static_cast<i64>(a.max_prefetch_depth)));
    a.min_chunk_size = static_cast<Bytes>(s->get_int(
        "adaptive_min_chunk_size", static_cast<i64>(a.min_chunk_size)));
    a.max_chunk_size = static_cast<Bytes>(s->get_int(
        "adaptive_max_chunk_size", static_cast<i64>(a.max_chunk_size)));
    a.min_interval_chunks = static_cast<u64>(s->get_int(
        "adaptive_min_interval_chunks", static_cast<i64>(a.min_interval_chunks)));
    a.starvation_wait_ratio =
        s->get_real("adaptive_starvation_wait_ratio", a.starvation_wait_ratio);
    a.grow_ratio = s->get_real("adaptive_grow_ratio", a.grow_ratio);
    a.pressure_ratio = s->get_real("adaptive_pressure_ratio", a.pressure_ratio);
  }
  if (const json::Value* b = v.find("backend")) {
    if (const json::Value* h = b->find("h2d")) {
      c.backend.h2d = BandwidthProfile::from_json(*h);
    }
    if (const json::Value* p = b->find("prepare")) {
      c.backend.prepare = BandwidthProfile::from_json(*p);
    }
    c.backend.consume_bytes_per_s =
        b->get_real("consume_bytes_per_s", c.backend.consume_bytes_per_s);
    c.backend.consume_jitter_fraction =
        b->get_real("consume_jitter_fraction", c.backend.consume_jitter_fraction);
    c.backend.device_memory_bytes = static_cast<Bytes>(b->get_int(
        "device_memory_bytes", static_cast<i64>(c.backend.device_memory_bytes)));
    c.backend.pinned_limit_bytes = static_cast<Bytes>(b->get_int(
        "pinned_limit_bytes", static_cast<i64>(c.backend.pinned_limit_bytes)));
    c.backend.verify_checksums =
        b->get_bool("verify_checksums", c.backend.verify_checksums);
  }
  if (const json::Value* r = v.find("run")) {
    c.repeat_passes = static_cast<u64>(
        r->get_int("repeat_passes", static_cast<i64>(c.repeat_passes)));
    c.max_virtual_seconds =
        r->get_real("max_virtual_seconds", c.max_virtual_seconds);
    c.max_chunks =
        static_cast<u64>(r->get_int("max_chunks", static_cast<i64>(c.max_chunks)));
    c.per_token_bytes = static_cast<Bytes>(
        r->get_int("per_token_bytes", static_cast<i64>(c.per_token_bytes)));
    c.resident_bytes = static_cast<Bytes>(
        r->get_int("resident_bytes", static_cast<i64>(c.resident_bytes)));
    c.exclude_non_per_token_chunks = r->get_bool(
        "exclude_non_per_token_chunks", c.exclude_non_per_token_chunks);
    c.cancel_at_chunk = static_cast<u64>(
        r->get_int("cancel_at_chunk", static_cast<i64>(c.cancel_at_chunk)));
    c.materialize_data = r->get_bool("materialize_data", c.materialize_data);
  }
  c.backend.seed = c.seed;
  c.validate();
  return c;
}

// ---------------------------------------------------------------------------
// run_simulation
// ---------------------------------------------------------------------------
SimResult run_simulation(const SimConfig& cfg_in, const std::string& run_id,
                         MetricsWriter* writer) {
  SimConfig cfg = cfg_in;
  cfg.backend.seed = cfg.seed;
  if (!cfg.materialize_data) {
    // データを流さない以上、checksum は成立しない。黙って「成功」にしない。
    cfg.backend.verify_checksums = false;
  }
  cfg.validate();

  const auto wall_begin = std::chrono::steady_clock::now();

  // --- Chunk 列とデータ源 ---------------------------------------------
  std::vector<SimChunk> base_chunks;
  std::string model_path = cfg.model_path;
  std::unique_ptr<HostStore> store;
  const bool synthetic = cfg.plan_path.empty();

  if (!synthetic) {
    base_chunks = load_plan_chunks(cfg.plan_path, model_path);
    if (model_path.empty()) {
      fail(ErrorKind::Invalid,
           "plan に model_path が無く、config でも指定されていない");
    }
    store = std::make_unique<HostStore>(model_path, cfg.host_store_mode);
  } else {
    store = std::make_unique<HostStore>("", HostStoreMode::Synthetic,
                                        cfg.synthetic_total_bytes);
  }

  // --- VRAM 常駐分をストリーミング対象から外す (§2.4) ------------------
  // 実行順の先頭から積む。Batch=1 Dense ではどれを選んでも総転送量は同じなので、
  // 選択方針は実装の単純さで決めてよい（residency_table() のコメント参照）。
  // 毎トークン読まれない Chunk（視覚エンコーダ・投影器など）は定常状態の
  // ストリーミング対象ではない。常に「常駐扱い」として外す。
  Bytes       non_per_token_bytes = 0;
  std::size_t non_per_token_chunks = 0;
  if (!synthetic && cfg.exclude_non_per_token_chunks) {
    std::vector<SimChunk> keep;
    keep.reserve(base_chunks.size());
    for (const SimChunk& c : base_chunks) {
      if (c.per_token) {
        keep.push_back(c);
      } else {
        non_per_token_bytes += c.bytes;
        ++non_per_token_chunks;
      }
    }
    if (keep.empty()) {
      fail(ErrorKind::Invalid,
           "毎トークン流す Chunk が 1 つも無い。plan と manifest の対応を確認すること");
    }
    base_chunks.swap(keep);
  }

  Bytes       resident_actual = 0;
  std::size_t resident_chunks = 0;
  if (cfg.resident_bytes > 0) {
    if (synthetic) {
      fail(ErrorKind::Invalid,
           "resident_bytes は実 plan の Run でのみ使える"
           "（合成 Chunk には実行順の意味が無い）");
    }
    std::vector<SimChunk> streamed;
    streamed.reserve(base_chunks.size());
    for (const SimChunk& c : base_chunks) {
      if (resident_actual + c.bytes <= cfg.resident_bytes) {
        resident_actual += c.bytes;
        ++resident_chunks;
      } else {
        streamed.push_back(c);
      }
    }
    if (streamed.empty()) {
      fail(ErrorKind::Invalid,
           "resident_bytes が大きすぎて流す Chunk が残らない。"
           "この構成では ALSR が不要である（全重みが VRAM に載る）");
    }
    base_chunks.swap(streamed);
  }

  // --- Scheduler ------------------------------------------------------
  std::unique_ptr<Scheduler> sched;
  if (cfg.scheduler == "adaptive") {
    AdaptiveSchedulerConfig a = cfg.adaptive;
    // 実 plan を使う Run では Chunk サイズは計画で決まっており動かせない。
    // 黙って無視せず、設定を落として理由をログへ残す。
    a.allow_chunk_size_change = synthetic;
    a.initial_prefetch_depth  = cfg.prefetch_depth;
    sched = make_adaptive_scheduler(a);
  } else {
    FixedSchedulerConfig f = cfg.fixed;
    f.prefetch_depth = cfg.prefetch_depth;
    sched = make_fixed_scheduler(f);
  }

  auto backend = make_simulator_backend(cfg.backend);

  // --- スロット容量とメモリ上限 (NFR-03) -------------------------------
  // synthetic では scheduler が Chunk サイズを動かし得るので、実際に使う
  // scheduler の上限までスロットを取る。使わない側の設定は見ない。
  Bytes slot_capacity = 0;
  if (synthetic) {
    slot_capacity = cfg.synthetic_chunk_bytes;
    if (cfg.scheduler == "adaptive") {
      slot_capacity = std::max({slot_capacity, cfg.adaptive.initial_chunk_size,
                                cfg.adaptive.max_chunk_size});
    } else {
      slot_capacity = std::max(slot_capacity, cfg.fixed.chunk_size);
    }
  } else {
    for (const auto& c : base_chunks) slot_capacity = std::max(slot_capacity, c.bytes);
  }
  if (slot_capacity == 0) fail(ErrorKind::Invalid, "slot capacity resolved to 0");

  const Bytes ring_bytes = slot_capacity * cfg.ring_depth;
  if (cfg.backend.pinned_limit_bytes > 0 &&
      ring_bytes > cfg.backend.pinned_limit_bytes) {
    std::ostringstream os;
    os << "ring_depth(" << cfg.ring_depth << ") x slot_capacity(" << slot_capacity
       << ") = " << ring_bytes << " が pinned 上限 "
       << cfg.backend.pinned_limit_bytes
       << " を超える。NFR-03 によりここで停止する";
    fail(ErrorKind::Invalid, os.str());
  }
  // 常駐重み + リングバッファが VRAM に収まらなければ物理的に成立しない。
  // 常駐分を数えずに「収まった」と言わない。
  const Bytes device_needed = ring_bytes + cfg.resident_bytes;
  if (cfg.backend.device_memory_bytes > 0 &&
      device_needed > cfg.backend.device_memory_bytes) {
    std::ostringstream os;
    os << "VRAM 不足: 常駐重み " << cfg.resident_bytes << " + ring "
       << ring_bytes << " = " << device_needed << " が device memory "
       << cfg.backend.device_memory_bytes << " を超える";
    fail(ErrorKind::Invalid, os.str());
  }

  RingBuffer ring(cfg.ring_depth, slot_capacity);

  // --- 実行 ------------------------------------------------------------
  SimResult result;
  result.scheduler_config = sched->config_json();
  result.backend_config   = backend->config_json();

  struct InFlight {
    SlotRef     ref;
    u64         seq = 0;
    SimChunk    chunk;
    double      prepare_start = 0.0;
    double      prepare_end = 0.0;
    double      copy_start = 0.0;
    double      copy_end = 0.0;
    std::size_t queue_depth_before = 0;
    Bytes       chunk_size_setting = 0;
    std::size_t prefetch_setting = 0;
    std::string action;
  };

  std::deque<InFlight>  inflight;
  std::vector<double>   slot_released(cfg.ring_depth, 0.0);
  double producer_busy = 0.0;
  double transfer_busy = 0.0;
  double consumer_busy = 0.0;

  std::vector<std::pair<double, double>> copy_intervals;
  std::vector<std::pair<double, double>> compute_intervals;

  Window window(cfg.stats_window);
  Bytes  current_chunk_size =
      (cfg.scheduler == "adaptive") ? cfg.adaptive.initial_chunk_size
                                    : cfg.fixed.chunk_size;
  std::size_t current_prefetch = cfg.prefetch_depth;

  u64   produced = 0;         ///< 投入済み Chunk 数
  u64   consumed = 0;         ///< 消費済み Chunk 数
  u64   expected_next_seq = 0;
  Bytes total_bytes = 0;
  Bytes peak_device_bytes = 0;
  u64   checksum_failures = 0;
  u64   order_violations = 0;
  double starvation_total = 0.0;
  double starvation_max = 0.0;
  u64    starvation_events = 0;
  std::vector<double> wait_series;

  const auto retire_one = [&]() {
    InFlight f = inflight.front();
    inflight.pop_front();

    ring.transition(f.ref, SlotState::Resident, SlotState::Consuming);

    const double compute_start = std::max(consumer_busy, f.copy_end);
    const double wait          = compute_start - consumer_busy;
    const double compute_dur   = backend->compute_time_s(f.chunk.bytes, compute_start);
    const double compute_end   = compute_start + compute_dur;

    bool checksum_ok = true;
    if (cfg.materialize_data) {
      const u32 crc = backend->consume(ring.host_buffer(f.ref), f.chunk.bytes);
      if (cfg.backend.verify_checksums && f.chunk.crc_known) {
        checksum_ok = (crc == f.chunk.expected_crc);
      }
    }
    if (!checksum_ok) ++checksum_failures;

    if (f.seq != expected_next_seq) ++order_violations;
    ++expected_next_seq;

    BufferSlot& slot = ring.slot(f.ref);
    slot.compute_start_s = compute_start;
    slot.compute_end_s   = compute_end;

    ChunkMetric m;
    m.chunk_id           = f.chunk.chunk_id;
    m.bytes              = f.chunk.bytes;
    m.queue_depth_before = f.queue_depth_before;
    m.queue_depth_after  = inflight.size();
    m.prepare_ms         = (f.prepare_end - f.prepare_start) * 1e3;
    m.h2d_ms             = (f.copy_end - f.copy_start) * 1e3;
    m.compute_ms         = compute_dur * 1e3;
    m.wait_ms            = wait * 1e3;
    m.effective_h2d_gbps =
        safe_div(static_cast<double>(f.chunk.bytes), (f.copy_end - f.copy_start)) / 1e9;
    m.checksum_ok           = checksum_ok;
    m.scheduler_action      = f.action;
    m.t_prepare_start_s     = f.prepare_start;
    m.t_copy_start_s        = f.copy_start;
    m.t_copy_end_s          = f.copy_end;
    m.t_compute_start_s     = compute_start;
    m.t_compute_end_s       = compute_end;
    m.slot_id               = f.ref.slot_id;
    m.slot_generation       = f.ref.generation;
    m.chunk_size_setting    = f.chunk_size_setting;
    m.prefetch_depth_setting = f.prefetch_setting;

    consumer_busy = compute_end;
    compute_intervals.emplace_back(compute_start, compute_end);
    copy_intervals.emplace_back(f.copy_start, f.copy_end);

    // ウォームアップ（最初の ring_depth 個）は立ち上がりなので枯渇に数えない。
    wait_series.push_back(wait);
    if (f.seq >= cfg.ring_depth && wait > 1e-12) {
      ++starvation_events;
      starvation_total += wait;
      starvation_max = std::max(starvation_max, wait);
    }

    // consumer エンジンの時間軸は [wait][compute] の連続で埋まる。
    // 窓内の待機率 = Σwait / Σ(wait + compute)。
    window.push(m.h2d_ms, m.compute_ms, wait * 1e3, (wait + compute_dur) * 1e3);

    ring.transition(f.ref, SlotState::Consuming, SlotState::Free);
    slot_released[f.ref.slot_id] = compute_end;

    total_bytes += f.chunk.bytes;
    ++consumed;

    result.metrics.push_back(m);
    if (writer != nullptr) writer->write(run_id, m);
  };

  // Chunk 列を repeat_passes 周する。synthetic では毎周同じ列を再生成する。
  std::vector<u8> synth_scratch(64 * 1024);
  Bytes synth_cursor = 0;
  u64   seq = 0;
  bool  stop = false;

  for (u64 pass = 0; pass < cfg.repeat_passes && !stop; ++pass) {
    std::size_t plan_idx = 0;
    synth_cursor = 0;

    for (;;) {
      // --- 候補を用意する（FR-06「次Chunk」の入力） --------------------
      // Scheduler が実際に選ぶまで消費しない。初期版は常に先頭を返すが、
      // 選択の責務は Scheduler 側にある。
      std::vector<std::size_t> pending;
      if (synthetic) {
        if (synth_cursor >= cfg.synthetic_total_bytes) break;
        pending.push_back(0);  // 合成は逐次生成なので候補は 1 つ
      } else {
        if (plan_idx >= base_chunks.size()) break;
        // 先読み候補は残り全部（実行順）。窓を切らずに渡す。
        for (std::size_t k = plan_idx; k < base_chunks.size(); ++k) {
          pending.push_back(k);
        }
      }

      if (cfg.max_chunks != 0 && produced >= cfg.max_chunks) { stop = true; break; }
      if (cfg.max_virtual_seconds > 0.0 && consumer_busy >= cfg.max_virtual_seconds) {
        stop = true;
        break;
      }

      // --- Scheduler ---------------------------------------------------
      SchedulerInput in;
      in.pending_chunk_indices  = pending;
      in.chunk_index            = produced;
      in.queue_depth            = inflight.size();
      in.ring_depth             = cfg.ring_depth;
      in.free_slots             = ring.free_count();
      in.p50_compute_ms         = window.p50_compute();
      in.p50_copy_ms            = window.p50_copy();
      in.p95_copy_ms            = window.p95_copy();
      in.recent_copy_cv         = window.copy_cv();
      in.wait_ratio             = window.wait_ratio();
      in.starvation_detected =
          (consumed > cfg.ring_depth) && (window.wait_ratio() > 0.0);
      in.vram_used_bytes        = slot_capacity * ring.in_flight();
      in.vram_capacity_bytes    = cfg.backend.device_memory_bytes;
      in.pinned_used_bytes      = ring_bytes;
      in.pinned_limit_bytes     = cfg.backend.pinned_limit_bytes;
      in.current_chunk_size     = current_chunk_size;
      in.current_prefetch_depth = current_prefetch;

      const SchedulerDecision d = sched->decide(in);
      if (d.changed) {
        std::ostringstream os;
        os << "chunk " << produced << ": " << d.action << " — " << d.reason;
        result.scheduler_change_log.push_back(os.str());
      }
      if (synthetic && d.chunk_size > 0) current_chunk_size = d.chunk_size;
      current_prefetch = std::clamp<std::size_t>(d.prefetch_depth, 1, cfg.ring_depth);

      // --- Scheduler が選んだ Chunk を取り出す -------------------------
      if (d.next_chunk_index == SchedulerDecision::kNoChunk) {
        fail(ErrorKind::State,
             "scheduler が候補ありの状態で next_chunk_index を返さなかった");
      }
      SimChunk next;
      if (synthetic) {
        const Bytes remain = cfg.synthetic_total_bytes - synth_cursor;
        next.bytes         = std::min(current_chunk_size, remain);
        next.source_offset = synth_cursor;
        next.chunk_id      = produced;
        next.crc_known     = cfg.backend.verify_checksums;
        synth_cursor += next.bytes;
      } else {
        if (d.next_chunk_index >= base_chunks.size()) {
          fail(ErrorKind::State, "scheduler が範囲外の chunk index を返した");
        }
        next = base_chunks[d.next_chunk_index];
        // 初期版の Scheduler は実行順の先頭しか返さないので、消費位置は
        // 単調に進む。並べ替えを許すなら未消費集合の管理が要る。
        if (d.next_chunk_index != plan_idx) {
          fail(ErrorKind::Unsupported,
               "Chunk の並べ替えは未実装（MoE の投機的プリフェッチで必要になる）");
        }
        ++plan_idx;
      }

      // --- スロット確保（必要なら古いものを消費して空ける） -------------
      while (inflight.size() >= current_prefetch || !ring.has_free()) {
        if (inflight.empty()) {
          fail(ErrorKind::State,
               "no in-flight chunk to retire but no free slot — ring 状態不整合");
        }
        retire_one();
      }

      const SlotRef ref = ring.acquire_free();  // FREE -> FILLING
      BufferSlot& slot = ring.slot(ref);
      slot.chunk_id      = next.chunk_id;
      slot.source_offset = next.source_offset;
      slot.payload_bytes = next.bytes;

      const double prep_start =
          std::max(producer_busy, slot_released[ref.slot_id]);
      const double prep_dur = backend->prepare_time_s(next.bytes, prep_start);
      const double prep_end = prep_start + prep_dur;
      producer_busy = prep_end;

      // 実データを読む（Host Store → slot バッファ）。ここは実測。
      if (cfg.materialize_data) {
        store->read_into(next.source_offset, next.bytes, ring.host_buffer(ref));
      }

      if (cfg.materialize_data && synthetic && next.crc_known) {
        // synthetic では期待値を「生成規則から直接」求める。
        // slot バッファとは別経路で作ることで、リング/オフセット処理の
        // 取り違えを検出できる。作業領域は使い回して大きな確保を避ける。
        Crc32c c;
        for (Bytes done = 0; done < next.bytes;) {
          const Bytes n = std::min<Bytes>(synth_scratch.size(), next.bytes - done);
          for (Bytes k = 0; k < n; ++k) {
            synth_scratch[static_cast<std::size_t>(k)] =
                HostStore::synthetic_byte(next.source_offset + done + k);
          }
          c.update(synth_scratch.data(), static_cast<std::size_t>(n));
          done += n;
        }
        next.expected_crc = c.value();
      }

      slot.fill_start_s = prep_start;
      slot.fill_end_s   = prep_end;
      ring.transition(ref, SlotState::Filling, SlotState::Ready);

      const double copy_start = std::max(transfer_busy, prep_end);
      const double copy_dur   = backend->h2d_time_s(next.bytes, copy_start);
      const double copy_end   = copy_start + copy_dur;
      transfer_busy = copy_end;

      slot.copy_start_s = copy_start;
      slot.copy_end_s   = copy_end;
      ring.transition(ref, SlotState::Ready, SlotState::Copying);
      ring.transition(ref, SlotState::Copying, SlotState::Resident);

      InFlight f;
      f.ref                = ref;
      f.seq                = seq++;
      f.chunk              = next;
      f.prepare_start      = prep_start;
      f.prepare_end        = prep_end;
      f.copy_start         = copy_start;
      f.copy_end           = copy_end;
      f.queue_depth_before = inflight.size();
      f.chunk_size_setting = current_chunk_size;
      f.prefetch_setting   = current_prefetch;
      f.action             = d.action;
      inflight.push_back(f);
      ++produced;

      peak_device_bytes =
          std::max(peak_device_bytes, slot_capacity * ring.in_flight());

      if (cfg.cancel_at_chunk != 0 && produced >= cfg.cancel_at_chunk) {
        result.cancelled = true;
        stop = true;
        break;
      }
    }
  }

  if (result.cancelled) {
    ring.cancel_all();
    inflight.clear();
    const auto hist = ring.state_histogram();
    result.cancel_released_all_slots =
        (hist[static_cast<std::size_t>(SlotState::Free)] == cfg.ring_depth);
  } else {
    while (!inflight.empty()) retire_one();
  }

  // --- 集計 -------------------------------------------------------------
  RunSummary& s = result.summary;
  s.run_id         = run_id;
  s.scheduler_name = sched->name();
  s.backend_name   = backend->name();
  s.chunk_count    = consumed;
  s.total_bytes    = total_bytes;
  s.virtual_seconds = consumer_busy;

  const auto wall_end = std::chrono::steady_clock::now();
  s.wall_seconds =
      std::chrono::duration<double>(wall_end - wall_begin).count();

  s.chunks_per_s = safe_div(static_cast<double>(consumed), consumer_busy);

  double copy_time_total = 0.0;
  std::vector<double> copy_ms, compute_ms, prepare_ms, wait_ms_v;
  copy_ms.reserve(result.metrics.size());
  for (const auto& m : result.metrics) {
    copy_ms.push_back(m.h2d_ms);
    compute_ms.push_back(m.compute_ms);
    prepare_ms.push_back(m.prepare_ms);
    wait_ms_v.push_back(m.wait_ms);
    copy_time_total += m.h2d_ms / 1e3;
  }
  s.copy_ms    = compute_percentiles(copy_ms);
  s.compute_ms = compute_percentiles(compute_ms);
  s.prepare_ms = compute_percentiles(prepare_ms);
  s.wait_ms    = compute_percentiles(wait_ms_v);

  s.effective_h2d_gbps =
      safe_div(static_cast<double>(total_bytes), copy_time_total) / 1e9;

  double wait_total = 0.0;
  for (const double w : wait_series) wait_total += w;
  s.gpu_wait_ratio = safe_div(wait_total, consumer_busy);

  std::sort(copy_intervals.begin(), copy_intervals.end());
  std::sort(compute_intervals.begin(), compute_intervals.end());
  s.copy_compute_overlap_ratio =
      safe_div(overlap_measure(copy_intervals, compute_intervals), copy_time_total);

  s.starvation_events = starvation_events;
  s.starvation_total_s = starvation_total;
  s.starvation_max_s = starvation_max;

  if (wait_series.size() >= 4) {
    const std::size_t n = wait_series.size();
    const std::size_t half = n / 2;
    double first = 0.0;
    double second = 0.0;
    for (std::size_t i = 0; i < half; ++i) first += wait_series[i];
    for (std::size_t i = half; i < n; ++i) second += wait_series[i];
    s.starvation_progression_ratio =
        safe_div(second / static_cast<double>(n - half),
                 first / static_cast<double>(half));

    // 終盤 1/4。consumer の時間軸は [wait][compute] の連続なので、
    // その区間の span は Σ(wait + compute) で厳密に求まる。
    const std::size_t q = n - n / 4;
    double late_wait = 0.0;
    double late_span = 0.0;
    for (std::size_t i = q; i < n; ++i) {
      late_wait += wait_series[i];
      late_span += wait_series[i] + result.metrics[i].compute_ms / 1e3;
    }
    s.late_wait_ratio = safe_div(late_wait, late_span);
  }

  s.per_token_bytes = cfg.per_token_bytes;
  if (cfg.per_token_bytes > 0) {
    const double tokens =
        static_cast<double>(total_bytes) / static_cast<double>(cfg.per_token_bytes);
    s.tokens_per_s = safe_div(tokens, consumer_busy);
  }

  s.pinned_bytes      = ring_bytes;
  s.peak_host_bytes   = ring_bytes;
  s.peak_device_bytes = peak_device_bytes;

  s.checksum_failures = checksum_failures;
  s.order_violations  = order_violations;
  s.missing_chunks    = produced - consumed;
  s.duplicate_chunks  = 0;
  s.unhandled_errors  = 0;
  s.scheduler_changes = sched->change_count();
  s.ring_wrap_arounds = ring.stats().wrap_arounds;
  s.data_materialized = cfg.materialize_data;
  s.resident_bytes = resident_actual + non_per_token_bytes;
  s.resident_chunk_count = resident_chunks + non_per_token_chunks;
  s.non_per_token_bytes = non_per_token_bytes;

  // ここでは environment_recorded は false のまま。証拠を書き出す側
  // (alsr-sim) が記録後に再評価する。空の failures と pass=false という
  // 誤解を招く状態を残さないため、必ず一度は評価しておく。
  evaluate_mandatory_gates(s);

  return result;
}

}  // namespace alsr
