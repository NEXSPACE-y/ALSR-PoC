#include "alsr/scheduler.hpp"

#include <algorithm>
#include <sstream>

namespace alsr {
namespace {

class FixedScheduler final : public Scheduler {
public:
  explicit FixedScheduler(FixedSchedulerConfig cfg) : cfg_(cfg) {}

  [[nodiscard]] std::string name() const override { return "fixed"; }

  SchedulerDecision decide(const SchedulerInput&) override {
    SchedulerDecision d;
    d.chunk_size = cfg_.chunk_size;
    d.prefetch_depth = cfg_.prefetch_depth;
    d.action = "keep";
    d.reason = "fixed scheduler: 制御しない（比較基準）";
    d.changed = false;
    return d;
  }

  [[nodiscard]] u64 change_count() const noexcept override { return 0; }

  [[nodiscard]] json::Value config_json() const override {
    json::Value j = json::Value::make_object();
    j.set("type", "fixed");
    j.set("chunk_size", cfg_.chunk_size);
    j.set("prefetch_depth", cfg_.prefetch_depth);
    return j;
  }

private:
  FixedSchedulerConfig cfg_;
};

class AdaptiveScheduler final : public Scheduler {
public:
  explicit AdaptiveScheduler(AdaptiveSchedulerConfig cfg)
      : cfg_(cfg),
        chunk_size_(cfg.initial_chunk_size),
        prefetch_depth_(cfg.initial_prefetch_depth) {
    if (cfg_.min_prefetch_depth == 0) {
      fail(ErrorKind::Invalid, "min_prefetch_depth must be >= 1");
    }
    if (cfg_.min_prefetch_depth > cfg_.max_prefetch_depth) {
      fail(ErrorKind::Invalid, "min_prefetch_depth > max_prefetch_depth");
    }
    prefetch_depth_ = std::clamp(prefetch_depth_, cfg_.min_prefetch_depth,
                                 cfg_.max_prefetch_depth);
  }

  [[nodiscard]] std::string name() const override { return "adaptive"; }

  // §8.3 の制御則 + ヒステリシス
  SchedulerDecision decide(const SchedulerInput& in) override {
    SchedulerDecision d;
    d.chunk_size = chunk_size_;
    d.prefetch_depth = prefetch_depth_;
    d.action = "keep";
    d.transfer_priority = 0;

    const bool pressure =
        (in.vram_capacity_bytes > 0 &&
         static_cast<double>(in.vram_used_bytes) >
             cfg_.pressure_ratio * static_cast<double>(in.vram_capacity_bytes)) ||
        (in.pinned_limit_bytes > 0 &&
         static_cast<double>(in.pinned_used_bytes) >
             cfg_.pressure_ratio * static_cast<double>(in.pinned_limit_bytes));

    const bool starving =
        in.starvation_detected || in.wait_ratio > cfg_.starvation_wait_ratio;

    // 枯渇は最優先。ヒステリシスより先に転送優先度だけは上げる。
    if (starving) d.transfer_priority = 1;

    // 起動直後はまだ一度も変更していないので、ヒステリシスの対象外。
    const bool cooled_down =
        !has_changed_ ||
        in.chunk_index >= last_change_chunk_ + cfg_.min_interval_chunks;

    if (!cooled_down) {
      std::ostringstream os;
      os << "hysteresis: 直近の変更から " << (in.chunk_index - last_change_chunk_)
         << " chunk しか経っていない (必要 " << cfg_.min_interval_chunks << ")";
      d.reason = os.str();
      if (starving) d.action = "emergency";
      return d;
    }

    if (starving) {
      // 供給が追いつかない → 先読みを深くし、Chunk を細かくして
      // 立ち上がりの待ち時間を減らす。
      const std::size_t new_depth =
          std::min(prefetch_depth_ + 1, cfg_.max_prefetch_depth);
      const Bytes new_size = cfg_.allow_chunk_size_change
                                 ? std::max(next_lower_size(chunk_size_),
                                            cfg_.min_chunk_size)
                                 : chunk_size_;
      if (new_depth != prefetch_depth_ || new_size != chunk_size_) {
        std::ostringstream os;
        os << "starvation: wait_ratio=" << in.wait_ratio << " > "
           << cfg_.starvation_wait_ratio << " → prefetch_depth " << prefetch_depth_
           << "->" << new_depth << ", chunk_size " << chunk_size_ << "->" << new_size;
        prefetch_depth_ = new_depth;
        chunk_size_ = new_size;
        commit(in, d, "shrink_chunk", os.str());
      } else {
        d.action = "emergency";
        d.reason = "starvation だが prefetch/chunk は既に端。物理的に供給不足の疑い";
      }
      return d;
    }

    if (pressure) {
      if (prefetch_depth_ > cfg_.min_prefetch_depth) {
        std::ostringstream os;
        os << "memory pressure: vram " << in.vram_used_bytes << "/"
           << in.vram_capacity_bytes << ", pinned " << in.pinned_used_bytes << "/"
           << in.pinned_limit_bytes << " → prefetch_depth " << prefetch_depth_
           << "->" << (prefetch_depth_ - 1);
        --prefetch_depth_;
        commit(in, d, "shallow", os.str());
      } else {
        d.reason = "memory pressure だが prefetch_depth は既に下限";
      }
      return d;
    }

    // 余裕がある。Chunk を大きくして発行・同期オーバーヘッドを減らす。
    const bool queue_stable =
        in.queue_depth >= prefetch_depth_ && in.recent_copy_cv < 0.25;
    if (cfg_.allow_chunk_size_change && queue_stable && in.p50_compute_ms > 0.0 &&
        in.p95_copy_ms < cfg_.grow_ratio * in.p50_compute_ms) {
      const Bytes new_size =
          std::min(next_higher_size(chunk_size_), cfg_.max_chunk_size);
      if (new_size != chunk_size_) {
        std::ostringstream os;
        os << "headroom: p95_copy=" << in.p95_copy_ms << "ms < "
           << cfg_.grow_ratio << " * p50_compute=" << in.p50_compute_ms
           << "ms かつ queue 安定 → chunk_size " << chunk_size_ << "->" << new_size;
        chunk_size_ = new_size;
        commit(in, d, "grow_chunk", os.str());
        return d;
      }
    }

    d.reason = "変更条件を満たさない (keep)";
    return d;
  }

  [[nodiscard]] u64 change_count() const noexcept override { return changes_; }

  [[nodiscard]] json::Value config_json() const override {
    json::Value j = json::Value::make_object();
    j.set("type", "adaptive");
    j.set("initial_chunk_size", cfg_.initial_chunk_size);
    j.set("initial_prefetch_depth", cfg_.initial_prefetch_depth);
    j.set("min_prefetch_depth", cfg_.min_prefetch_depth);
    j.set("max_prefetch_depth", cfg_.max_prefetch_depth);
    j.set("min_chunk_size", cfg_.min_chunk_size);
    j.set("max_chunk_size", cfg_.max_chunk_size);
    j.set("min_interval_chunks", cfg_.min_interval_chunks);
    j.set("starvation_wait_ratio", cfg_.starvation_wait_ratio);
    j.set("grow_ratio", cfg_.grow_ratio);
    j.set("pressure_ratio", cfg_.pressure_ratio);
    j.set("allow_chunk_size_change", cfg_.allow_chunk_size_change);
    return j;
  }

private:
  void commit(const SchedulerInput& in, SchedulerDecision& d,
              const char* action, std::string reason) {
    d.chunk_size = chunk_size_;
    d.prefetch_depth = prefetch_depth_;
    d.action = action;
    d.reason = std::move(reason);
    d.changed = true;
    last_change_chunk_ = in.chunk_index;
    has_changed_ = true;
    ++changes_;
  }

  AdaptiveSchedulerConfig cfg_;
  Bytes       chunk_size_;
  std::size_t prefetch_depth_;
  u64         last_change_chunk_ = 0;
  bool        has_changed_ = false;
  u64         changes_ = 0;
};

}  // namespace

const std::vector<Bytes>& chunk_size_ladder() {
  static const std::vector<Bytes> kLadder = {
      16 * MiB, 32 * MiB, 64 * MiB, 128 * MiB, 256 * MiB, 512 * MiB, 1024 * MiB};
  return kLadder;
}

Bytes next_higher_size(Bytes current) {
  const auto& l = chunk_size_ladder();
  for (std::size_t i = 0; i < l.size(); ++i) {
    if (l[i] > current) return l[i];
  }
  return l.back() > current ? l.back() : current;
}

Bytes next_lower_size(Bytes current) {
  const auto& l = chunk_size_ladder();
  for (std::size_t i = l.size(); i-- > 0;) {
    if (l[i] < current) return l[i];
  }
  return l.front() < current ? l.front() : current;
}

std::unique_ptr<Scheduler> make_fixed_scheduler(const FixedSchedulerConfig& cfg) {
  return std::make_unique<FixedScheduler>(cfg);
}

std::unique_ptr<Scheduler> make_adaptive_scheduler(
    const AdaptiveSchedulerConfig& cfg) {
  return std::make_unique<AdaptiveScheduler>(cfg);
}

}  // namespace alsr
