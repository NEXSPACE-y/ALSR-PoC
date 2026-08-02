#include "alsr/bandwidth.hpp"

#include <cmath>
#include <sstream>

namespace alsr {

void BandwidthProfile::validate() const {
  if (!(bytes_per_s > 0.0) || !std::isfinite(bytes_per_s)) {
    fail(ErrorKind::Invalid, "bandwidth bytes_per_s must be finite and > 0");
  }
  if (fixed_latency_s < 0.0 || !std::isfinite(fixed_latency_s)) {
    fail(ErrorKind::Invalid, "fixed_latency_s must be finite and >= 0");
  }
  if (jitter_fraction < 0.0 || jitter_fraction >= 1.0) {
    fail(ErrorKind::Invalid, "jitter_fraction must be in [0, 1)");
  }
  if (pause_period_s < 0.0 || pause_duration_s < 0.0) {
    fail(ErrorKind::Invalid, "pause_period_s / pause_duration_s must be >= 0");
  }
  if (pause_period_s > 0.0 && pause_duration_s >= pause_period_s) {
    fail(ErrorKind::Invalid,
         "pause_duration_s >= pause_period_s では常時停止になる");
  }
}

json::Value BandwidthProfile::to_json() const {
  json::Value j = json::Value::make_object();
  j.set("bytes_per_s", bytes_per_s);
  j.set("gb_per_s", bytes_per_s / 1e9);
  j.set("fixed_latency_s", fixed_latency_s);
  j.set("jitter_fraction", jitter_fraction);
  j.set("pause_period_s", pause_period_s);
  j.set("pause_duration_s", pause_duration_s);
  return j;
}

BandwidthProfile BandwidthProfile::from_json(const json::Value& v) {
  BandwidthProfile p;
  if (const json::Value* g = v.find("gb_per_s")) {
    p.bytes_per_s = g->as_real() * 1e9;
  }
  p.bytes_per_s      = v.get_real("bytes_per_s", p.bytes_per_s);
  p.fixed_latency_s  = v.get_real("fixed_latency_s", p.fixed_latency_s);
  p.jitter_fraction  = v.get_real("jitter_fraction", p.jitter_fraction);
  p.pause_period_s   = v.get_real("pause_period_s", p.pause_period_s);
  p.pause_duration_s = v.get_real("pause_duration_s", p.pause_duration_s);
  p.validate();
  return p;
}

BandwidthModel::BandwidthModel(BandwidthProfile profile, u64 seed)
    : profile_(profile), rng_(seed) {
  profile_.validate();
  next_pause_at_s_ = profile_.pause_period_s;
}

double BandwidthModel::transfer_time_s(Bytes n, double now_s) {
  double t = profile_.fixed_latency_s +
             static_cast<double>(n) / profile_.bytes_per_s;

  if (profile_.jitter_fraction > 0.0) {
    std::uniform_real_distribution<double> dist(-profile_.jitter_fraction,
                                                profile_.jitter_fraction);
    t *= (1.0 + dist(rng_));
  }

  // 一時停止。転送区間が停止周期をまたいだら停止時間を足す。
  if (profile_.pause_period_s > 0.0 && profile_.pause_duration_s > 0.0) {
    while (next_pause_at_s_ <= now_s + t) {
      t += profile_.pause_duration_s;
      next_pause_at_s_ += profile_.pause_period_s;
      ++pause_hits_;
    }
  }

  if (t < 0.0) t = 0.0;  // jitter_fraction < 1 なので通常起きない
  return t;
}

}  // namespace alsr
