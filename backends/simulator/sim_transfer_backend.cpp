// simulator backend — M5 上でスケジューラの論理を検証するための仮想時間実装。
//
// これは CUDA 性能の予測ではない (設計書 §3.1 / FR-07)。
// capabilities().measured_on_real_hardware は必ず false を返す。
#include "alsr/transfer_backend.hpp"

#include "alsr/checksum.hpp"

namespace alsr {
namespace {

class SimulatorBackend final : public TransferBackend {
public:
  explicit SimulatorBackend(SimulatorBackendConfig cfg)
      : cfg_(cfg),
        h2d_(cfg.h2d, cfg.seed ^ 0x483d'8b93'2f1a'0001ull),
        prepare_(cfg.prepare, cfg.seed ^ 0x483d'8b93'2f1a'0002ull),
        consume_rng_(cfg.seed ^ 0x483d'8b93'2f1a'0003ull) {
    if (!(cfg_.consume_bytes_per_s > 0.0)) {
      fail(ErrorKind::Invalid, "consume_bytes_per_s must be > 0");
    }
    if (cfg_.consume_jitter_fraction < 0.0 ||
        cfg_.consume_jitter_fraction >= 1.0) {
      fail(ErrorKind::Invalid, "consume_jitter_fraction must be in [0, 1)");
    }
  }

  [[nodiscard]] std::string name() const override { return "simulator"; }

  [[nodiscard]] TransferCapabilities capabilities() const override {
    TransferCapabilities c;
    c.device_overlap = true;   // 仮想的に重ねられるものとして扱う
    c.async_engine_count = 1;
    c.pinned_memory_supported = false;
    c.measured_on_real_hardware = false;  // ← ここが false である限り性能主張はしない
    c.pinned_limit_bytes = cfg_.pinned_limit_bytes;
    c.device_memory_bytes = cfg_.device_memory_bytes;
    return c;
  }

  [[nodiscard]] double h2d_time_s(Bytes n, double now_s) override {
    return h2d_.transfer_time_s(n, now_s);
  }

  [[nodiscard]] double prepare_time_s(Bytes n, double now_s) override {
    return prepare_.transfer_time_s(n, now_s);
  }

  [[nodiscard]] double compute_time_s(Bytes n, double) override {
    double t = static_cast<double>(n) / cfg_.consume_bytes_per_s;
    if (cfg_.consume_jitter_fraction > 0.0) {
      std::uniform_real_distribution<double> dist(-cfg_.consume_jitter_fraction,
                                                   cfg_.consume_jitter_fraction);
      t *= (1.0 + dist(consume_rng_));
    }
    return t < 0.0 ? 0.0 : t;
  }

  u32 consume(const u8* host_data, Bytes n) override {
    if (!cfg_.verify_checksums) return 0;
    // 実データに触る。ここを飛ばすと「転送できたつもり」の検証にしかならない。
    return crc32c(host_data, static_cast<std::size_t>(n));
  }

  [[nodiscard]] json::Value config_json() const override {
    json::Value j = json::Value::make_object();
    j.set("type", "simulator");
    j.set("h2d", cfg_.h2d.to_json());
    j.set("prepare", cfg_.prepare.to_json());
    j.set("consume_bytes_per_s", cfg_.consume_bytes_per_s);
    j.set("consume_gb_per_s", cfg_.consume_bytes_per_s / 1e9);
    j.set("consume_jitter_fraction", cfg_.consume_jitter_fraction);
    j.set("device_memory_bytes", cfg_.device_memory_bytes);
    j.set("pinned_limit_bytes", cfg_.pinned_limit_bytes);
    j.set("verify_checksums", cfg_.verify_checksums);
    j.set("seed", cfg_.seed);
    j.set("measured_on_real_hardware", false);
    j.set("caveat",
          "仮想時間。M5 には独立 VRAM への PCIe H2D 経路が無いため、"
          "この結果で CUDA の成立を主張しない (設計書 §3.1)");
    return j;
  }

private:
  SimulatorBackendConfig cfg_;
  BandwidthModel         h2d_;
  BandwidthModel         prepare_;
  std::mt19937_64        consume_rng_;
};

}  // namespace

std::unique_ptr<TransferBackend> make_simulator_backend(
    const SimulatorBackendConfig& cfg) {
  return std::make_unique<SimulatorBackend>(cfg);
}

}  // namespace alsr
