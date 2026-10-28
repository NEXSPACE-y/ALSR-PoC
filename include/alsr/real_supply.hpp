// ALSR PoC — real Model Store -> Preparation -> Ready Queue -> finite Ring.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <set>
#include <thread>
#include <vector>

#include "alsr/host_store.hpp"
#include "alsr/ring_buffer.hpp"
#include "alsr/scheduler.hpp"

namespace alsr {

struct RealSupplyChunk {
  u64 chunk_id = 0;
  std::string tensor_key;
  Bytes source_offset = 0;
  Bytes offset_in_tensor = 0;
  Bytes bytes = 0;
  u32 crc32c = 0;
  std::string sha256;
};

struct RealSupplyEvent {
  std::string event;
  std::string phase;
  std::string tensor_key;
  u64 chunk_id = 0;
  std::size_t slot_id = 0;
  u64 generation = 0;
  Bytes source_offset = 0;
  Bytes offset_in_tensor = 0;
  Bytes bytes = 0;
  std::string from_state;
  std::string to_state;
  std::size_t queue_depth = 0;
  std::size_t prefetch_depth = 0;
  double prepare_ms = 0.0;
  u64 transition_sequence = 0;
};

using RealSupplyEventCallback = std::function<void(const RealSupplyEvent&)>;

struct PreparedRange {
  SlotRef slot;
  const u8* data = nullptr;
  std::string tensor_key;
  u64 chunk_id = 0;
  Bytes source_offset = 0;
  Bytes offset_in_tensor = 0;
  Bytes bytes = 0;
};

struct RealSupplyStats {
  u64 phases_started = 0;
  u64 phases_completed = 0;
  u64 scheduler_decisions = 0;
  u64 chunks_prepared = 0;
  u64 chunks_acquired = 0;
  u64 chunks_retired = 0;
  u64 prefetch_hits = 0;
  u64 consumer_waits = 0;
  Bytes bytes_prepared = 0;
  Bytes bytes_supplied = 0;
  std::size_t peak_ready_queue = 0;
  double prepare_seconds = 0.0;
  double consumer_wait_seconds = 0.0;
};

class RealSupplyPipeline {
 public:
  RealSupplyPipeline(std::string model_path, HostStoreMode store_mode,
                     std::size_t ring_depth, Bytes slot_capacity,
                     std::size_t prefetch_depth,
                     RealSupplyEventCallback event_callback = {});
  ~RealSupplyPipeline();

  RealSupplyPipeline(const RealSupplyPipeline&) = delete;
  RealSupplyPipeline& operator=(const RealSupplyPipeline&) = delete;

  void start_phase(std::string phase, std::vector<RealSupplyChunk> plan);
  PreparedRange acquire(const std::string& tensor_key,
                        Bytes offset_in_tensor, Bytes remaining_bytes);
  void retire(const SlotRef& slot);
  void finish_phase();
  void cancel();

  [[nodiscard]] const RealSupplyStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const RingBufferStats& ring_stats() const noexcept {
    return ring_.stats();
  }
  [[nodiscard]] Bytes store_bytes_read() const noexcept {
    return store_.bytes_read();
  }
  [[nodiscard]] HostStoreMode store_mode() const noexcept { return store_.mode(); }
  [[nodiscard]] std::size_t prefetch_depth() const noexcept {
    return prefetch_depth_;
  }
  [[nodiscard]] bool failed() const;
  [[nodiscard]] std::string error() const;

 private:
  struct ReadyItem {
    SlotRef slot;
    RealSupplyChunk chunk;
    std::string phase;
  };

  void worker_loop();
  void set_error_locked(std::string message);
  void emit(const RealSupplyEvent& event) const;
  RealSupplyEvent transition_event(const ReadyItem& item,
                                   std::string from, std::string to,
                                   std::size_t queue_depth);

  HostStore store_;
  RingBuffer ring_;
  std::unique_ptr<Scheduler> scheduler_;
  std::size_t prefetch_depth_ = 0;
  RealSupplyEventCallback event_callback_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  bool shutdown_ = false;
  bool phase_active_ = false;
  bool worker_busy_ = false;
  std::string phase_;
  std::vector<RealSupplyChunk> plan_;
  std::size_t next_plan_index_ = 0;
  std::deque<ReadyItem> ready_;
  std::map<std::size_t, ReadyItem> active_;
  std::string error_;
  std::set<u64> sha_verified_chunks_;
  RealSupplyStats stats_;
  u64 next_transition_sequence_ = 0;
};

}  // namespace alsr
