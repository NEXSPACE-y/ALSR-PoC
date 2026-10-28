#include "alsr/real_supply.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "alsr/checksum.hpp"

namespace alsr {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

RealSupplyPipeline::RealSupplyPipeline(
    std::string model_path, HostStoreMode store_mode,
    std::size_t ring_depth, Bytes slot_capacity,
    std::size_t prefetch_depth, RealSupplyEventCallback event_callback)
    : store_(std::move(model_path), store_mode),
      ring_(ring_depth, slot_capacity),
      scheduler_(make_fixed_scheduler(
          FixedSchedulerConfig{slot_capacity, prefetch_depth})),
      prefetch_depth_(prefetch_depth),
      event_callback_(std::move(event_callback)),
      worker_(&RealSupplyPipeline::worker_loop, this) {
  if (prefetch_depth == 0 || prefetch_depth > ring_depth) {
    fail(ErrorKind::Invalid,
         "real supply prefetch depth must be in [1, ring_depth]");
  }
}

RealSupplyPipeline::~RealSupplyPipeline() {
  {
    std::lock_guard lock(mutex_);
    shutdown_ = true;
    phase_active_ = false;
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void RealSupplyPipeline::start_phase(
    std::string phase, std::vector<RealSupplyChunk> plan) {
  const std::string started_phase = phase;
  {
    std::lock_guard lock(mutex_);
    if (phase_active_ || worker_busy_ || !ready_.empty() || !active_.empty() ||
        ring_.in_flight() != 0) {
      fail(ErrorKind::State,
           "real supply phase started before the previous phase retired");
    }
    if (!error_.empty()) {
      fail(ErrorKind::State, "real supply pipeline is failed: " + error_);
    }
    for (const auto& chunk : plan) {
      if (chunk.bytes == 0 || chunk.bytes > ring_.slot_capacity()) {
        fail(ErrorKind::Invalid,
             "real supply chunk does not fit the finite ring slot");
      }
      if (chunk.source_offset > store_.file_size() ||
          chunk.bytes > store_.file_size() - chunk.source_offset) {
        fail(ErrorKind::Invalid, "real supply chunk is outside the model file");
      }
      if (chunk.sha256.empty()) {
        fail(ErrorKind::Invalid, "real supply chunk lacks SHA-256");
      }
    }
    phase_ = std::move(phase);
    plan_ = std::move(plan);
    next_plan_index_ = 0;
    phase_active_ = true;
    ++stats_.phases_started;
  }
  emit(RealSupplyEvent{.event = "PHASE_SUPPLY_START",
                       .phase = started_phase,
                       .prefetch_depth = prefetch_depth_});
  cv_.notify_all();
}

PreparedRange RealSupplyPipeline::acquire(
    const std::string& tensor_key, Bytes offset_in_tensor,
    Bytes remaining_bytes) {
  ReadyItem item;
  RealSupplyEvent state_event;
  {
    std::unique_lock lock(mutex_);
    const bool ready_without_wait = !ready_.empty();
    const auto wait_begin = Clock::now();
    cv_.wait(lock, [&] {
      return !error_.empty() || !ready_.empty() || !phase_active_ || shutdown_;
    });
    if (!error_.empty()) fail(ErrorKind::Io, error_);
    if (shutdown_ || !phase_active_ || ready_.empty()) {
      fail(ErrorKind::State, "real supply range requested without READY data");
    }
    const auto wait_end = Clock::now();
    if (ready_without_wait) {
      ++stats_.prefetch_hits;
    } else {
      ++stats_.consumer_waits;
      stats_.consumer_wait_seconds +=
          std::chrono::duration<double>(wait_end - wait_begin).count();
    }
    item = ready_.front();
    if (item.chunk.tensor_key != tensor_key ||
        item.chunk.offset_in_tensor != offset_in_tensor ||
        item.chunk.bytes > remaining_bytes) {
      set_error_locked("real supply request differs from scheduled READY chunk");
      fail(ErrorKind::Consistency, error_);
    }
    ready_.pop_front();
    ring_.transition(item.slot, SlotState::Ready, SlotState::Copying);
    active_.emplace(item.slot.slot_id, item);
    ++stats_.chunks_acquired;
    stats_.bytes_supplied += item.chunk.bytes;
    state_event = transition_event(item, "READY", "COPYING", ready_.size());
  }
  emit(state_event);
  cv_.notify_all();

  return PreparedRange{.slot = item.slot,
                       .data = ring_.host_buffer(item.slot),
                       .tensor_key = item.chunk.tensor_key,
                       .chunk_id = item.chunk.chunk_id,
                       .source_offset = item.chunk.source_offset,
                       .offset_in_tensor = item.chunk.offset_in_tensor,
                       .bytes = item.chunk.bytes};
}

void RealSupplyPipeline::retire(const SlotRef& slot) {
  std::vector<RealSupplyEvent> events;
  {
    std::lock_guard lock(mutex_);
    const auto active = active_.find(slot.slot_id);
    if (active == active_.end() || active->second.slot != slot) {
      fail(ErrorKind::State, "real supply retired an unknown slot lease");
    }
    const ReadyItem item = active->second;

    ring_.transition(slot, SlotState::Copying, SlotState::Resident);
    events.push_back(transition_event(item, "COPYING", "RESIDENT", ready_.size()));
    ring_.transition(slot, SlotState::Resident, SlotState::InUse);
    events.push_back(transition_event(item, "RESIDENT", "IN_USE", ready_.size()));
    ring_.transition(slot, SlotState::InUse, SlotState::Retired);
    events.push_back(transition_event(item, "IN_USE", "RETIRED", ready_.size()));
    ring_.transition(slot, SlotState::Retired, SlotState::Empty);
    events.push_back(transition_event(item, "RETIRED", "EMPTY", ready_.size()));
    active_.erase(active);
    ++stats_.chunks_retired;
  }
  for (const auto& event : events) emit(event);
  cv_.notify_all();
}

void RealSupplyPipeline::finish_phase() {
  std::string completed_phase;
  {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] {
      return !error_.empty() ||
             (phase_active_ && next_plan_index_ == plan_.size() &&
              !worker_busy_ && ready_.empty() && active_.empty() &&
              ring_.in_flight() == 0);
    });
    if (!error_.empty()) fail(ErrorKind::Io, error_);
    if (!phase_active_) fail(ErrorKind::State, "real supply phase is not active");
    completed_phase = phase_;
    phase_active_ = false;
    plan_.clear();
    ++stats_.phases_completed;
  }
  emit(RealSupplyEvent{.event = "PHASE_SUPPLY_COMPLETE",
                       .phase = completed_phase,
                       .prefetch_depth = prefetch_depth_});
  cv_.notify_all();
}

void RealSupplyPipeline::cancel() {
  {
    std::lock_guard lock(mutex_);
    phase_active_ = false;
    ready_.clear();
    active_.clear();
    plan_.clear();
    next_plan_index_ = 0;
    ring_.cancel_all();
  }
  cv_.notify_all();
}

bool RealSupplyPipeline::failed() const {
  std::lock_guard lock(mutex_);
  return !error_.empty();
}

std::string RealSupplyPipeline::error() const {
  std::lock_guard lock(mutex_);
  return error_;
}

void RealSupplyPipeline::worker_loop() {
  for (;;) {
    ReadyItem item;
    SchedulerDecision decision;
    RealSupplyEvent preparing_event;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] {
        return shutdown_ || !error_.empty() ||
               (phase_active_ && next_plan_index_ < plan_.size() &&
                ring_.has_free() &&
                ring_.in_flight() < prefetch_depth_);
      });
      if (shutdown_) return;
      if (!error_.empty()) continue;

      SchedulerInput input;
      input.chunk_index = static_cast<u64>(next_plan_index_);
      input.pending_chunk_indices.push_back(next_plan_index_);
      input.queue_depth = ring_.in_flight();
      input.ring_depth = ring_.depth();
      input.free_slots = ring_.free_count();
      input.current_chunk_size = ring_.slot_capacity();
      input.current_prefetch_depth = prefetch_depth_;
      decision = scheduler_->decide(input);
      ++stats_.scheduler_decisions;
      if (decision.next_chunk_index != next_plan_index_ ||
          decision.prefetch_depth != prefetch_depth_) {
        set_error_locked("fixed real supply scheduler changed the approved plan");
        cv_.notify_all();
        continue;
      }

      item.chunk = plan_[next_plan_index_++];
      item.slot = ring_.acquire_free();
      item.phase = phase_;
      BufferSlot& slot = ring_.slot(item.slot);
      slot.chunk_id = item.chunk.chunk_id;
      slot.source_offset = item.chunk.source_offset;
      slot.payload_bytes = item.chunk.bytes;
      slot.checksum_expected_known = true;
      slot.checksum_expected = item.chunk.crc32c;
      worker_busy_ = true;
      preparing_event = transition_event(
          item, "EMPTY", "PREPARING", ready_.size());
    }

    emit(RealSupplyEvent{.event = "SCHEDULER_DECISION",
                         .phase = item.phase,
                         .tensor_key = item.chunk.tensor_key,
                         .chunk_id = item.chunk.chunk_id,
                         .slot_id = item.slot.slot_id,
                         .generation = item.slot.generation,
                         .source_offset = item.chunk.source_offset,
                         .offset_in_tensor = item.chunk.offset_in_tensor,
                         .bytes = item.chunk.bytes,
                         .to_state = decision.action,
                         .prefetch_depth = decision.prefetch_depth});
    emit(preparing_event);

    std::string failure;
    double prepare_ms = 0.0;
    try {
      const auto begin = Clock::now();
      u8* data = ring_.host_buffer(item.slot);
      store_.read_into(item.chunk.source_offset, item.chunk.bytes, data);
      const auto end = Clock::now();
      prepare_ms = elapsed_ms(begin, end);
      const bool crc_matches =
          crc32c(data, static_cast<std::size_t>(item.chunk.bytes)) ==
          item.chunk.crc32c;
      bool sha_matches = true;
      if (!sha_verified_chunks_.contains(item.chunk.chunk_id)) {
        sha_matches =
            sha256_hex(data, static_cast<std::size_t>(item.chunk.bytes)) ==
            item.chunk.sha256;
        if (sha_matches) sha_verified_chunks_.insert(item.chunk.chunk_id);
      }
      if (!crc_matches || !sha_matches) {
        failure = "real supply prepared Chunk checksum mismatch";
      }
    } catch (const std::exception& error) {
      failure = error.what();
    }

    RealSupplyEvent ready_event;
    bool preparation_failed = false;
    {
      std::lock_guard lock(mutex_);
      worker_busy_ = false;
      if (!failure.empty()) {
        preparation_failed = true;
        try {
          ring_.transition(item.slot, SlotState::Preparing, SlotState::Error);
        } catch (...) {
        }
        set_error_locked(std::move(failure));
      } else {
        BufferSlot& slot = ring_.slot(item.slot);
        slot.checksum = item.chunk.crc32c;
        ring_.transition(item.slot, SlotState::Preparing, SlotState::Ready);
        ready_.push_back(item);
        ++stats_.chunks_prepared;
        stats_.bytes_prepared += item.chunk.bytes;
        stats_.prepare_seconds += prepare_ms / 1000.0;
        stats_.peak_ready_queue = std::max(stats_.peak_ready_queue, ready_.size());
        ready_event = transition_event(
            item, "PREPARING", "READY", ready_.size());
        ready_event.prepare_ms = prepare_ms;
      }
    }
    if (preparation_failed) {
      emit(RealSupplyEvent{.event = "SUPPLY_ERROR",
                           .phase = item.phase,
                           .tensor_key = item.chunk.tensor_key,
                           .chunk_id = item.chunk.chunk_id,
                           .slot_id = item.slot.slot_id,
                           .generation = item.slot.generation,
                           .to_state = "ERROR"});
    } else {
      emit(ready_event);
    }
    cv_.notify_all();
  }
}

void RealSupplyPipeline::set_error_locked(std::string message) {
  if (error_.empty()) error_ = std::move(message);
}

void RealSupplyPipeline::emit(const RealSupplyEvent& event) const {
  if (event_callback_) event_callback_(event);
}

RealSupplyEvent RealSupplyPipeline::transition_event(
    const ReadyItem& item, std::string from, std::string to,
    std::size_t queue_depth) {
  return RealSupplyEvent{.event = "SLOT_STATE",
                         .phase = item.phase,
                         .tensor_key = item.chunk.tensor_key,
                         .chunk_id = item.chunk.chunk_id,
                         .slot_id = item.slot.slot_id,
                         .generation = item.slot.generation,
                         .source_offset = item.chunk.source_offset,
                         .offset_in_tensor = item.chunk.offset_in_tensor,
                         .bytes = item.chunk.bytes,
                         .from_state = std::move(from),
                         .to_state = std::move(to),
                         .queue_depth = queue_depth,
                         .prefetch_depth = prefetch_depth_,
                         .transition_sequence = ++next_transition_sequence_};
}

}  // namespace alsr
