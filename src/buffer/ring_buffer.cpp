#include "alsr/ring_buffer.hpp"

#include <sstream>

namespace alsr {
namespace {

/// 許可された遷移表。ここに無い組み合わせはすべて違反。
bool transition_allowed(SlotState from, SlotState to) noexcept {
  switch (from) {
    case SlotState::Free:      return to == SlotState::Filling;
    case SlotState::Filling:   return to == SlotState::Ready;
    case SlotState::Ready:     return to == SlotState::Copying;
    case SlotState::Copying:   return to == SlotState::Resident;
    case SlotState::Resident:  return to == SlotState::Consuming;
    case SlotState::Consuming: return to == SlotState::Free;
  }
  return false;
}

}  // namespace

std::string_view to_string(SlotState s) noexcept {
  switch (s) {
    case SlotState::Free:      return "FREE";
    case SlotState::Filling:   return "FILLING";
    case SlotState::Ready:     return "READY";
    case SlotState::Copying:   return "COPYING";
    case SlotState::Resident:  return "RESIDENT";
    case SlotState::Consuming: return "CONSUMING";
  }
  return "?";
}

RingBuffer::RingBuffer(std::size_t depth, Bytes slot_capacity)
    : slot_capacity_(slot_capacity) {
  if (depth == 0) fail(ErrorKind::Invalid, "ring buffer depth must be >= 1");
  if (slot_capacity == 0) fail(ErrorKind::Invalid, "slot capacity must be >= 1");

  slots_.resize(depth);
  buffers_.resize(depth);
  for (std::size_t i = 0; i < depth; ++i) {
    slots_[i].slot_id = i;
    slots_[i].generation = 1;  // 0 は「無効」を表すため使わない
    slots_[i].state = SlotState::Free;
    buffers_[i].assign(static_cast<std::size_t>(slot_capacity), 0);
    free_ids_.push_back(i);
  }
  stats_.depth = depth;
  stats_.slot_capacity = slot_capacity;
}

void RingBuffer::check_ref(const SlotRef& ref) const {
  if (ref.slot_id >= slots_.size()) {
    std::ostringstream os;
    os << "slot id " << ref.slot_id << " out of range (depth=" << slots_.size() << ")";
    fail(ErrorKind::State, os.str());
  }
  const BufferSlot& s = slots_[ref.slot_id];
  if (s.generation != ref.generation) {
    std::ostringstream os;
    os << "stale slot handle (ABA): slot " << ref.slot_id << " generation "
       << ref.generation << " but current is " << s.generation;
    fail(ErrorKind::State, os.str());
  }
}

SlotRef RingBuffer::acquire_free() {
  if (free_ids_.empty()) {
    fail(ErrorKind::State,
         "no FREE slot available. has_free() を確認してから呼ぶこと");
  }
  const std::size_t id = free_ids_.front();
  free_ids_.pop_front();

  BufferSlot& s = slots_[id];
  if (s.state != SlotState::Free) {
    std::ostringstream os;
    os << "slot " << id << " was in free list but state is " << to_string(s.state);
    fail(ErrorKind::State, os.str());
  }
  s.state = SlotState::Filling;

  ++stats_.acquisitions;
  if (stats_.acquisitions > slots_.size()) ++stats_.wrap_arounds;
  const std::size_t inflight = in_flight();
  if (inflight > stats_.peak_in_flight) stats_.peak_in_flight = inflight;

  return s.ref();
}

void RingBuffer::transition(const SlotRef& ref, SlotState from, SlotState to) {
  check_ref(ref);
  BufferSlot& s = slots_[ref.slot_id];

  if (s.state != from || !transition_allowed(from, to)) {
    ++stats_.illegal_transition_attempts;
    std::ostringstream os;
    os << "illegal transition on slot " << ref.slot_id << ": requested "
       << to_string(from) << " -> " << to_string(to) << " but slot is "
       << to_string(s.state);
    fail(ErrorKind::State, os.str());
  }

  s.state = to;
  if (to == SlotState::Free) {
    // 世代を進めることで、解放後の古いハンドルを検出できるようにする。
    ++s.generation;
    s.released_at_s = s.compute_end_s;
    free_ids_.push_back(ref.slot_id);
  }
}

BufferSlot& RingBuffer::slot(const SlotRef& ref) {
  check_ref(ref);
  return slots_[ref.slot_id];
}

const BufferSlot& RingBuffer::slot(const SlotRef& ref) const {
  check_ref(ref);
  return slots_[ref.slot_id];
}

u8* RingBuffer::host_buffer(const SlotRef& ref) {
  check_ref(ref);
  return buffers_[ref.slot_id].data();
}

void RingBuffer::cancel_all() {
  ++stats_.cancellations;
  free_ids_.clear();
  for (BufferSlot& s : slots_) {
    ++s.generation;  // 既存ハンドルを一括無効化
    s.state = SlotState::Free;
    s.chunk_id = 0;
    s.payload_bytes = 0;
    s.checksum = 0;
    s.checksum_expected_known = false;
    free_ids_.push_back(s.slot_id);
  }
}

std::vector<std::size_t> RingBuffer::state_histogram() const {
  std::vector<std::size_t> h(6, 0);
  for (const BufferSlot& s : slots_) {
    h[static_cast<std::size_t>(s.state)] += 1;
  }
  return h;
}

}  // namespace alsr
