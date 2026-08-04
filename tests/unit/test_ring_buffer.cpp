#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "alsr/ring_buffer.hpp"

using namespace alsr;

namespace {

/// FREE から一周させる。
void full_cycle(RingBuffer& ring, const SlotRef& r) {
  ring.transition(r, SlotState::Preparing, SlotState::Ready);
  ring.transition(r, SlotState::Ready, SlotState::Copying);
  ring.transition(r, SlotState::Copying, SlotState::Resident);
  ring.transition(r, SlotState::Resident, SlotState::InUse);
  ring.transition(r, SlotState::InUse, SlotState::Retired);
  ring.transition(r, SlotState::Retired, SlotState::Empty);
}

}  // namespace

TEST_CASE("RingBuffer: 正しい状態遷移を一周できる", "[unit][ring]") {
  RingBuffer ring(2, 1024);
  REQUIRE(ring.depth() == 2);
  REQUIRE(ring.free_count() == 2);

  const SlotRef r = ring.acquire_free();
  REQUIRE(ring.slot(r).state == SlotState::Preparing);
  REQUIRE(ring.in_flight() == 1);
  REQUIRE(ring.free_count() == 1);

  full_cycle(ring, r);
  REQUIRE(ring.free_count() == 2);
  REQUIRE(ring.in_flight() == 0);
}

TEST_CASE("RingBuffer: 遷移表にない遷移は State エラー", "[unit][ring]") {
  RingBuffer ring(2, 1024);
  const SlotRef r = ring.acquire_free();  // FILLING

  // FILLING から直接 RESIDENT へは行けない
  try {
    ring.transition(r, SlotState::Preparing, SlotState::Resident);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::State);
  }
  REQUIRE(ring.stats().illegal_transition_attempts == 1);

  // from が現在状態と違う場合も拒否
  REQUIRE_THROWS_AS(ring.transition(r, SlotState::Ready, SlotState::Copying), Error);

  // 逆走も拒否
  REQUIRE_THROWS_AS(ring.transition(r, SlotState::Preparing, SlotState::Empty), Error);
}

TEST_CASE("RingBuffer: 世代番号で ABA を検出する", "[unit][ring]") {
  RingBuffer ring(1, 1024);
  const SlotRef old = ring.acquire_free();
  const u64 gen0 = old.generation;
  full_cycle(ring, old);

  // 同じ slot_id が再利用されるが世代が進む
  const SlotRef fresh = ring.acquire_free();
  REQUIRE(fresh.slot_id == old.slot_id);
  REQUIRE(fresh.generation == gen0 + 1);

  // 古いハンドルでは触れない
  try {
    ring.slot(old);
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::State);
  }
  REQUIRE_THROWS_AS(ring.transition(old, SlotState::Preparing, SlotState::Ready), Error);
  REQUIRE_THROWS_AS(ring.host_buffer(old), Error);
}

TEST_CASE("RingBuffer: producer は consumer を上書きできない", "[unit][ring]") {
  RingBuffer ring(2, 1024);
  const SlotRef a = ring.acquire_free();
  const SlotRef b = ring.acquire_free();
  REQUIRE(a.slot_id != b.slot_id);
  REQUIRE_FALSE(ring.has_free());

  // FREE が無い状態での取得は失敗する（消費中のスロットを奪わない）
  try {
    ring.acquire_free();
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::State);
  }

  // バッファは別領域
  ring.host_buffer(a)[0] = 0xAA;
  ring.host_buffer(b)[0] = 0xBB;
  REQUIRE(ring.host_buffer(a)[0] == 0xAA);
  REQUIRE(ring.host_buffer(b)[0] == 0xBB);
}

TEST_CASE("RingBuffer: wrap-around を多数回繰り返しても壊れない", "[unit][ring]") {
  constexpr std::size_t kDepth = 3;
  constexpr int kRounds = 1000;
  RingBuffer ring(kDepth, 256);

  std::vector<u64> last_gen(kDepth, 0);
  for (int i = 0; i < kRounds; ++i) {
    const SlotRef r = ring.acquire_free();
    REQUIRE(r.generation > last_gen[r.slot_id]);
    last_gen[r.slot_id] = r.generation;
    ring.slot(r).chunk_id = static_cast<u64>(i);
    full_cycle(ring, r);
  }
  REQUIRE(ring.free_count() == kDepth);
  REQUIRE(ring.stats().acquisitions == kRounds);
  REQUIRE(ring.stats().wrap_arounds == kRounds - kDepth);
  REQUIRE(ring.stats().illegal_transition_attempts == 0);
}

TEST_CASE("RingBuffer: cancel で全スロットが安全に解放される", "[unit][ring]") {
  RingBuffer ring(4, 512);
  const SlotRef a = ring.acquire_free();
  const SlotRef b = ring.acquire_free();
  ring.transition(a, SlotState::Preparing, SlotState::Ready);
  ring.transition(a, SlotState::Ready, SlotState::Copying);
  ring.transition(b, SlotState::Preparing, SlotState::Ready);
  REQUIRE(ring.in_flight() == 2);

  ring.cancel_all();

  REQUIRE(ring.free_count() == 4);
  REQUIRE(ring.in_flight() == 0);
  const auto hist = ring.state_histogram();
  REQUIRE(hist[static_cast<std::size_t>(SlotState::Empty)] == 4);

  // cancel 前のハンドルはすべて無効
  REQUIRE_THROWS_AS(ring.slot(a), Error);
  REQUIRE_THROWS_AS(ring.slot(b), Error);

  // cancel 後も正常に使える
  const SlotRef c = ring.acquire_free();
  REQUIRE_NOTHROW(full_cycle(ring, c));
  REQUIRE(ring.stats().cancellations == 1);
}

TEST_CASE("RingBuffer: 不正な構築を拒否する", "[unit][ring]") {
  REQUIRE_THROWS_AS(RingBuffer(0, 1024), Error);
  REQUIRE_THROWS_AS(RingBuffer(2, 0), Error);
}

TEST_CASE("RingBuffer: 範囲外の slot_id を拒否する", "[unit][ring]") {
  RingBuffer ring(2, 128);
  const SlotRef bogus{99, 1};
  REQUIRE_THROWS_AS(ring.slot(bogus), Error);
}

TEST_CASE("RingBuffer: peak_in_flight を追跡する", "[unit][ring]") {
  RingBuffer ring(4, 128);
  const SlotRef a = ring.acquire_free();
  const SlotRef b = ring.acquire_free();
  const SlotRef c = ring.acquire_free();
  REQUIRE(ring.stats().peak_in_flight == 3);
  full_cycle(ring, a);
  full_cycle(ring, b);
  full_cycle(ring, c);
  REQUIRE(ring.stats().peak_in_flight == 3);
}
