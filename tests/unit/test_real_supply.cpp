#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/host_store.hpp"
#include "alsr/real_supply.hpp"
#include "test_helpers.hpp"

using namespace alsr;
using namespace alsr::test;

namespace {

std::vector<RealSupplyChunk> make_plan(const GgufBuildResult& built,
                                       const std::string& path) {
  HostStore source(path, HostStoreMode::Read);
  std::vector<RealSupplyChunk> plan;
  constexpr Bytes kChunk = 4096;
  Bytes offset = built.data_start;
  for (u64 id = 1; id <= 4; ++id) {
    std::vector<u8> bytes(static_cast<std::size_t>(kChunk));
    source.read_into(offset, kChunk, bytes.data());
    plan.push_back(RealSupplyChunk{
        .chunk_id = id,
        .tensor_key = "weight\n0",
        .source_offset = offset,
        .offset_in_tensor = (id - 1) * kChunk,
        .bytes = kChunk,
        .crc32c = crc32c(bytes.data(), bytes.size()),
        .sha256 = sha256_hex(bytes.data(), bytes.size())});
    offset += kChunk;
  }
  return plan;
}

}  // namespace

TEST_CASE("RealSupply: explicit range I/Oから有限Ringを通してbyteを供給する",
          "[unit][real-supply]") {
  TempDir temp("real-supply");
  const auto built = write_test_gguf(
      temp.file("model.gguf"), {{"weight", 0, {8192}}});
  const auto plan = make_plan(built, built.path);
  HostStore expected(built.path, HostStoreMode::Mmap);

  std::mutex event_mutex;
  std::vector<RealSupplyEvent> events;
  RealSupplyPipeline pipeline(
      built.path, HostStoreMode::Read, 2, 4096, 2,
      [&](const RealSupplyEvent& event) {
        std::lock_guard lock(event_mutex);
        events.push_back(event);
      });
  pipeline.start_phase("decode", plan);

  for (const auto& chunk : plan) {
    const PreparedRange range = pipeline.acquire(
        chunk.tensor_key, chunk.offset_in_tensor, chunk.bytes);
    std::vector<u8> reference(static_cast<std::size_t>(chunk.bytes));
    expected.read_into(chunk.source_offset, chunk.bytes, reference.data());
    REQUIRE(std::equal(reference.begin(), reference.end(), range.data));
    pipeline.retire(range.slot);
  }
  pipeline.finish_phase();

  REQUIRE(pipeline.store_mode() == HostStoreMode::Read);
  REQUIRE(pipeline.store_bytes_read() == 4 * 4096);
  REQUIRE(pipeline.stats().phases_completed == 1);
  REQUIRE(pipeline.stats().scheduler_decisions == 4);
  REQUIRE(pipeline.stats().chunks_prepared == 4);
  REQUIRE(pipeline.stats().chunks_acquired == 4);
  REQUIRE(pipeline.stats().chunks_retired == 4);
  REQUIRE(pipeline.stats().prefetch_hits + pipeline.stats().consumer_waits == 4);
  REQUIRE(pipeline.stats().bytes_prepared == 4 * 4096);
  REQUIRE(pipeline.ring_stats().depth == 2);
  REQUIRE(pipeline.ring_stats().peak_in_flight >= 1);
  REQUIRE(pipeline.ring_stats().illegal_transition_attempts == 0);

  std::lock_guard lock(event_mutex);
  REQUIRE(std::count_if(events.begin(), events.end(), [](const auto& event) {
            return event.event == "SLOT_STATE" && event.to_state == "READY";
          }) == 4);
  REQUIRE(std::count_if(events.begin(), events.end(), [](const auto& event) {
            return event.event == "SLOT_STATE" && event.to_state == "EMPTY";
          }) == 4);
  std::vector<u64> transition_sequences;
  for (const auto& event : events) {
    if (event.event == "SLOT_STATE") {
      REQUIRE(event.transition_sequence > 0);
      transition_sequences.push_back(event.transition_sequence);
    }
  }
  std::ranges::sort(transition_sequences);
  REQUIRE(transition_sequences.size() == 4 * 7);
  for (std::size_t i = 0; i < transition_sequences.size(); ++i) {
    REQUIRE(transition_sequences[i] == i + 1);
  }
}

TEST_CASE("RealSupply: Runtime要求順がManifest計画と違えば停止する",
          "[unit][real-supply]") {
  TempDir temp("real-supply-order");
  const auto built = write_test_gguf(
      temp.file("model.gguf"), {{"weight", 0, {8192}}});
  const auto plan = make_plan(built, built.path);
  RealSupplyPipeline pipeline(
      built.path, HostStoreMode::Read, 2, 4096, 2);
  pipeline.start_phase("decode", plan);
  REQUIRE_THROWS_AS(pipeline.acquire("different\n0", 0, 4096), Error);
  pipeline.cancel();
}
