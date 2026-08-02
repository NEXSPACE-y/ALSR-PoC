#include <catch2/catch_test_macros.hpp>

#include "alsr/ggml_types.hpp"

using namespace alsr;

TEST_CASE("ggml: 既知型のサイズ計算", "[unit][ggml]") {
  // F32 (block 1, 4 byte)
  REQUIRE(ggml_tensor_bytes(0, {3840}) == 3840u * 4u);
  REQUIRE(ggml_tensor_bytes(0, {3840, 2}) == 3840u * 4u * 2u);
  // Q4_K (block 256, 144 byte)
  REQUIRE(ggml_tensor_bytes(12, {3840, 2048}) == (3840u / 256u) * 144u * 2048u);
  // Q6_K (block 256, 210 byte)
  REQUIRE(ggml_tensor_bytes(14, {3840, 262144}) ==
          (3840ull / 256ull) * 210ull * 262144ull);
}

TEST_CASE("ggml: gemma4:12b の実測値と一致する", "[unit][ggml]") {
  // 実機 blob (sha256-1278394b...) から読んだ値。テーブルが壊れたら落ちる。
  REQUIRE(ggml_tensor_bytes(14, {3840, 262144}) == 825753600ull);  // token_embd
  REQUIRE(ggml_tensor_bytes(12, {3840, 2048}) == 4423680ull);      // attn_k
  REQUIRE(ggml_tensor_bytes(0, {3840}) == 15360ull);               // norm
  REQUIRE(ggml_tensor_bytes(0, {256}) == 1024ull);                 // rope_freqs
  REQUIRE(ggml_tensor_bytes(0, {1}) == 4ull);  // layer_output_scale
}

TEST_CASE("ggml: 未知型は推測せず Unsupported", "[unit][ggml]") {
  REQUIRE(ggml_type_info(9999) == nullptr);
  REQUIRE(ggml_type_name(9999) == "unknown(9999)");
  try {
    (void)ggml_tensor_bytes(9999, {32});
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Unsupported);
  }
  // 削除済み ID (Q4_2 = 4) もテーブルに載せない
  REQUIRE(ggml_type_info(4) == nullptr);
  REQUIRE(ggml_type_info(31) == nullptr);
}

TEST_CASE("ggml: 中間量子化型は格納型として拒否する", "[unit][ggml]") {
  // Q8_1(9) / Q8_K(15) は GGUF 格納型としては現れない想定
  REQUIRE(ggml_type_info(9) != nullptr);
  REQUIRE_FALSE(ggml_type_info(9)->storable);
  try {
    (void)ggml_tensor_bytes(9, {32});
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Unsupported);
  }
}

TEST_CASE("ggml: ブロック境界に揃わない ne[0] は Format エラー", "[unit][ggml]") {
  try {
    (void)ggml_tensor_bytes(12, {100});  // Q4_K は 256 の倍数が必要
    FAIL("should have thrown");
  } catch (const Error& e) {
    REQUIRE(e.kind() == ErrorKind::Format);
  }
  REQUIRE_NOTHROW(ggml_tensor_bytes(12, {256}));
  REQUIRE_NOTHROW(ggml_tensor_bytes(2, {32}));    // Q4_0 は 32
  REQUIRE_THROWS_AS(ggml_tensor_bytes(2, {31}), Error);
}

TEST_CASE("ggml: rank と次元の異常", "[unit][ggml]") {
  REQUIRE_THROWS_AS(ggml_tensor_bytes(0, {}), Error);
  REQUIRE_THROWS_AS(ggml_tensor_bytes(0, {1, 1, 1, 1, 1}), Error);
  // 桁溢れ
  REQUIRE_THROWS_AS(
      ggml_tensor_bytes(0, {1ull << 40, 1ull << 40, 1ull << 40}), Error);
}

TEST_CASE("ggml: block_bytes は分割粒度として使える", "[unit][ggml]") {
  REQUIRE(ggml_block_bytes(0) == 4);     // F32
  REQUIRE(ggml_block_bytes(12) == 144);  // Q4_K
  REQUIRE(ggml_block_bytes(14) == 210);  // Q6_K
  REQUIRE_THROWS_AS(ggml_block_bytes(9999), Error);
}
