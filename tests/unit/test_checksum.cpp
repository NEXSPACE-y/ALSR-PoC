#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "alsr/checksum.hpp"

using namespace alsr;

TEST_CASE("CRC32C: 既知ベクタ", "[unit][checksum]") {
  const std::string s = "123456789";
  REQUIRE(crc32c(s.data(), s.size()) == 0xE3069283u);
  REQUIRE(crc32c_software(s.data(), s.size()) == 0xE3069283u);
  REQUIRE(crc32c("", 0) == 0u);
}

TEST_CASE("CRC32C: ハードウェア経路とソフトウェア経路が一致する",
          "[unit][checksum]") {
  // 長さを変えて境界（8 バイト未満の端数、非整列）を全部通す
  std::vector<u8> buf(1024);
  for (std::size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<u8>((i * 131 + 7) & 0xFF);
  }
  for (std::size_t len = 0; len <= 200; ++len) {
    for (std::size_t off = 0; off < 9; ++off) {
      INFO("len=" << len << " off=" << off);
      REQUIRE(crc32c(buf.data() + off, len) ==
              crc32c_software(buf.data() + off, len));
    }
  }
}

TEST_CASE("CRC32C: 逐次更新と一括計算が一致する", "[unit][checksum]") {
  std::vector<u8> buf(4096);
  for (std::size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<u8>((i * 17) & 0xFF);
  }
  for (const std::size_t split : {1u, 7u, 64u, 1000u, 4095u}) {
    Crc32c c;
    c.update(buf.data(), split);
    c.update(buf.data() + split, buf.size() - split);
    INFO("split=" << split);
    REQUIRE(c.value() == crc32c(buf.data(), buf.size()));
  }
}

TEST_CASE("CRC32C: 1 ビット違いを検出する", "[unit][checksum]") {
  std::vector<u8> a(256, 0x5A);
  std::vector<u8> b = a;
  b[128] ^= 0x01;
  REQUIRE(crc32c(a.data(), a.size()) != crc32c(b.data(), b.size()));
}

TEST_CASE("SHA-256: 既知ベクタ (FIPS 180-4)", "[unit][checksum]") {
  REQUIRE(sha256_hex("", 0) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  REQUIRE(sha256_hex("abc", 3) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string m =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  REQUIRE(sha256_hex(m.data(), m.size()) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("SHA-256: ブロック境界と逐次更新", "[unit][checksum]") {
  // 55/56/63/64/65 バイトはパディング分岐の境界
  for (const std::size_t n : {55u, 56u, 57u, 63u, 64u, 65u, 119u, 120u, 128u}) {
    std::string data(n, 'a');
    Sha256 h;
    for (std::size_t i = 0; i < n; ++i) h.update(data.data() + i, 1);
    INFO("n=" << n);
    REQUIRE(h.hex() == sha256_hex(data.data(), data.size()));
  }
}

TEST_CASE("SHA-256: 1,000,000 個の 'a'", "[unit][checksum]") {
  const std::string a(1000000, 'a');
  REQUIRE(sha256_hex(a.data(), a.size()) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
