#include "alsr/checksum.hpp"

#include <cstring>
#include <fstream>
#include <vector>

#if defined(ALSR_ENABLE_HW_CRC) && defined(__ARM_FEATURE_CRC32)
#  include <arm_acle.h>
#  define ALSR_HAVE_ARM_CRC32 1
#else
#  define ALSR_HAVE_ARM_CRC32 0
#endif

namespace alsr {
namespace {

// --------------------------------------------------------------------------
// CRC-32C slicing-by-8 テーブル（起動時に一度だけ生成）
// --------------------------------------------------------------------------
constexpr u32 kPoly = 0x82F63B78u;  // reflected Castagnoli

struct Crc32cTable {
  std::array<std::array<u32, 256>, 8> t{};

  Crc32cTable() {
    for (u32 i = 0; i < 256; ++i) {
      u32 c = i;
      for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (kPoly & (~(c & 1u) + 1u));
      t[0][i] = c;
    }
    for (u32 i = 0; i < 256; ++i) {
      for (std::size_t s = 1; s < 8; ++s) {
        const u32 prev = t[s - 1][i];
        t[s][i] = (prev >> 8) ^ t[0][prev & 0xFFu];
      }
    }
  }
};

const Crc32cTable& table() {
  static const Crc32cTable kTable;
  return kTable;
}

/// 先頭 4 バイトを「最初のバイトが最下位」となる u32 として読む。
/// 反転 CRC の定義に合わせた読み方であり、ホスト側のバイト順に依存しない。
inline u32 load_le32(const u8* p) noexcept {
  return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
         (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}

u32 crc32c_sw_update(u32 crc, const u8* p, std::size_t len) noexcept {
  const Crc32cTable& T = table();

  while (len >= 8) {
    const u32 lo = load_le32(p) ^ crc;
    const u32 hi = load_le32(p + 4);
    crc = T.t[7][lo & 0xFFu] ^ T.t[6][(lo >> 8) & 0xFFu] ^
          T.t[5][(lo >> 16) & 0xFFu] ^ T.t[4][(lo >> 24) & 0xFFu] ^
          T.t[3][hi & 0xFFu] ^ T.t[2][(hi >> 8) & 0xFFu] ^
          T.t[1][(hi >> 16) & 0xFFu] ^ T.t[0][(hi >> 24) & 0xFFu];
    p += 8;
    len -= 8;
  }
  while (len-- > 0) {
    crc = T.t[0][(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
  }
  return crc;
}

#if ALSR_HAVE_ARM_CRC32
u32 crc32c_hw_update(u32 crc, const u8* p, std::size_t len) noexcept {
  while (len >= 8) {
    const u64 word = static_cast<u64>(load_le32(p)) |
                     (static_cast<u64>(load_le32(p + 4)) << 32);
    crc = __crc32cd(crc, word);
    p += 8;
    len -= 8;
  }
  while (len-- > 0) crc = __crc32cb(crc, *p++);
  return crc;
}
#endif

u32 crc32c_update(u32 crc, const u8* p, std::size_t len) noexcept {
#if ALSR_HAVE_ARM_CRC32
  return crc32c_hw_update(crc, p, len);
#else
  return crc32c_sw_update(crc, p, len);
#endif
}

}  // namespace

void Crc32c::update(const void* data, std::size_t len) noexcept {
  state_ = crc32c_update(state_, static_cast<const u8*>(data), len);
}

u32 crc32c(const void* data, std::size_t len) noexcept {
  return ~crc32c_update(0xFFFFFFFFu, static_cast<const u8*>(data), len);
}

u32 crc32c_software(const void* data, std::size_t len) noexcept {
  return ~crc32c_sw_update(0xFFFFFFFFu, static_cast<const u8*>(data), len);
}

bool crc32c_hardware_enabled() noexcept { return ALSR_HAVE_ARM_CRC32 != 0; }

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------
namespace {

constexpr std::array<u32, 64> kK = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline u32 rotr(u32 x, unsigned n) noexcept { return (x >> n) | (x << (32u - n)); }

}  // namespace

void Sha256::reset() noexcept {
  h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  buf_len_ = 0;
  total_bits_ = 0;
}

void Sha256::compress(const u8* block) noexcept {
  std::array<u32, 64> w{};
  for (std::size_t i = 0; i < 16; ++i) {
    w[i] = (static_cast<u32>(block[i * 4]) << 24) |
           (static_cast<u32>(block[i * 4 + 1]) << 16) |
           (static_cast<u32>(block[i * 4 + 2]) << 8) |
           (static_cast<u32>(block[i * 4 + 3]));
  }
  for (std::size_t i = 16; i < 64; ++i) {
    const u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  u32 a = h_[0], b = h_[1], c = h_[2], d = h_[3];
  u32 e = h_[4], f = h_[5], g = h_[6], hh = h_[7];

  for (std::size_t i = 0; i < 64; ++i) {
    const u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const u32 ch = (e & f) ^ (~e & g);
    const u32 t1 = hh + S1 + ch + kK[i] + w[i];
    const u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const u32 maj = (a & b) ^ (a & c) ^ (b & c);
    const u32 t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }

  h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
  h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
}

void Sha256::update(const void* data, std::size_t len) noexcept {
  const u8* p = static_cast<const u8*>(data);
  total_bits_ += static_cast<u64>(len) * 8u;

  if (buf_len_ > 0) {
    const std::size_t need = 64 - buf_len_;
    const std::size_t take = len < need ? len : need;
    std::memcpy(buf_.data() + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    len -= take;
    if (buf_len_ == 64) { compress(buf_.data()); buf_len_ = 0; }
  }
  while (len >= 64) { compress(p); p += 64; len -= 64; }
  if (len > 0) { std::memcpy(buf_.data(), p, len); buf_len_ = len; }
}

std::array<u8, 32> Sha256::digest() noexcept {
  const u64 bits = total_bits_;

  // update() は 64 バイト溜まるたびに flush するので、ここでは buf_len_ < 64。
  buf_[buf_len_++] = static_cast<u8>(0x80);
  if (buf_len_ > 56) {
    while (buf_len_ < 64) buf_[buf_len_++] = 0;
    compress(buf_.data());
    buf_len_ = 0;
  }
  while (buf_len_ < 56) buf_[buf_len_++] = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    buf_[56 + i] = static_cast<u8>((bits >> (56u - 8u * i)) & 0xFFu);
  }
  compress(buf_.data());

  std::array<u8, 32> out{};
  for (std::size_t i = 0; i < 8; ++i) {
    out[i * 4]     = static_cast<u8>((h_[i] >> 24) & 0xFFu);
    out[i * 4 + 1] = static_cast<u8>((h_[i] >> 16) & 0xFFu);
    out[i * 4 + 2] = static_cast<u8>((h_[i] >> 8) & 0xFFu);
    out[i * 4 + 3] = static_cast<u8>(h_[i] & 0xFFu);
  }
  return out;
}

std::string Sha256::hex() {
  const auto d = digest();
  static constexpr char kHex[] = "0123456789abcdef";
  std::string s;
  s.reserve(64);
  for (const u8 byte : d) {
    s += kHex[(byte >> 4) & 0xF];
    s += kHex[byte & 0xF];
  }
  return s;
}

std::string sha256_hex(const void* data, std::size_t len) {
  Sha256 h;
  h.update(data, len);
  return h.hex();
}

std::string sha256_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(ErrorKind::Io, "cannot open file for sha256: " + path);

  Sha256 h;
  std::vector<char> buf(4 * 1024 * 1024);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = in.gcount();
    if (got > 0) h.update(buf.data(), static_cast<std::size_t>(got));
  }
  if (in.bad()) fail(ErrorKind::Io, "read error during sha256: " + path);
  return h.hex();
}

}  // namespace alsr
