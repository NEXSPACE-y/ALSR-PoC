// テスト用ヘルパ — 合成 GGUF ファイルの生成など。
#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/ggml_types.hpp"

namespace alsr::test {

struct TensorSpec {
  std::string      name;
  u32              type_id = 0;
  std::vector<u64> dims;
};

/// テスト用の一時ディレクトリ。デストラクタで消す。
class TempDir {
public:
  explicit TempDir(const std::string& tag) {
    namespace fs = std::filesystem;
    // 決定論的な名前。テストはシングルスレッドで走らせる。
    path_ = fs::temp_directory_path() / ("alsr-test-" + tag);
    std::error_code ec;
    fs::remove_all(path_, ec);
    fs::create_directories(path_, ec);
    if (ec) fail(ErrorKind::Io, "cannot create temp dir: " + path_.string());
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  [[nodiscard]] std::string file(const std::string& name) const {
    return (path_ / name).string();
  }
  [[nodiscard]] std::string dir() const { return path_.string(); }

private:
  std::filesystem::path path_;
};

namespace detail {

inline void put_u32(std::string& out, u32 v) {
  for (int i = 0; i < 4; ++i) out += static_cast<char>((v >> (8 * i)) & 0xFFu);
}
inline void put_u64(std::string& out, u64 v) {
  for (int i = 0; i < 8; ++i) out += static_cast<char>((v >> (8 * i)) & 0xFFu);
}
inline void put_str(std::string& out, const std::string& s) {
  put_u64(out, s.size());
  out += s;
}

}  // namespace detail

/// テスト用データの生成規則（planner の checksum 検証と揃える）。
inline u8 test_byte(Bytes abs_offset) {
  const u64 x = abs_offset * 2862933555777941757ull + 3037000493ull;
  return static_cast<u8>((x >> 29) & 0xFFu);
}

struct GgufBuildResult {
  std::string      path;
  Bytes            file_size = 0;
  Bytes            data_start = 0;
  std::vector<Bytes> tensor_offsets;  ///< データ領域先頭からの相対
  std::vector<Bytes> tensor_bytes;
};

/// 最小構成の有効な GGUF v3 を書き出す。
inline GgufBuildResult write_test_gguf(const std::string& path,
                                       const std::vector<TensorSpec>& tensors,
                                       Bytes alignment = 32,
                                       u32 version = 3,
                                       const std::string& arch = "testarch",
                                       u32 block_count = 2) {
  using namespace detail;
  GgufBuildResult r;
  r.path = path;

  // --- テンソルのサイズとオフセット ---
  Bytes cursor = 0;
  for (const auto& t : tensors) {
    const Bytes n = ggml_tensor_bytes(t.type_id, t.dims);
    r.tensor_offsets.push_back(cursor);
    r.tensor_bytes.push_back(n);
    cursor = align_up(cursor + n, alignment);
  }
  const Bytes data_bytes = cursor;

  // --- ヘッダ ---
  std::string head;
  head += "GGUF";
  put_u32(head, version);
  put_u64(head, tensors.size());
  put_u64(head, 3);  // kv count

  put_str(head, "general.architecture");
  put_u32(head, 8);  // string
  put_str(head, arch);

  put_str(head, "general.alignment");
  put_u32(head, 4);  // uint32
  put_u32(head, static_cast<u32>(alignment));

  put_str(head, arch + ".block_count");
  put_u32(head, 4);  // uint32
  put_u32(head, block_count);

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    put_str(head, t.name);
    put_u32(head, static_cast<u32>(t.dims.size()));
    for (const u64 d : t.dims) put_u64(head, d);
    put_u32(head, t.type_id);
    put_u64(head, r.tensor_offsets[i]);
  }

  r.data_start = align_up(head.size(), alignment);
  head.append(static_cast<std::size_t>(r.data_start - head.size()), '\0');

  // --- 書き出し ---
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) fail(ErrorKind::Io, "cannot create test gguf: " + path);
  out.write(head.data(), static_cast<std::streamsize>(head.size()));

  std::vector<char> buf(64 * 1024);
  Bytes written = 0;
  while (written < data_bytes) {
    const Bytes n = std::min<Bytes>(buf.size(), data_bytes - written);
    for (Bytes k = 0; k < n; ++k) {
      buf[static_cast<std::size_t>(k)] =
          static_cast<char>(test_byte(r.data_start + written + k));
    }
    out.write(buf.data(), static_cast<std::streamsize>(n));
    written += n;
  }
  out.flush();
  if (!out) fail(ErrorKind::Io, "write failed for test gguf: " + path);
  out.close();

  r.file_size = r.data_start + data_bytes;
  return r;
}

}  // namespace alsr::test
