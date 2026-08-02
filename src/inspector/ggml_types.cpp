#include "alsr/ggml_types.hpp"

#include <sstream>
#include <unordered_map>

namespace alsr {
namespace {

// ---------------------------------------------------------------------------
// ggml の block 定義。
//   storable=false は「GGUF ファイル内のテンソル格納型としては現れない、
//   あるいは本 PoC が扱わない」もの。現れたら明示エラーにする。
//   削除済み ID（Q4_2/Q4_3、Q4_0_4_4 系、IQ4_NL_4_4 系）はテーブルに載せず、
//   未知 ID として扱う。
// ---------------------------------------------------------------------------
const std::vector<GgmlTypeInfo>& table_impl() {
  static const std::vector<GgmlTypeInfo> kTable = {
      { 0, "F32",      1,   4,   true,  ""},
      { 1, "F16",      1,   2,   true,  ""},
      { 2, "Q4_0",     32,  18,  true,  ""},
      { 3, "Q4_1",     32,  20,  true,  ""},
      { 6, "Q5_0",     32,  22,  true,  ""},
      { 7, "Q5_1",     32,  24,  true,  ""},
      { 8, "Q8_0",     32,  34,  true,  ""},
      { 9, "Q8_1",     32,  36,  false,
        "中間量子化型。GGUF 格納型としては想定外。現れたら実装を見直すこと"},
      {10, "Q2_K",     256, 84,  true,  ""},
      {11, "Q3_K",     256, 110, true,  ""},
      {12, "Q4_K",     256, 144, true,  ""},
      {13, "Q5_K",     256, 176, true,  ""},
      {14, "Q6_K",     256, 210, true,  ""},
      {15, "Q8_K",     256, 292, false,
        "中間量子化型。GGUF 格納型としては想定外。現れたら実装を見直すこと"},
      {16, "IQ2_XXS",  256, 66,  true,  ""},
      {17, "IQ2_XS",   256, 74,  true,  ""},
      {18, "IQ3_XXS",  256, 98,  true,  ""},
      {19, "IQ1_S",    256, 50,  true,  ""},
      {20, "IQ4_NL",   32,  18,  true,  ""},
      {21, "IQ3_S",    256, 110, true,  ""},
      {22, "IQ2_S",    256, 82,  true,  ""},
      {23, "IQ4_XS",   256, 136, true,  ""},
      {24, "I8",       1,   1,   true,  ""},
      {25, "I16",      1,   2,   true,  ""},
      {26, "I32",      1,   4,   true,  ""},
      {27, "I64",      1,   8,   true,  ""},
      {28, "F64",      1,   8,   true,  ""},
      {29, "IQ1_M",    256, 56,  true,  ""},
      {30, "BF16",     1,   2,   true,  ""},
      {34, "TQ1_0",    256, 54,  true,  ""},
      {35, "TQ2_0",    256, 66,  true,  ""},
      {39, "MXFP4",    32,  17,  true,  ""},
  };
  return kTable;
}

const std::unordered_map<u32, const GgmlTypeInfo*>& index() {
  static const std::unordered_map<u32, const GgmlTypeInfo*> kIndex = [] {
    std::unordered_map<u32, const GgmlTypeInfo*> m;
    for (const auto& e : table_impl()) m.emplace(e.id, &e);
    return m;
  }();
  return kIndex;
}

const GgmlTypeInfo& require_storable(u32 type_id) {
  const GgmlTypeInfo* info = ggml_type_info(type_id);
  if (info == nullptr) {
    std::ostringstream os;
    os << "unknown ggml type id " << type_id
       << " (削除済み ID か、本 PoC のテーブル未収録。推測で処理しない)";
    fail(ErrorKind::Unsupported, os.str());
  }
  if (!info->storable) {
    std::ostringstream os;
    os << "ggml type " << info->name << " (id " << type_id
       << ") は格納型として未対応: " << info->note;
    fail(ErrorKind::Unsupported, os.str());
  }
  return *info;
}

}  // namespace

const std::vector<GgmlTypeInfo>& ggml_type_table() { return table_impl(); }

const GgmlTypeInfo* ggml_type_info(u32 id) noexcept {
  const auto& m = index();
  const auto it = m.find(id);
  return it == m.end() ? nullptr : it->second;
}

std::string ggml_type_name(u32 id) {
  if (const GgmlTypeInfo* i = ggml_type_info(id)) return i->name;
  return "unknown(" + std::to_string(id) + ")";
}

Bytes ggml_block_bytes(u32 type_id) { return require_storable(type_id).block_bytes; }

Bytes ggml_tensor_bytes(u32 type_id, const std::vector<u64>& dims) {
  const GgmlTypeInfo& info = require_storable(type_id);

  if (dims.empty()) {
    fail(ErrorKind::Format, "tensor has zero dimensions");
  }
  if (dims.size() > 4) {
    std::ostringstream os;
    os << "tensor rank " << dims.size() << " > 4 は ggml の想定外";
    fail(ErrorKind::Unsupported, os.str());
  }

  const u64 ne0 = dims[0];
  if (info.block_elems == 0) {
    fail(ErrorKind::Unsupported, "block_elems == 0 (テーブル定義不備)");
  }
  if (ne0 % info.block_elems != 0) {
    std::ostringstream os;
    os << "ne[0]=" << ne0 << " が " << info.name << " のブロック長 "
       << info.block_elems << " の倍数でない";
    fail(ErrorKind::Format, os.str());
  }

  // 行数 = ne1*ne2*ne3。オーバーフローを都度検査する。
  u64 rows = 1;
  for (std::size_t d = 1; d < dims.size(); ++d) {
    if (dims[d] != 0 && rows > (~0ull) / dims[d]) {
      fail(ErrorKind::Format, "tensor element count overflow");
    }
    rows *= dims[d];
  }

  const u64 blocks_per_row = ne0 / info.block_elems;
  if (blocks_per_row != 0 && info.block_bytes > (~0ull) / blocks_per_row) {
    fail(ErrorKind::Format, "tensor byte size overflow (row)");
  }
  const u64 row_bytes = blocks_per_row * info.block_bytes;
  if (rows != 0 && row_bytes != 0 && rows > (~0ull) / row_bytes) {
    fail(ErrorKind::Format, "tensor byte size overflow (total)");
  }
  return row_bytes * rows;
}

}  // namespace alsr
