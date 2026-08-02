#include "alsr/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "alsr/ggml_types.hpp"

namespace alsr {
namespace {

constexpr char kMagic[4] = {'G', 'G', 'U', 'F'};

/// GGUF ヘッダ部を読むための、境界検査付きストリームリーダ。
/// ファイル全体を読み込まない（モデル本体は 7GB 級のため）。
class HeaderReader {
public:
  HeaderReader(const std::string& path, Bytes file_size)
      : in_(path, std::ios::binary), path_(path), file_size_(file_size) {
    if (!in_) fail(ErrorKind::Io, "cannot open GGUF for reading: " + path);
  }

  [[nodiscard]] Bytes pos() const noexcept { return pos_; }

  void read_raw(void* dst, Bytes n) {
    if (n > file_size_ || pos_ > file_size_ - n) {
      std::ostringstream os;
      os << "short read: want " << n << " bytes at offset " << pos_
         << " but file is " << file_size_ << " bytes";
      fail(ErrorKind::Format, os.str());
    }
    in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in_ || static_cast<Bytes>(in_.gcount()) != n) {
      fail(ErrorKind::Io, "read failed on " + path_);
    }
    pos_ += n;
  }

  template <class T>
  T read_pod() {
    static_assert(std::is_trivially_copyable_v<T>);
    unsigned char buf[sizeof(T)];
    read_raw(buf, sizeof(T));
    T v;
    std::memcpy(&v, buf, sizeof(T));  // GGUF はリトルエンディアン固定
    return v;
  }

  std::string read_string() {
    const u64 len = read_pod<u64>();
    // 単一の文字列が数 MB を超えるのは想定外。壊れたファイルで
    // 巨大 allocate に走らないよう上限を設ける。
    constexpr u64 kMaxStringBytes = 64ull * 1024ull * 1024ull;
    if (len > kMaxStringBytes) {
      std::ostringstream os;
      os << "implausible string length " << len << " at offset " << pos_;
      fail(ErrorKind::Format, os.str());
    }
    std::string s;
    s.resize(static_cast<std::size_t>(len));
    if (len > 0) read_raw(s.data(), len);
    return s;
  }

private:
  std::ifstream in_;
  std::string   path_;
  Bytes         file_size_ = 0;
  Bytes         pos_ = 0;
};

json::Value read_scalar(HeaderReader& r, GgufValueType t) {
  switch (t) {
    case GgufValueType::Uint8:   return json::Value(r.read_pod<u8>());
    case GgufValueType::Int8:    return json::Value(r.read_pod<std::int8_t>());
    case GgufValueType::Uint16:  return json::Value(r.read_pod<u16>());
    case GgufValueType::Int16:   return json::Value(r.read_pod<std::int16_t>());
    case GgufValueType::Uint32:  return json::Value(r.read_pod<u32>());
    case GgufValueType::Int32:   return json::Value(r.read_pod<i32>());
    case GgufValueType::Float32: return json::Value(r.read_pod<float>());
    case GgufValueType::Bool:    return json::Value(r.read_pod<u8>() != 0);
    case GgufValueType::String:  return json::Value(r.read_string());
    case GgufValueType::Uint64:  return json::Value(r.read_pod<u64>());
    case GgufValueType::Int64:   return json::Value(r.read_pod<i64>());
    case GgufValueType::Float64: return json::Value(r.read_pod<double>());
    case GgufValueType::Array:
      fail(ErrorKind::Format, "nested arrays are not part of GGUF");
  }
  std::ostringstream os;
  os << "unknown GGUF value type id " << static_cast<u32>(t);
  fail(ErrorKind::Format, os.str());
}

/// 配列を読む。要素は必ず全部消費する（オフセットがずれると全部壊れるため）。
/// ただし JSON に載せるのは array_inline_limit まで。
json::Value read_array(HeaderReader& r, std::size_t inline_limit) {
  const auto elem_type = static_cast<GgufValueType>(r.read_pod<u32>());
  const u64  count     = r.read_pod<u64>();

  json::Value inlined = json::Value::make_array();
  for (u64 k = 0; k < count; ++k) {
    json::Value v = read_scalar(r, elem_type);
    if (k < inline_limit) inlined.push_back(std::move(v));
  }

  if (count <= inline_limit) return inlined;

  json::Value summary = json::Value::make_object();
  summary.set("_elided", true);
  summary.set("_element_type", std::string(gguf_value_type_name(elem_type)));
  summary.set("_count", count);
  summary.set("_head", std::move(inlined));
  return summary;
}

/// レイアウト検査。テンソル範囲の重複・欠落を「説明可能か」で分類する。
GgufLayoutReport check_layout(const GgufFile& f) {
  GgufLayoutReport rep;
  rep.data_start = f.data_start;
  rep.data_bytes = f.file_size >= f.data_start ? f.file_size - f.data_start : 0;

  if (f.file_size < f.data_start) {
    rep.issues.push_back({"out_of_range", f.data_start, 0,
                          "data_start がファイルサイズを超えている"});
    return rep;
  }

  std::vector<const GgufTensorInfo*> sorted;
  sorted.reserve(f.tensors.size());
  for (const auto& t : f.tensors) sorted.push_back(&t);
  std::sort(sorted.begin(), sorted.end(),
            [](const GgufTensorInfo* a, const GgufTensorInfo* b) {
              if (a->rel_offset != b->rel_offset) return a->rel_offset < b->rel_offset;
              return a->file_index < b->file_index;
            });

  Bytes cursor = 0;  // データ領域先頭からの相対位置
  for (const GgufTensorInfo* t : sorted) {
    rep.tensor_bytes_total += t->nbytes;

    if (t->rel_offset + t->nbytes > rep.data_bytes) {
      std::ostringstream os;
      os << t->name << " がデータ領域の外へはみ出す (rel_offset=" << t->rel_offset
         << ", nbytes=" << t->nbytes << ", data_bytes=" << rep.data_bytes << ")";
      rep.issues.push_back({"out_of_range", t->abs_offset, t->nbytes, os.str()});
      continue;
    }

    if (t->rel_offset < cursor) {
      std::ostringstream os;
      os << t->name << " が直前のテンソルと重複 (rel_offset=" << t->rel_offset
         << " < cursor=" << cursor << ")";
      rep.issues.push_back({"overlap", t->abs_offset, cursor - t->rel_offset, os.str()});
    } else if (t->rel_offset > cursor) {
      const Bytes gap = t->rel_offset - cursor;
      if (t->rel_offset == align_up(cursor, f.alignment)) {
        rep.padding_bytes += gap;  // alignment で説明できる
      } else {
        std::ostringstream os;
        os << t->name << " の直前に説明できない空きがある (" << gap
           << " bytes, alignment=" << f.alignment << ")";
        rep.issues.push_back({"gap", f.data_start + cursor, gap, os.str()});
        rep.unexplained_bytes += gap;
      }
    }
    cursor = std::max(cursor, t->rel_offset + t->nbytes);
  }

  // 末尾。alignment 未満なら埋めとして説明可能。
  if (rep.data_bytes > cursor) {
    const Bytes tail = rep.data_bytes - cursor;
    if (rep.data_bytes == align_up(cursor, f.alignment)) {
      rep.padding_bytes += tail;
    } else {
      std::ostringstream os;
      os << "ファイル末尾に説明できない " << tail << " bytes が残る";
      rep.issues.push_back({"tail", f.data_start + cursor, tail, os.str()});
      rep.unexplained_bytes += tail;
    }
  }
  return rep;
}

}  // namespace

std::string_view gguf_value_type_name(GgufValueType t) noexcept {
  switch (t) {
    case GgufValueType::Uint8:   return "uint8";
    case GgufValueType::Int8:    return "int8";
    case GgufValueType::Uint16:  return "uint16";
    case GgufValueType::Int16:   return "int16";
    case GgufValueType::Uint32:  return "uint32";
    case GgufValueType::Int32:   return "int32";
    case GgufValueType::Float32: return "float32";
    case GgufValueType::Bool:    return "bool";
    case GgufValueType::String:  return "string";
    case GgufValueType::Array:   return "array";
    case GgufValueType::Uint64:  return "uint64";
    case GgufValueType::Int64:   return "int64";
    case GgufValueType::Float64: return "float64";
  }
  return "unknown";
}

const GgufTensorInfo* GgufFile::find_tensor(std::string_view name) const noexcept {
  for (const auto& t : tensors) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

GgufFile read_gguf(const std::string& path, const GgufReadOptions& opt) {
  GgufFile f;
  f.path = path;

  {
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    if (!probe) fail(ErrorKind::Io, "cannot open GGUF: " + path);
    const auto end = probe.tellg();
    if (end < 0) fail(ErrorKind::Io, "cannot determine size of " + path);
    f.file_size = static_cast<Bytes>(end);
  }

  HeaderReader r(path, f.file_size);

  char magic[4];
  r.read_raw(magic, 4);
  if (std::memcmp(magic, kMagic, 4) != 0) {
    fail(ErrorKind::Format,
         "not a GGUF file (magic mismatch): " + path +
             " — Ollama blob なら model レイヤーの digest を確認すること");
  }

  f.version = r.read_pod<u32>();
  if (f.version != 3) {
    std::ostringstream os;
    os << "GGUF version " << f.version
       << " は本 PoC 未対応 (対応: v3)。推測で読み進めない";
    fail(ErrorKind::Unsupported, os.str());
  }

  f.tensor_count = r.read_pod<u64>();
  f.kv_count     = r.read_pod<u64>();

  constexpr u64 kMaxTensors = 1'000'000;
  constexpr u64 kMaxKv      = 100'000;
  if (f.tensor_count > kMaxTensors || f.kv_count > kMaxKv) {
    std::ostringstream os;
    os << "implausible header counts: tensors=" << f.tensor_count
       << " kv=" << f.kv_count;
    fail(ErrorKind::Format, os.str());
  }

  // --- メタデータ KV ---------------------------------------------------
  f.metadata = json::Value::make_object();
  for (u64 i = 0; i < f.kv_count; ++i) {
    std::string key = r.read_string();
    const auto  vt  = static_cast<GgufValueType>(r.read_pod<u32>());
    json::Value v = (vt == GgufValueType::Array) ? read_array(r, opt.array_inline_limit)
                                                 : read_scalar(r, vt);
    f.metadata.set(std::move(key), std::move(v));
  }

  if (const json::Value* a = f.metadata.find("general.architecture")) {
    if (a->is_string()) f.architecture = a->as_string();
  }
  if (const json::Value* al = f.metadata.find("general.alignment")) {
    if (!al->is_int() || al->as_int() <= 0) {
      fail(ErrorKind::Format, "general.alignment が正の整数でない");
    }
    f.alignment = static_cast<Bytes>(al->as_int());
    if ((f.alignment & (f.alignment - 1)) != 0) {
      std::ostringstream os;
      os << "general.alignment=" << f.alignment << " が 2 の冪でない";
      fail(ErrorKind::Format, os.str());
    }
  }

  // --- テンソル情報 ----------------------------------------------------
  f.tensors.reserve(static_cast<std::size_t>(f.tensor_count));
  for (u64 i = 0; i < f.tensor_count; ++i) {
    GgufTensorInfo t;
    t.file_index = static_cast<std::size_t>(i);
    t.name = r.read_string();

    const u32 rank = r.read_pod<u32>();
    if (rank == 0 || rank > 4) {
      std::ostringstream os;
      os << "tensor '" << t.name << "' の rank=" << rank << " が想定外 (1..4)";
      fail(ErrorKind::Format, os.str());
    }
    t.dims.resize(rank);
    for (u32 d = 0; d < rank; ++d) t.dims[d] = r.read_pod<u64>();

    t.type_id    = r.read_pod<u32>();
    t.rel_offset = r.read_pod<u64>();
    t.nbytes     = ggml_tensor_bytes(t.type_id, t.dims);  // 未対応型はここで失敗

    if (t.rel_offset % f.alignment != 0) {
      std::ostringstream os;
      os << "tensor '" << t.name << "' の offset " << t.rel_offset
         << " が alignment " << f.alignment << " の倍数でない";
      fail(ErrorKind::Format, os.str());
    }
    f.tensors.push_back(std::move(t));
  }

  f.header_end = r.pos();
  f.data_start = align_up(f.header_end, f.alignment);
  for (auto& t : f.tensors) t.abs_offset = f.data_start + t.rel_offset;

  f.layout = check_layout(f);
  return f;
}

json::Value to_json(const GgufFile& f) {
  json::Value root = json::Value::make_object();
  root.set("path", f.path);
  root.set("file_size_bytes", f.file_size);
  root.set("gguf_version", f.version);
  root.set("architecture", f.architecture);
  root.set("alignment", f.alignment);
  root.set("header_end", f.header_end);
  root.set("data_start", f.data_start);
  root.set("tensor_count", f.tensor_count);
  root.set("kv_count", f.kv_count);
  root.set("metadata", f.metadata);

  json::Value tensors = json::Value::make_array();
  for (const auto& t : f.tensors) {
    json::Value j = json::Value::make_object();
    j.set("index", t.file_index);
    j.set("name", t.name);
    j.set("type_id", t.type_id);
    j.set("type", ggml_type_name(t.type_id));
    json::Value dims = json::Value::make_array();
    for (const u64 d : t.dims) dims.push_back(json::Value(d));
    j.set("shape", std::move(dims));
    j.set("rel_offset", t.rel_offset);
    j.set("abs_offset", t.abs_offset);
    j.set("bytes", t.nbytes);
    // FR-01「alignment を出力する」。ファイル全体の値と、このテンソルが
    // それを満たしているかを両方出す。
    j.set("alignment", f.alignment);
    j.set("alignment_ok", (t.rel_offset % f.alignment) == 0);
    const GgmlTypeInfo* info = ggml_type_info(t.type_id);
    j.set("block_elems", info ? info->block_elems : 0);
    j.set("block_bytes", info ? info->block_bytes : 0);
    tensors.push_back(std::move(j));
  }
  root.set("tensors", std::move(tensors));

  json::Value lay = json::Value::make_object();
  lay.set("data_start", f.layout.data_start);
  lay.set("data_bytes", f.layout.data_bytes);
  lay.set("tensor_bytes_total", f.layout.tensor_bytes_total);
  lay.set("padding_bytes", f.layout.padding_bytes);
  lay.set("unexplained_bytes", f.layout.unexplained_bytes);
  lay.set("ok", f.layout.ok());
  json::Value issues = json::Value::make_array();
  for (const auto& is : f.layout.issues) {
    json::Value j = json::Value::make_object();
    j.set("kind", is.kind);
    j.set("offset", is.offset);
    j.set("length", is.length);
    j.set("detail", is.detail);
    issues.push_back(std::move(j));
  }
  lay.set("issues", std::move(issues));
  root.set("layout", std::move(lay));

  return root;
}

}  // namespace alsr
