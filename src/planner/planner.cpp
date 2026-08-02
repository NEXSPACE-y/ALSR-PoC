#include "alsr/planner.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

#include "alsr/checksum.hpp"
#include "alsr/ggml_types.hpp"

namespace alsr {
namespace {

/// テンソル内部を切ってよい最小単位。
/// 量子ブロック境界と alignment の両方を満たす必要があるので最小公倍数。
Bytes split_granule(const GgufTensorInfo& t, Bytes alignment) {
  const Bytes block = ggml_block_bytes(t.type_id);
  return std::lcm(block, alignment == 0 ? Bytes{1} : alignment);
}

struct SortedTensor {
  const GgufTensorInfo* info = nullptr;
  u64  exec_order = 0;
};

/// 切ってよい位置（絶対オフセット）を全部集める。
std::vector<Bytes> build_cut_points(const GgufFile& f,
                                    const std::vector<SortedTensor>& ts,
                                    const PlanOptions& opt) {
  std::vector<Bytes> cuts;
  cuts.reserve(ts.size() * 2 + 8);
  cuts.push_back(f.data_start);
  cuts.push_back(f.file_size);

  const Bytes split_limit =
      (opt.mode == ChunkMode::Fixed) ? opt.target_bytes : opt.max_bytes;

  for (const auto& st : ts) {
    const GgufTensorInfo& t = *st.info;
    cuts.push_back(t.abs_offset);
    cuts.push_back(t.abs_offset + t.nbytes);

    if (split_limit > 0 && t.nbytes > split_limit) {
      const Bytes g = split_granule(t, f.alignment);
      Bytes piece = align_down(split_limit, g);
      if (piece == 0) piece = g;  // granule > 上限のときは granule 単位で刻む
      for (Bytes off = piece; off < t.nbytes; off += piece) {
        cuts.push_back(t.abs_offset + off);
      }
    }
  }

  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  // データ領域外は落とす
  cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                            [&](Bytes c) {
                              return c < f.data_start || c > f.file_size;
                            }),
             cuts.end());
  return cuts;
}

/// [begin, end) に掛かるテンソル片を列挙する。
void attach_tensors(const std::vector<SortedTensor>& ts, Chunk& c) {
  const Bytes cbeg = c.source_offset;
  const Bytes cend = c.source_offset + c.bytes;

  bool first = true;
  for (const auto& st : ts) {
    const GgufTensorInfo& t = *st.info;
    const Bytes tbeg = t.abs_offset;
    const Bytes tend = t.abs_offset + t.nbytes;
    if (tend <= cbeg) continue;
    if (tbeg >= cend) break;  // ts は offset 昇順

    const Bytes lo = std::max(tbeg, cbeg);
    const Bytes hi = std::min(tend, cend);
    if (hi <= lo) continue;

    ChunkTensorRef ref;
    ref.tensor_index     = t.file_index;
    ref.name             = t.name;
    ref.offset_in_chunk  = lo - cbeg;
    ref.offset_in_tensor = lo - tbeg;
    ref.bytes            = hi - lo;
    ref.partial          = (ref.bytes != t.nbytes);
    ref.exec_order       = st.exec_order;
    c.tensor_bytes += ref.bytes;

    if (first) {
      c.first_exec_order = ref.exec_order;
      c.last_exec_order  = ref.exec_order;
      first = false;
    } else {
      c.first_exec_order = std::min(c.first_exec_order, ref.exec_order);
      c.last_exec_order  = std::max(c.last_exec_order, ref.exec_order);
    }
    c.tensors.push_back(std::move(ref));
  }
  c.padding_bytes = c.bytes - c.tensor_bytes;
}

/// checksum をまとめて計算する。Chunk は offset 昇順なので逐次読みで済む。
void compute_checksums(const std::string& path, std::vector<Chunk>& chunks,
                       Bytes read_buffer_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(ErrorKind::Io, "cannot open model for checksum: " + path);

  const std::size_t bufsz =
      static_cast<std::size_t>(std::max<Bytes>(read_buffer_bytes, 64 * KiB));
  std::vector<char> buf(bufsz);

  for (Chunk& c : chunks) {
    in.seekg(static_cast<std::streamoff>(c.source_offset), std::ios::beg);
    if (!in) {
      std::ostringstream os;
      os << "seek failed at offset " << c.source_offset;
      fail(ErrorKind::Io, os.str());
    }
    Crc32c crc;
    Bytes remaining = c.bytes;
    while (remaining > 0) {
      const std::size_t want =
          static_cast<std::size_t>(std::min<Bytes>(remaining, bufsz));
      in.read(buf.data(), static_cast<std::streamsize>(want));
      const std::streamsize got = in.gcount();
      if (got <= 0 || static_cast<std::size_t>(got) != want) {
        std::ostringstream os;
        os << "short read while checksumming chunk " << c.chunk_id << " at "
           << c.source_offset << " (+" << (c.bytes - remaining) << ")";
        fail(ErrorKind::Io, os.str());
      }
      crc.update(buf.data(), want);
      remaining -= want;
    }
    c.crc32c = crc.value();
    c.checksum_computed = true;
  }
}

/// checksum を含まない構造だけの正規表現（決定論チェック用）。
std::string structure_digest(const ChunkPlan& p) {
  Sha256 h;
  const auto feed = [&h](const std::string& s) { h.update(s.data(), s.size()); };
  std::ostringstream os;
  os << "mode=" << to_string(p.options.mode) << ";target=" << p.options.target_bytes
     << ";min=" << p.options.min_bytes << ";max=" << p.options.max_bytes
     << ";align=" << p.alignment << ";data_start=" << p.data_start
     << ";file_size=" << p.file_size << ";n=" << p.chunks.size() << "\n";
  feed(os.str());
  for (const Chunk& c : p.chunks) {
    std::ostringstream l;
    l << c.chunk_id << ',' << c.source_offset << ',' << c.bytes << ','
      << c.tensor_bytes << ',' << c.padding_bytes << ',' << c.first_exec_order
      << ',' << c.last_exec_order << ',' << c.tensors.size() << '\n';
    feed(l.str());
  }
  return h.hex();
}

}  // namespace

std::string_view to_string(ChunkMode m) noexcept {
  switch (m) {
    case ChunkMode::Fixed:    return "fixed";
    case ChunkMode::Variable: return "variable";
  }
  return "unknown";
}

ChunkMode chunk_mode_from_string(std::string_view s) {
  if (s == "fixed") return ChunkMode::Fixed;
  if (s == "variable") return ChunkMode::Variable;
  fail(ErrorKind::Invalid,
       "unknown chunk mode '" + std::string(s) + "' (fixed|variable)");
}

Bytes ChunkPlan::min_chunk_bytes() const noexcept {
  Bytes v = ~Bytes{0};
  for (const auto& c : chunks) v = std::min(v, c.bytes);
  return chunks.empty() ? 0 : v;
}

Bytes ChunkPlan::max_chunk_bytes() const noexcept {
  Bytes v = 0;
  for (const auto& c : chunks) v = std::max(v, c.bytes);
  return v;
}

ChunkPlan build_plan(const GgufFile& f, const Manifest& m, const PlanOptions& opt) {
  if (opt.mode == ChunkMode::Fixed && opt.target_bytes == 0) {
    fail(ErrorKind::Invalid, "target_bytes must be > 0 for fixed mode");
  }
  if (opt.mode == ChunkMode::Variable && opt.max_bytes == 0) {
    fail(ErrorKind::Invalid, "max_bytes must be > 0 for variable mode");
  }
  if (opt.mode == ChunkMode::Variable && opt.min_bytes > opt.max_bytes) {
    fail(ErrorKind::Invalid, "min_bytes must be <= max_bytes");
  }
  if (f.file_size < f.data_start) {
    fail(ErrorKind::Consistency, "file_size < data_start");
  }

  ChunkPlan plan;
  plan.model_path   = f.path;
  plan.model_sha256 = m.model_sha256;
  plan.options      = opt;
  plan.alignment    = f.alignment;
  plan.data_start   = f.data_start;
  plan.file_size    = f.file_size;

  // exec_order をテンソル添字から引けるようにする
  std::vector<u64> exec_of_tensor(f.tensors.size(), 0);
  for (const auto& e : m.entries) {
    if (e.tensor_index < exec_of_tensor.size()) {
      exec_of_tensor[e.tensor_index] = e.exec_order;
    }
  }

  std::vector<SortedTensor> ts;
  ts.reserve(f.tensors.size());
  for (const auto& t : f.tensors) ts.push_back({&t, exec_of_tensor[t.file_index]});
  std::sort(ts.begin(), ts.end(), [](const SortedTensor& a, const SortedTensor& b) {
    if (a.info->abs_offset != b.info->abs_offset) {
      return a.info->abs_offset < b.info->abs_offset;
    }
    return a.info->file_index < b.info->file_index;
  });

  const std::vector<Bytes> cuts = build_cut_points(f, ts, opt);
  if (cuts.size() < 2) {
    fail(ErrorKind::Consistency, "cut point generation produced no interval");
  }

  // 目標サイズ。Variable は「min_bytes に達するまで併合」なので下限を使う。
  const Bytes accumulate_to =
      (opt.mode == ChunkMode::Fixed) ? opt.target_bytes : opt.min_bytes;

  u64 next_id = 0;
  std::size_t i = 1;  // cuts[0] == data_start
  Bytes cur = cuts[0];
  while (cur < f.file_size) {
    std::size_t j = i;
    while (j + 1 < cuts.size() && cuts[j + 1] - cur <= accumulate_to) ++j;
    const Bytes next = cuts[j];
    if (next <= cur) {
      fail(ErrorKind::Consistency, "cut points are not strictly increasing");
    }

    Chunk c;
    c.chunk_id      = next_id++;
    c.source_offset = cur;
    c.bytes         = next - cur;
    attach_tensors(ts, c);
    plan.chunks.push_back(std::move(c));

    cur = next;
    i = j + 1;
  }

  for (const Chunk& c : plan.chunks) {
    plan.covered_bytes       += c.bytes;
    plan.tensor_bytes_total  += c.tensor_bytes;
    plan.padding_bytes_total += c.padding_bytes;
  }

  // テンソルが複数 Chunk に跨った件数
  {
    std::vector<char> seen(f.tensors.size(), 0);
    for (const Chunk& c : plan.chunks) {
      for (const auto& r : c.tensors) {
        if (r.partial && !seen[r.tensor_index]) {
          seen[r.tensor_index] = 1;
          ++plan.split_tensor_count;
        }
      }
    }
  }

  // 実行順の並び。ファイル順との逆転数も数える。
  plan.exec_sequence.resize(plan.chunks.size());
  std::iota(plan.exec_sequence.begin(), plan.exec_sequence.end(), std::size_t{0});
  std::stable_sort(plan.exec_sequence.begin(), plan.exec_sequence.end(),
                   [&](std::size_t a, std::size_t b) {
                     return plan.chunks[a].first_exec_order <
                            plan.chunks[b].first_exec_order;
                   });
  for (std::size_t k = 0; k + 1 < plan.exec_sequence.size(); ++k) {
    if (plan.exec_sequence[k] > plan.exec_sequence[k + 1]) ++plan.exec_order_inversions;
  }

  if (opt.compute_checksums) {
    compute_checksums(f.path, plan.chunks, opt.read_buffer_bytes);
  }
  plan.plan_structure_sha256 = structure_digest(plan);
  return plan;
}

void verify_plan(const GgufFile& f, const ChunkPlan& plan) {
  if (plan.chunks.empty()) {
    fail(ErrorKind::Consistency, "plan has no chunks");
  }

  // 1. 隙間なく重複なく敷き詰めているか
  Bytes cursor = plan.data_start;
  for (const Chunk& c : plan.chunks) {
    if (c.source_offset != cursor) {
      std::ostringstream os;
      os << "chunk " << c.chunk_id << " starts at " << c.source_offset
         << " but previous chunk ended at " << cursor;
      fail(ErrorKind::Consistency, os.str());
    }
    if (c.bytes == 0) {
      std::ostringstream os;
      os << "chunk " << c.chunk_id << " is empty";
      fail(ErrorKind::Consistency, os.str());
    }
    if (c.tensor_bytes + c.padding_bytes != c.bytes) {
      std::ostringstream os;
      os << "chunk " << c.chunk_id << " byte accounting mismatch: tensor="
         << c.tensor_bytes << " padding=" << c.padding_bytes
         << " total=" << c.bytes;
      fail(ErrorKind::Consistency, os.str());
    }
    cursor += c.bytes;
  }
  if (cursor != plan.file_size) {
    std::ostringstream os;
    os << "plan covers up to " << cursor << " but file_size is " << plan.file_size;
    fail(ErrorKind::Consistency, os.str());
  }

  // 2. 各テンソルのバイトがちょうど 1 回ずつ現れるか
  std::vector<Bytes> covered(f.tensors.size(), 0);
  for (const Chunk& c : plan.chunks) {
    for (const auto& r : c.tensors) {
      if (r.tensor_index >= covered.size()) {
        fail(ErrorKind::Consistency, "chunk references unknown tensor index");
      }
      if (r.offset_in_chunk + r.bytes > c.bytes) {
        fail(ErrorKind::Consistency, "tensor ref exceeds chunk bounds");
      }
      // 断片は必ず前の断片の続きから始まること
      if (r.offset_in_tensor != covered[r.tensor_index]) {
        std::ostringstream os;
        os << "tensor '" << r.name << "' fragment starts at " << r.offset_in_tensor
           << " but " << covered[r.tensor_index] << " bytes were already covered";
        fail(ErrorKind::Consistency, os.str());
      }
      covered[r.tensor_index] += r.bytes;
    }
  }
  for (std::size_t k = 0; k < f.tensors.size(); ++k) {
    if (covered[k] != f.tensors[k].nbytes) {
      std::ostringstream os;
      os << "tensor '" << f.tensors[k].name << "' covered " << covered[k]
         << " of " << f.tensors[k].nbytes << " bytes";
      fail(ErrorKind::Consistency, os.str());
    }
  }

  // 3. テンソル内部の切れ目が量子ブロック境界と alignment を満たすか
  for (const Chunk& c : plan.chunks) {
    for (const auto& r : c.tensors) {
      if (r.offset_in_tensor == 0) continue;  // テンソル先頭は元々アラインされている
      const Bytes g = split_granule(f.tensors[r.tensor_index], plan.alignment);
      if (r.offset_in_tensor % g != 0) {
        std::ostringstream os;
        os << "tensor '" << r.name << "' split at " << r.offset_in_tensor
           << " which is not a multiple of granule " << g;
        fail(ErrorKind::Consistency, os.str());
      }
    }
  }
}

json::Value to_json(const ChunkPlan& plan) {
  json::Value root = json::Value::make_object();
  root.set("model_path", plan.model_path);
  root.set("model_sha256", plan.model_sha256);
  root.set("alignment", plan.alignment);
  root.set("data_start", plan.data_start);
  root.set("file_size", plan.file_size);

  json::Value o = json::Value::make_object();
  o.set("mode", std::string(to_string(plan.options.mode)));
  o.set("target_bytes", plan.options.target_bytes);
  o.set("min_bytes", plan.options.min_bytes);
  o.set("max_bytes", plan.options.max_bytes);
  o.set("compute_checksums", plan.options.compute_checksums);
  root.set("options", std::move(o));

  json::Value sum = json::Value::make_object();
  sum.set("chunk_count", plan.chunks.size());
  sum.set("covered_bytes", plan.covered_bytes);
  sum.set("tensor_bytes_total", plan.tensor_bytes_total);
  sum.set("padding_bytes_total", plan.padding_bytes_total);
  sum.set("min_chunk_bytes", plan.min_chunk_bytes());
  sum.set("max_chunk_bytes", plan.max_chunk_bytes());
  sum.set("split_tensor_count", plan.split_tensor_count);
  sum.set("exec_order_inversions", plan.exec_order_inversions);
  sum.set("plan_structure_sha256", plan.plan_structure_sha256);
  root.set("summary", std::move(sum));

  json::Value chunks = json::Value::make_array();
  for (const Chunk& c : plan.chunks) {
    json::Value j = json::Value::make_object();
    j.set("chunk_id", c.chunk_id);
    j.set("source_offset", c.source_offset);
    j.set("bytes", c.bytes);
    j.set("tensor_bytes", c.tensor_bytes);
    j.set("padding_bytes", c.padding_bytes);
    j.set("first_exec_order", c.first_exec_order);
    j.set("last_exec_order", c.last_exec_order);
    j.set("checksum_computed", c.checksum_computed);
    j.set("crc32c", c.crc32c);
    json::Value ts = json::Value::make_array();
    for (const auto& r : c.tensors) {
      json::Value t = json::Value::make_object();
      t.set("tensor_index", r.tensor_index);
      t.set("name", r.name);
      t.set("offset_in_chunk", r.offset_in_chunk);
      t.set("offset_in_tensor", r.offset_in_tensor);
      t.set("bytes", r.bytes);
      t.set("partial", r.partial);
      t.set("exec_order", r.exec_order);
      ts.push_back(std::move(t));
    }
    j.set("tensors", std::move(ts));
    chunks.push_back(std::move(j));
  }
  root.set("chunks", std::move(chunks));

  json::Value seq = json::Value::make_array();
  for (const std::size_t idx : plan.exec_sequence) seq.push_back(json::Value(idx));
  root.set("exec_sequence", std::move(seq));

  return root;
}

ChunkPlan plan_from_json(const json::Value& v) {
  ChunkPlan p;
  p.model_path   = v.get_string("model_path", "");
  p.model_sha256 = v.get_string("model_sha256", "");
  p.alignment    = static_cast<Bytes>(v.get_int("alignment", 32));
  p.data_start   = static_cast<Bytes>(v.get_int("data_start", 0));
  p.file_size    = static_cast<Bytes>(v.get_int("file_size", 0));

  const json::Value& o = v.at("options");
  p.options.mode = chunk_mode_from_string(o.get_string("mode", "fixed"));
  p.options.target_bytes = static_cast<Bytes>(o.get_int("target_bytes", 0));
  p.options.min_bytes    = static_cast<Bytes>(o.get_int("min_bytes", 0));
  p.options.max_bytes    = static_cast<Bytes>(o.get_int("max_bytes", 0));
  p.options.compute_checksums = o.get_bool("compute_checksums", false);

  for (const json::Value& cj : v.at("chunks").as_array()) {
    Chunk c;
    c.chunk_id         = static_cast<u64>(cj.at("chunk_id").as_int());
    c.source_offset    = static_cast<Bytes>(cj.at("source_offset").as_int());
    c.bytes            = static_cast<Bytes>(cj.at("bytes").as_int());
    c.tensor_bytes     = static_cast<Bytes>(cj.get_int("tensor_bytes", 0));
    c.padding_bytes    = static_cast<Bytes>(cj.get_int("padding_bytes", 0));
    c.first_exec_order = static_cast<u64>(cj.get_int("first_exec_order", 0));
    c.last_exec_order  = static_cast<u64>(cj.get_int("last_exec_order", 0));
    c.checksum_computed = cj.get_bool("checksum_computed", false);
    c.crc32c           = static_cast<u32>(cj.get_int("crc32c", 0));
    if (const json::Value* ts = cj.find("tensors")) {
      for (const json::Value& tj : ts->as_array()) {
        ChunkTensorRef r;
        r.tensor_index     = static_cast<std::size_t>(tj.get_int("tensor_index", 0));
        r.name             = tj.get_string("name", "");
        r.offset_in_chunk  = static_cast<Bytes>(tj.get_int("offset_in_chunk", 0));
        r.offset_in_tensor = static_cast<Bytes>(tj.get_int("offset_in_tensor", 0));
        r.bytes            = static_cast<Bytes>(tj.get_int("bytes", 0));
        r.partial          = tj.get_bool("partial", false);
        r.exec_order       = static_cast<u64>(tj.get_int("exec_order", 0));
        c.tensors.push_back(std::move(r));
      }
    }
    p.covered_bytes       += c.bytes;
    p.tensor_bytes_total  += c.tensor_bytes;
    p.padding_bytes_total += c.padding_bytes;
    p.chunks.push_back(std::move(c));
  }

  if (const json::Value* s = v.find("exec_sequence")) {
    for (const json::Value& e : s->as_array()) {
      p.exec_sequence.push_back(static_cast<std::size_t>(e.as_int()));
    }
  }
  if (p.exec_sequence.size() != p.chunks.size()) {
    p.exec_sequence.resize(p.chunks.size());
    std::iota(p.exec_sequence.begin(), p.exec_sequence.end(), std::size_t{0});
  }
  if (const json::Value* s = v.find("summary")) {
    p.plan_structure_sha256 = s->get_string("plan_structure_sha256", "");
    p.split_tensor_count =
        static_cast<std::size_t>(s->get_int("split_tensor_count", 0));
    p.exec_order_inversions =
        static_cast<std::size_t>(s->get_int("exec_order_inversions", 0));
  }
  return p;
}

}  // namespace alsr
