#include "alsr/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace alsr::json {
namespace {

constexpr const char* kHexDigits = "0123456789abcdef";

void append_u16_escape(std::string& out, unsigned cp) {
  out += "\\u";
  out += kHexDigits[(cp >> 12) & 0xF];
  out += kHexDigits[(cp >> 8) & 0xF];
  out += kHexDigits[(cp >> 4) & 0xF];
  out += kHexDigits[cp & 0xF];
}

void append_utf8(std::string& out, unsigned cp) {
  if (cp <= 0x7F) {
    out += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// --------------------------------------------------------------------------
// 再帰下降パーサ
// --------------------------------------------------------------------------
class Parser {
public:
  explicit Parser(std::string_view text) : s_(text) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value(0);
    skip_ws();
    if (p_ != s_.size()) error("trailing characters after top-level value");
    return v;
  }

private:
  static constexpr int kMaxDepth = 128;

  std::string_view s_;
  std::size_t      p_ = 0;

  [[noreturn]] void error(const std::string& msg) const {
    std::ostringstream os;
    os << "JSON parse error at byte " << p_ << ": " << msg;
    fail(ErrorKind::Format, os.str());
  }

  [[nodiscard]] bool eof() const noexcept { return p_ >= s_.size(); }
  [[nodiscard]] char peek() const {
    if (eof()) error("unexpected end of input");
    return s_[p_];
  }

  void skip_ws() {
    while (p_ < s_.size()) {
      const char c = s_[p_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p_;
      else break;
    }
  }

  void expect(char c) {
    if (eof() || s_[p_] != c) {
      error(std::string("expected '") + c + "'");
    }
    ++p_;
  }

  bool literal(std::string_view lit) {
    if (s_.substr(p_, lit.size()) == lit) {
      p_ += lit.size();
      return true;
    }
    return false;
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) error("nesting too deep");
    switch (peek()) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': return Value(parse_string());
      case 't': if (literal("true"))  return Value(true);  error("invalid literal");
      case 'f': if (literal("false")) return Value(false); error("invalid literal");
      case 'n': if (literal("null"))  return Value();      error("invalid literal");
      default:  return parse_number();
    }
  }

  /// 重複キーは「後勝ち」。JSON 仕様は未定義動作なので方針を固定して文書化する。
  Value parse_object(int depth) {
    expect('{');
    Value obj = Value::make_object();
    skip_ws();
    if (!eof() && peek() == '}') { ++p_; return obj; }
    for (;;) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      obj.set(std::move(key), parse_value(depth + 1));
      skip_ws();
      if (eof()) error("unterminated object");
      if (s_[p_] == ',') { ++p_; continue; }
      if (s_[p_] == '}') { ++p_; break; }
      error("expected ',' or '}' in object");
    }
    return obj;
  }

  Value parse_array(int depth) {
    expect('[');
    Array items;
    skip_ws();
    if (!eof() && peek() == ']') { ++p_; return Value::make_array(std::move(items)); }
    for (;;) {
      skip_ws();
      items.push_back(parse_value(depth + 1));
      skip_ws();
      if (eof()) error("unterminated array");
      if (s_[p_] == ',') { ++p_; continue; }
      if (s_[p_] == ']') { ++p_; break; }
      error("expected ',' or ']' in array");
    }
    return Value::make_array(std::move(items));
  }

  unsigned parse_hex4() {
    if (p_ + 4 > s_.size()) error("truncated \\u escape");
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9')      v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
      else error("invalid hex digit in \\u escape");
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    for (;;) {
      if (eof()) error("unterminated string");
      const unsigned char c = static_cast<unsigned char>(s_[p_++]);
      if (c == '"') break;
      if (c < 0x20) error("unescaped control character in string");
      if (c != '\\') { out += static_cast<char>(c); continue; }

      if (eof()) error("unterminated escape");
      const char e = s_[p_++];
      switch (e) {
        case '"':  out += '"';  break;
        case '\\': out += '\\'; break;
        case '/':  out += '/';  break;
        case 'b':  out += '\b'; break;
        case 'f':  out += '\f'; break;
        case 'n':  out += '\n'; break;
        case 'r':  out += '\r'; break;
        case 't':  out += '\t'; break;
        case 'u': {
          unsigned cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate
            if (p_ + 1 < s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
              p_ += 2;
              const unsigned lo = parse_hex4();
              if (lo < 0xDC00 || lo > 0xDFFF) error("invalid low surrogate");
              cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
            } else {
              error("lone high surrogate");
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            error("lone low surrogate");
          }
          append_utf8(out, cp);
          break;
        }
        default: error("invalid escape character");
      }
    }
    return out;
  }

  Value parse_number() {
    const std::size_t start = p_;
    if (!eof() && s_[p_] == '-') ++p_;
    if (eof()) error("truncated number");

    if (s_[p_] == '0') {
      ++p_;
    } else if (s_[p_] >= '1' && s_[p_] <= '9') {
      while (!eof() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
    } else {
      error("invalid number");
    }

    bool is_real = false;
    if (!eof() && s_[p_] == '.') {
      is_real = true;
      ++p_;
      if (eof() || s_[p_] < '0' || s_[p_] > '9') error("digit expected after '.'");
      while (!eof() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
    }
    if (!eof() && (s_[p_] == 'e' || s_[p_] == 'E')) {
      is_real = true;
      ++p_;
      if (!eof() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
      if (eof() || s_[p_] < '0' || s_[p_] > '9') error("digit expected in exponent");
      while (!eof() && s_[p_] >= '0' && s_[p_] <= '9') ++p_;
    }

    const std::string text(s_.substr(start, p_ - start));
    if (!is_real) {
      errno = 0;
      char* endp = nullptr;
      const long long v = std::strtoll(text.c_str(), &endp, 10);
      if (errno == 0 && endp != nullptr && *endp == '\0') {
        return Value(static_cast<std::int64_t>(v));
      }
      // int64 に収まらない整数は実数として保持する（情報を捨てるより明示的）。
    }
    errno = 0;
    char* endp = nullptr;
    const double d = std::strtod(text.c_str(), &endp);
    if (endp == nullptr || *endp != '\0') error("invalid number literal");
    return Value(d);
  }
};

}  // namespace

// ---------------------------------------------------------------------------

std::string_view to_string(Type t) noexcept {
  switch (t) {
    case Type::Null:   return "null";
    case Type::Bool:   return "bool";
    case Type::Int:    return "int";
    case Type::Real:   return "real";
    case Type::String: return "string";
    case Type::Array:  return "array";
    case Type::Object: return "object";
  }
  return "?";
}

std::string format_real(double v) {
  if (!std::isfinite(v)) {
    fail(ErrorKind::Invalid,
         "non-finite double cannot be represented in JSON (got " +
             std::string(std::isnan(v) ? "NaN" : (v > 0 ? "+Inf" : "-Inf")) +
             "). 計測値の除算ガードを確認すること。");
  }
  char buf[64];
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*g", prec, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  std::string s(buf);
  if (s.find_first_of(".eE") == std::string::npos) s += ".0";
  return s;
}

std::string quote(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  out += '"';
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (c < 0x20) append_u16_escape(out, c);
        else          out += ch;   // UTF-8 はそのまま通す
    }
  }
  out += '"';
  return out;
}

Value Value::make_array(Array a) {
  Value v;
  v.t_ = Type::Array;
  v.arr_ = std::move(a);
  return v;
}

Value Value::make_object() {
  Value v;
  v.t_ = Type::Object;
  return v;
}

namespace {
[[noreturn]] void type_error(Type want, Type got) {
  fail(ErrorKind::Invalid,
       std::string("JSON type mismatch: expected ") + std::string(to_string(want)) +
           ", got " + std::string(to_string(got)));
}
}  // namespace

bool Value::as_bool() const {
  if (t_ != Type::Bool) type_error(Type::Bool, t_);
  return b_;
}

std::int64_t Value::as_int() const {
  if (t_ != Type::Int) type_error(Type::Int, t_);
  return i_;
}

double Value::as_real() const {
  if (t_ == Type::Real) return d_;
  if (t_ == Type::Int)  return static_cast<double>(i_);
  type_error(Type::Real, t_);
}

const std::string& Value::as_string() const {
  if (t_ != Type::String) type_error(Type::String, t_);
  return s_;
}

const Array& Value::as_array() const {
  if (t_ != Type::Array) type_error(Type::Array, t_);
  return arr_;
}

Array& Value::as_array() {
  if (t_ != Type::Array) type_error(Type::Array, t_);
  return arr_;
}

void Value::push_back(Value v) {
  if (t_ == Type::Null) { t_ = Type::Array; }
  if (t_ != Type::Array) type_error(Type::Array, t_);
  arr_.push_back(std::move(v));
}

void Value::set(std::string key, Value v) {
  if (t_ == Type::Null) { t_ = Type::Object; }
  if (t_ != Type::Object) type_error(Type::Object, t_);
  for (std::size_t idx = 0; idx < keys_.size(); ++idx) {
    if (keys_[idx] == key) { arr_[idx] = std::move(v); return; }
  }
  keys_.push_back(std::move(key));
  arr_.push_back(std::move(v));
}

std::size_t Value::object_size() const {
  if (t_ != Type::Object) type_error(Type::Object, t_);
  return keys_.size();
}

const std::string& Value::key_at(std::size_t i) const {
  if (t_ != Type::Object) type_error(Type::Object, t_);
  if (i >= keys_.size()) fail(ErrorKind::Invalid, "JSON object index out of range");
  return keys_[i];
}

const Value& Value::value_at(std::size_t i) const {
  if (t_ != Type::Object) type_error(Type::Object, t_);
  if (i >= arr_.size()) fail(ErrorKind::Invalid, "JSON object index out of range");
  return arr_[i];
}

const Value* Value::find(std::string_view key) const noexcept {
  if (t_ != Type::Object) return nullptr;
  for (std::size_t idx = 0; idx < keys_.size(); ++idx) {
    if (keys_[idx] == key) return &arr_[idx];
  }
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) {
    fail(ErrorKind::Invalid, "required JSON key missing: '" + std::string(key) + "'");
  }
  return *v;
}

bool Value::get_bool(std::string_view k, bool def) const {
  const Value* v = find(k);
  return v ? v->as_bool() : def;
}

std::int64_t Value::get_int(std::string_view k, std::int64_t def) const {
  const Value* v = find(k);
  return v ? v->as_int() : def;
}

double Value::get_real(std::string_view k, double def) const {
  const Value* v = find(k);
  return v ? v->as_real() : def;
}

std::string Value::get_string(std::string_view k, std::string def) const {
  const Value* v = find(k);
  return v ? v->as_string() : std::move(def);
}

void Value::dump_to(std::string& out, int indent, int depth) const {
  const bool pretty = indent >= 0;
  const auto newline_indent = [&](int d) {
    if (!pretty) return;
    out += '\n';
    out.append(static_cast<std::size_t>(indent * d), ' ');
  };

  switch (t_) {
    case Type::Null:   out += "null";  return;
    case Type::Bool:   out += (b_ ? "true" : "false"); return;
    case Type::Int:    out += std::to_string(i_); return;
    case Type::Real:   out += format_real(d_); return;
    case Type::String: out += quote(s_); return;

    case Type::Array: {
      if (arr_.empty()) { out += "[]"; return; }
      out += '[';
      bool first = true;
      for (const auto& item : arr_) {
        if (!first) out += ',';
        first = false;
        newline_indent(depth + 1);
        item.dump_to(out, indent, depth + 1);
      }
      newline_indent(depth);
      out += ']';
      return;
    }

    case Type::Object: {
      if (keys_.empty()) { out += "{}"; return; }
      out += '{';
      for (std::size_t idx = 0; idx < keys_.size(); ++idx) {
        if (idx != 0) out += ',';
        newline_indent(depth + 1);
        out += quote(keys_[idx]);
        out += ':';
        if (pretty) out += ' ';
        arr_[idx].dump_to(out, indent, depth + 1);
      }
      newline_indent(depth);
      out += '}';
      return;
    }
  }
}

std::string Value::dump(int indent) const {
  std::string out;
  dump_to(out, indent, 0);
  return out;
}

Value Value::parse(std::string_view text) {
  Parser p(text);
  return p.parse_document();
}

Value Value::parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(ErrorKind::Io, "cannot open JSON file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) fail(ErrorKind::Io, "read error on JSON file: " + path);
  try {
    return parse(ss.str());
  } catch (const Error& e) {
    fail(e.kind(), path + ": " + e.raw());
  }
}

}  // namespace alsr::json
