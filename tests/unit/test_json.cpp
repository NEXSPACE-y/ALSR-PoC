#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <string>

#include "alsr/json.hpp"

using namespace alsr;
using json::Value;

TEST_CASE("JSON: スカラの往復", "[unit][json]") {
  REQUIRE(Value(true).dump() == "true");
  REQUIRE(Value(false).dump() == "false");
  REQUIRE(Value().dump() == "null");
  REQUIRE(Value(42).dump() == "42");
  REQUIRE(Value(-7).dump() == "-7");
  REQUIRE(Value(std::uint64_t{7381382048}).dump() == "7381382048");
  REQUIRE(Value("abc").dump() == "\"abc\"");
}

TEST_CASE("JSON: 実数は最短往復表現になる", "[unit][json]") {
  REQUIRE(json::format_real(1.0) == "1.0");
  REQUIRE(json::format_real(0.5) == "0.5");
  REQUIRE(json::format_real(3.33) == "3.33");
  // 往復すること
  for (const double v : {0.1, 1e-7, 1.7976931348623157e308, -2.5, 1234567.891}) {
    const std::string s = json::format_real(v);
    REQUIRE(Value::parse(s).as_real() == v);
  }
}

TEST_CASE("JSON: 非有限値は捏造せず失敗する", "[unit][json]") {
  REQUIRE_THROWS_AS(json::format_real(std::numeric_limits<double>::infinity()), Error);
  REQUIRE_THROWS_AS(json::format_real(std::numeric_limits<double>::quiet_NaN()), Error);
}

TEST_CASE("JSON: オブジェクトは挿入順を保つ（再現性）", "[unit][json]") {
  Value o = Value::make_object();
  o.set("zebra", 1);
  o.set("apple", 2);
  o.set("mango", 3);
  REQUIRE(o.dump() == R"({"zebra":1,"apple":2,"mango":3})");

  // 上書きしても位置は変わらない
  o.set("zebra", 9);
  REQUIRE(o.dump() == R"({"zebra":9,"apple":2,"mango":3})");
  REQUIRE(o.object_size() == 3);
  REQUIRE(o.key_at(0) == "zebra");
}

TEST_CASE("JSON: 同一入力から同一バイト列（決定論）", "[unit][json]") {
  const std::string src =
      R"({"b":[1,2,{"c":null}],"a":{"x":1.5,"y":"日本語 \u00e9"},"z":true})";
  const Value v1 = Value::parse(src);
  const Value v2 = Value::parse(v1.dump());
  REQUIRE(v1.dump() == v2.dump());
  REQUIRE(v1.dump(2) == v2.dump(2));
}

TEST_CASE("JSON: パース失敗は Error(Format)", "[unit][json]") {
  const char* bad[] = {
      "{", "[1,]", "{\"a\"}", "{\"a\":}", "01", "1.", "1e", "\"unterminated",
      "tru", "{\"a\":1}x", "[\"\\uD800\"]", "[\"\\q\"]",
  };
  for (const char* s : bad) {
    INFO("input: " << s);
    REQUIRE_THROWS_AS(Value::parse(s), Error);
  }
}

TEST_CASE("JSON: 制御文字とサロゲートペア", "[unit][json]") {
  const Value v = Value::parse(R"("a\u0001b\uD83D\uDE00c")");
  const std::string& s = v.as_string();
  REQUIRE(s.size() == 3 + 1 + 4);  // a, \x01, b, emoji(4byte), c
  // 書き戻すと制御文字はエスケープされる
  REQUIRE(v.dump().find("\\u0001") != std::string::npos);
  // 往復
  REQUIRE(Value::parse(v.dump()).as_string() == s);
}

TEST_CASE("JSON: 型不一致は既定値化せず投げる", "[unit][json]") {
  const Value v = Value::parse(R"({"n":1,"s":"x"})");
  REQUIRE(v.at("n").as_int() == 1);
  REQUIRE_THROWS_AS(v.at("n").as_string(), Error);
  REQUIRE_THROWS_AS(v.at("s").as_int(), Error);
  REQUIRE_THROWS_AS(v.at("missing"), Error);
  REQUIRE(v.get_int("missing", 7) == 7);
  // int は real として読める（逆は不可）
  REQUIRE(v.at("n").as_real() == 1.0);
}

TEST_CASE("JSON: 深すぎるネストは失敗する", "[unit][json]") {
  std::string deep;
  for (int i = 0; i < 200; ++i) deep += '[';
  for (int i = 0; i < 200; ++i) deep += ']';
  REQUIRE_THROWS_AS(Value::parse(deep), Error);
}

TEST_CASE("JSON: 大きな整数はバイト数として正確に保持される", "[unit][json]") {
  // モデルサイズ 7,381,382,048 が double 化されないこと
  const Value v = Value::parse("7381382048");
  REQUIRE(v.is_int());
  REQUIRE(v.as_int() == 7381382048LL);
  REQUIRE(v.dump() == "7381382048");
}
