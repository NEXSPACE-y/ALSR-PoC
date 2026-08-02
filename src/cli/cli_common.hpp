// CLI 共通ユーティリティ。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "alsr/common.hpp"
#include "alsr/json.hpp"

namespace alsr::cli {

/// --key value / --flag 形式の最小パーサ。
class Args {
public:
  Args(int argc, char** argv) {
    program_ = argc > 0 ? argv[0] : "alsr";
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        positional_.push_back(std::move(a));
        continue;
      }
      a = a.substr(2);
      const std::size_t eq = a.find('=');
      if (eq != std::string::npos) {
        opts_[a.substr(0, eq)] = a.substr(eq + 1);
        continue;
      }
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        opts_[a] = argv[++i];
      } else {
        opts_[a] = "true";
      }
    }
  }

  [[nodiscard]] bool has(const std::string& k) const { return opts_.count(k) > 0; }

  [[nodiscard]] std::string get(const std::string& k, const std::string& def = "") const {
    const auto it = opts_.find(k);
    return it == opts_.end() ? def : it->second;
  }

  [[nodiscard]] std::string require(const std::string& k) const {
    const auto it = opts_.find(k);
    if (it == opts_.end()) {
      fail(ErrorKind::Invalid, "missing required option --" + k);
    }
    return it->second;
  }

  [[nodiscard]] u64 get_u64(const std::string& k, u64 def) const {
    const auto it = opts_.find(k);
    if (it == opts_.end()) return def;
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(it->second.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0') {
      fail(ErrorKind::Invalid, "--" + k + " expects an integer, got '" + it->second + "'");
    }
    return static_cast<u64>(v);
  }

  [[nodiscard]] double get_double(const std::string& k, double def) const {
    const auto it = opts_.find(k);
    if (it == opts_.end()) return def;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(it->second.c_str(), &end);
    if (errno != 0 || end == nullptr || *end != '\0') {
      fail(ErrorKind::Invalid, "--" + k + " expects a number, got '" + it->second + "'");
    }
    return v;
  }

  [[nodiscard]] bool flag(const std::string& k) const {
    const auto it = opts_.find(k);
    return it != opts_.end() && it->second != "false" && it->second != "0";
  }

  [[nodiscard]] const std::string& program() const { return program_; }
  [[nodiscard]] const std::vector<std::string>& positional() const { return positional_; }

  /// 未知オプションを検出する。誤字を黙って無視しない。
  void reject_unknown(const std::vector<std::string>& known) const {
    for (const auto& [k, v] : opts_) {
      bool ok = false;
      for (const auto& n : known) {
        if (k == n) { ok = true; break; }
      }
      if (!ok) fail(ErrorKind::Invalid, "unknown option --" + k);
    }
  }

private:
  std::string program_;
  std::map<std::string, std::string> opts_;
  std::vector<std::string> positional_;
};

/// "gemma4:12b" 形式の Ollama 参照から model レイヤー(blob)の実体パスを得る。
/// 見つからなければ Error。読み取りのみ。
[[nodiscard]] inline std::string resolve_ollama_blob(const std::string& ref) {
  namespace fs = std::filesystem;

  std::string name = ref;
  std::string tag = "latest";
  const std::size_t colon = ref.find(':');
  if (colon != std::string::npos) {
    name = ref.substr(0, colon);
    tag  = ref.substr(colon + 1);
  }

  const char* home = std::getenv("HOME");
  if (home == nullptr) fail(ErrorKind::Io, "HOME is not set");
  const fs::path root = fs::path(home) / ".ollama" / "models";

  // ライブラリ以外の namespace も探す
  std::vector<fs::path> candidates;
  candidates.push_back(root / "manifests" / "registry.ollama.ai" / "library" / name / tag);
  const fs::path manifests = root / "manifests";
  std::error_code ec;
  if (fs::exists(manifests, ec)) {
    for (auto it = fs::recursive_directory_iterator(manifests, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
      if (!it->is_regular_file(ec)) continue;
      if (it->path().filename() == tag &&
          it->path().parent_path().filename() == name) {
        candidates.push_back(it->path());
      }
    }
  }

  for (const fs::path& m : candidates) {
    if (!fs::exists(m, ec)) continue;
    const json::Value j = json::Value::parse_file(m.string());
    const json::Value* layers = j.find("layers");
    if (layers == nullptr) continue;
    for (const json::Value& l : layers->as_array()) {
      if (l.get_string("mediaType", "") != "application/vnd.ollama.image.model") {
        continue;
      }
      std::string digest = l.get_string("digest", "");
      for (char& ch : digest) {
        if (ch == ':') ch = '-';
      }
      const fs::path blob = root / "blobs" / digest;
      if (!fs::exists(blob, ec)) {
        fail(ErrorKind::Io, "manifest points to missing blob: " + blob.string());
      }
      return blob.string();
    }
  }
  fail(ErrorKind::Io,
       "Ollama manifest for '" + ref + "' not found under " + root.string());
}

/// --model / --ollama のどちらかを解決する。
[[nodiscard]] inline std::string resolve_model_path(const Args& a) {
  if (a.has("model")) return a.get("model");
  if (a.has("ollama")) return resolve_ollama_blob(a.get("ollama"));
  fail(ErrorKind::Invalid, "--model <path> か --ollama <name:tag> のどちらかが必要");
}

/// 例外を捕まえて終了コードへ落とす共通ラッパ。
template <class F>
int run_main(F&& body) {
  try {
    return body();
  } catch (const Error& e) {
    std::cerr << "ERROR[" << to_string(e.kind()) << "] " << e.raw() << "\n";
    return e.kind() == ErrorKind::Unsupported ? 3 : 1;
  } catch (const std::exception& e) {
    std::cerr << "ERROR[unhandled] " << e.what() << "\n";
    return 2;
  }
}

}  // namespace alsr::cli
