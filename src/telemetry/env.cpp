#include "alsr/env.hpp"

#include <array>
#include <cstdio>
#include <string>

#include "alsr/checksum.hpp"

#ifndef ALSR_VERSION_STRING
#  define ALSR_VERSION_STRING "unknown"
#endif

namespace alsr {
namespace {

std::string trim(std::string s) {
  const auto not_space = [](unsigned char c) {
    return c != ' ' && c != '\t' && c != '\n' && c != '\r';
  };
  std::size_t b = 0;
  while (b < s.size() && !not_space(static_cast<unsigned char>(s[b]))) ++b;
  std::size_t e = s.size();
  while (e > b && !not_space(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

json::Value text_or_unavailable(const std::string& cmd) {
  const std::string out = trim(capture_command(cmd));
  if (out.empty()) return json::Value(std::string("unavailable"));
  return json::Value(out);
}

std::string compiler_string() {
#if defined(__clang_version__)
  return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_VER);
#else
  return "unknown";
#endif
}

}  // namespace

std::string capture_command(const std::string& cmd) {
  // stderr は捨てる。取得できないこと自体は異常ではない。
  // パイプ全体を包まないと、先頭コマンドの stderr（nvcc: command not found 等）が
  // 漏れて Run のログを汚す。
  const std::string full = "{ " + cmd + " ; } 2>/dev/null";
  std::FILE* pipe = ::popen(full.c_str(), "r");
  if (pipe == nullptr) return {};

  std::string out;
  std::array<char, 512> buf{};
  while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    out += buf.data();
    if (out.size() > 256 * 1024) break;  // 想定外に長い出力は打ち切る
  }
  ::pclose(pipe);
  return out;
}

json::Value collect_environment() {
  json::Value e = json::Value::make_object();

  json::Value build = json::Value::make_object();
  build.set("alsr_version", std::string(ALSR_VERSION_STRING));
  build.set("compiler", compiler_string());
  build.set("cpp_standard", static_cast<long long>(__cplusplus));
  build.set("crc32c_hardware", crc32c_hardware_enabled());
#if defined(NDEBUG)
  build.set("ndebug", true);
#else
  build.set("ndebug", false);
#endif
  build.set("compiled_at", std::string(__DATE__ " " __TIME__));
  e.set("build", std::move(build));

  json::Value os = json::Value::make_object();
  os.set("uname", text_or_unavailable("uname -a"));
  os.set("product_version", text_or_unavailable("sw_vers -productVersion"));
  os.set("build_version", text_or_unavailable("sw_vers -buildVersion"));
  e.set("os", std::move(os));

  json::Value cpu = json::Value::make_object();
  cpu.set("brand", text_or_unavailable("sysctl -n machdep.cpu.brand_string"));
  cpu.set("logical_cores", text_or_unavailable("sysctl -n hw.ncpu"));
  cpu.set("perf_levels", text_or_unavailable("sysctl -n hw.nperflevels"));
  e.set("cpu", std::move(cpu));

  json::Value mem = json::Value::make_object();
  mem.set("memsize_bytes", text_or_unavailable("sysctl -n hw.memsize"));
  mem.set("page_size", text_or_unavailable("sysctl -n hw.pagesize"));
  mem.set("vm_stat_head", text_or_unavailable("vm_stat | head -6"));
  e.set("memory", std::move(mem));

  // GPU / PCIe。M5 では該当しない項目が多いので、無ければ unavailable。
  json::Value gpu = json::Value::make_object();
  gpu.set("metal_or_display",
          text_or_unavailable(
              "system_profiler SPDisplaysDataType 2>/dev/null | head -12"));
  gpu.set("nvidia_smi", text_or_unavailable("nvidia-smi --query-gpu=name,memory.total,"
                                            "pcie.link.gen.current,pcie.link.width.current"
                                            " --format=csv,noheader"));
  gpu.set("cuda_version", text_or_unavailable("nvcc --version | tail -2"));
  gpu.set("note",
          "M5 には独立 VRAM への PCIe H2D 経路が無い。GPU 項目が unavailable"
          "であることは異常ではない (設計書 §3.1)。");
  e.set("gpu", std::move(gpu));

  json::Value tools = json::Value::make_object();
  tools.set("ollama_version", text_or_unavailable("ollama --version"));
  tools.set("ollama_ps", text_or_unavailable("ollama ps"));
  tools.set("cmake", text_or_unavailable("cmake --version | head -1"));
  tools.set("python", text_or_unavailable("python3 --version"));
  e.set("tools", std::move(tools));

  json::Value git = json::Value::make_object();
  git.set("commit", text_or_unavailable("git rev-parse HEAD"));
  git.set("branch", text_or_unavailable("git rev-parse --abbrev-ref HEAD"));
  const std::string dirty = trim(capture_command("git status --porcelain"));
  git.set("dirty", !dirty.empty());
  e.set("git", std::move(git));

  e.set("collection_note",
        "取得できなかった項目は \"unavailable\"。捏造しない (設計書 §20)。");
  return e;
}

}  // namespace alsr
