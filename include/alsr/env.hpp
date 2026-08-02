// ALSR PoC — 環境証拠の収集 (§3.3)。
//
// 「起動時に必ず収集する環境証拠」を機械可読で残す。
// 取得できなかった項目は捏造せず null / "unavailable" にする。
#pragma once

#include <string>

#include "alsr/json.hpp"

namespace alsr {

/// 実行環境を収集する。外部コマンドが無い場合でも失敗せず、
/// 該当項目を "unavailable" にして返す。
[[nodiscard]] json::Value collect_environment();

/// コマンドを実行して標準出力を返す。失敗時は空文字列。
/// 収集以外の用途に使わないこと。
[[nodiscard]] std::string capture_command(const std::string& cmd);

}  // namespace alsr
