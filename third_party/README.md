# third_party

ビルド時に取得する外部依存の置き場。中身は Git 管理外（`.gitignore` 参照）。

| 依存 | 版 | 固定方法 | ライセンス |
|---|---|---|---|
| Catch2 | v3.15.3 | `GIT_TAG 8b08d4d79514f45f7e4ce2a607ac9c94e920d1bb`（annotated tag の commit SHA） | BSL-1.0 |

タグは動き得るが commit SHA は動かない。NFR-02（再現性）のため、
`CMakeLists.txt` では必ず commit SHA で固定する。

コア（`alsr_core`）とすべての CLI は**外部依存ゼロ**である。
JSON も CRC32C も SHA-256 も自前実装（理由は `docs/DECISIONS.md` DR-0003）。
ここに取得物があるのはテストのためだけ。
