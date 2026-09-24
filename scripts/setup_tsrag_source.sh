#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/repository_packages/external_rag/TS-RAG"
URL="https://github.com/UConn-DSIS/TS-RAG"
REV="73ac807789d2e61b8a3dfc8514e3fc947fe185cc"

if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing path: $DEST" >&2
  exit 2
fi

mkdir -p "$(dirname "$DEST")"
git clone --no-checkout "$URL" "$DEST"
git -C "$DEST" checkout --detach "$REV"
git -C "$DEST" apply "$ROOT/third_party/patches/ts-rag-anchor-rag.patch"
echo "TS-RAG source is ready at $DEST"
