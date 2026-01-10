#!/usr/bin/env bash
set -euo pipefail

# Extract a .tar.gz bundle to a target directory.
#
# Usage:
#   ./extract_tarball_to_dir.sh <tar.gz> [out_dir]

TARBALL="${1:?usage: $0 <tar.gz> [out_dir]}"
OUT_DIR="${2:-}"

if [[ ! -f "$TARBALL" ]]; then
  echo "[err] not a file: $TARBALL" >&2
  exit 2
fi

if [[ -z "$OUT_DIR" ]]; then
  base="$(basename "$TARBALL")"
  base="${base%.tar.gz}"
  base="${base%.tgz}"
  OUT_DIR="$(dirname "$TARBALL")/${base}"
fi

mkdir -p "$OUT_DIR"
echo "[tar] extracting ${TARBALL} -> ${OUT_DIR}"
tar -xzf "$TARBALL" -C "$OUT_DIR"
echo "[done] extracted to: ${OUT_DIR}"

