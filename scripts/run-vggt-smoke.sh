#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-vggt-smoke.sh REFERENCE_IMAGE OUTPUT_DIR

Runs the open RefWorld-0 smoke path inside the pinned VGGT CUDA container:
  1. VGGT source camera/depth/raw confidence
  2. source-only 3DGS PLY diagnostic
  3. warp-only near-view neighborhood

Environment:
  VGGT_ROOT=/opt/vggt     pinned VGGT checkout inside the reference container
  RUN_TESTS=0|1           run focused CPU tests before inference (default: 1)
  ROTATION_ONLY=0|1       omit depth-ratio translations (default: 0)

This script does not call Marble or any proprietary world-generation API.
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

REFERENCE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
OUTPUT=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$2")
VGGT_ROOT=${VGGT_ROOT:-/opt/vggt}
RUN_TESTS=${RUN_TESTS:-1}
ROTATION_ONLY=${ROTATION_ONLY:-0}

if [[ ! -f "$REFERENCE" ]]; then
  echo "reference image not found: $REFERENCE" >&2
  exit 2
fi
if [[ ! -d "$VGGT_ROOT/.git" ]]; then
  echo "VGGT checkout not found at $VGGT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT"

if [[ "$RUN_TESTS" == "1" ]]; then
  python3 -m pytest -q \
    tests/test_vggt_source.py \
    tests/test_source_geometry.py \
    tests/test_pinhole_warp.py \
    tests/test_proposals.py \
    tests/test_splats.py \
    tests/test_schemas.py
elif [[ "$RUN_TESTS" != "0" ]]; then
  echo "RUN_TESTS must be 0 or 1" >&2
  exit 2
fi

SOURCE_DIR="$OUTPUT/source-geometry"
SPLAT_DIR="$OUTPUT/source-splat"
WARP_DIR="$OUTPUT/warp-only"

refworld-vggt-source \
  --vggt-root "$VGGT_ROOT" \
  --reference "$REFERENCE" \
  --output "$SOURCE_DIR" \
  --seed 0

refworld-source-splat \
  --reference "$REFERENCE" \
  --source-geometry "$SOURCE_DIR/source-geometry.safe.json" \
  --output "$SPLAT_DIR"

WARP_ARGS=(
  --reference "$REFERENCE"
  --source-geometry "$SOURCE_DIR/source-geometry.safe.json"
  --output "$WARP_DIR"
)
if [[ "$ROTATION_ONLY" == "1" ]]; then
  WARP_ARGS+=(--rotation-only)
elif [[ "$ROTATION_ONLY" != "0" ]]; then
  echo "ROTATION_ONLY must be 0 or 1" >&2
  exit 2
fi
refworld-warp-only "${WARP_ARGS[@]}"

cat <<EOF

RefWorld open smoke artifacts:
  source geometry: $SOURCE_DIR/source-geometry.safe.json
  source splat:    $SPLAT_DIR/source-splat.safe.json
  warp neighborhood: $WARP_DIR/warp-only.safe.json

Next diagnostic:
  render $SPLAT_DIR/source-splat.ply at the camera recorded in
  $SPLAT_DIR/source-splat.safe.json using the pinned renderer/ harness.

A successful smoke run is not a RefWorld method result. It only establishes the
open geometry/export/warp path before any repaint or canonical-world fitting.
EOF
