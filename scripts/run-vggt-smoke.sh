#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/run-vggt-smoke.sh REFERENCE_IMAGE OUTPUT_DIR

Runs the open RefWorld-0 smoke path inside the pinned VGGT CUDA container:
  1. VGGT source camera/depth/raw confidence
  2. source-only 3DGS PLY + renderer-ready source camera
  3. warp-only near-view neighborhood

Environment:
  VGGT_ROOT=/opt/vggt     pinned VGGT checkout inside the reference container
  RUN_TESTS=0|1           run focused CPU tests before inference (default: 1)
  ROTATION_ONLY=0|1       omit depth-ratio translations (default: 0)

Prefer an OUTPUT_DIR under this checkout, e.g. outputs/smoke/source-01, so the
pinned renderer can serve the resulting PLY without weakening its path sandbox.

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

read -r WIDTH HEIGHT < <(
  python3 - "$REFERENCE" <<'PY'
from PIL import Image
import sys
with Image.open(sys.argv[1]) as image:
    print(image.width, image.height)
PY
)

cat <<EOF

RefWorld open smoke artifacts:
  source geometry:   $SOURCE_DIR/source-geometry.safe.json
  source splat:      $SPLAT_DIR/source-splat.safe.json
  renderer camera:  $SPLAT_DIR/source-camera.json
  warp neighborhood:$WARP_DIR/warp-only.safe.json
EOF

if REL_SPLAT=$(python3 - "$SPLAT_DIR" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys
try:
    print(Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve()).as_posix())
except ValueError:
    raise SystemExit(1)
PY
); then
  cat <<EOF

After the GPU container exits, render the source-only diagnostic from the host checkout:
  cd renderer
  npm install
  npx playwright install chromium
  npm run capture -- \\
    --asset $REL_SPLAT/source-splat.ply \\
    --camera $REL_SPLAT/source-camera.json \\
    --out $REL_SPLAT/source-render.png \\
    --width $WIDTH --height $HEIGHT

The capture CLI resolves these arguments against the repository root even when
invoked from renderer/. Compare $REL_SPLAT/source-render.png against the original
source **before** introducing any repaint model. A mismatch here implicates
camera/depth/splat/renderer plumbing rather than hidden-view synthesis.
EOF
else
  cat <<EOF

The output directory is outside this checkout. The pinned renderer intentionally
refuses to serve external paths. Move/copy the source-splat directory under
$REPO_ROOT before rendering, or rerun with OUTPUT_DIR=outputs/smoke/<name>.
EOF
fi

cat <<'EOF'

A successful smoke run is not a RefWorld method result. It only establishes the
open geometry/export/warp path before repaint or canonical-world fitting.
EOF
