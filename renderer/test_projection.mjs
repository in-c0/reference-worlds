import assert from "node:assert/strict";
import test from "node:test";
import { projectionFromIntrinsics } from "./projection.mjs";

function multiply4(matrix, point) {
  return [0, 1, 2, 3].map((row) =>
    matrix[row * 4] * point[0] +
    matrix[row * 4 + 1] * point[1] +
    matrix[row * 4 + 2] * point[2] +
    matrix[row * 4 + 3] * point[3]
  );
}

function ndc(matrix, point) {
  const clip = multiply4(matrix, [...point, 1]);
  return clip.slice(0, 3).map((value) => value / clip[3]);
}

test("optical axis maps to NDC center for centered principal point", () => {
  const p = projectionFromIntrinsics([100, 0, 100, 0, 100, 50, 0, 0, 1], 200, 100, 0.1, 100);
  assert.deepEqual(ndc(p, [0, 0, -1]).slice(0, 2), [0, 0]);
});

test("positive camera Y maps toward the top of the image", () => {
  const p = projectionFromIntrinsics([100, 0, 100, 0, 100, 50, 0, 0, 1], 200, 100, 0.1, 100);
  const xy = ndc(p, [0, 0.5, -1]).slice(0, 2);
  assert.ok(Math.abs(xy[0]) < 1e-12);
  assert.ok(Math.abs(xy[1] - 1) < 1e-12);
});

test("off-center principal point maps optical axis to corresponding NDC offset", () => {
  const p = projectionFromIntrinsics([100, 0, 120, 0, 100, 40, 0, 0, 1], 200, 100, 0.1, 100);
  const xy = ndc(p, [0, 0, -1]).slice(0, 2);
  assert.ok(Math.abs(xy[0] - 0.2) < 1e-12);
  assert.ok(Math.abs(xy[1] - 0.2) < 1e-12);
});

test("rightward camera-space point maps to positive NDC X", () => {
  const p = projectionFromIntrinsics([100, 0, 100, 0, 100, 50, 0, 0, 1], 200, 100, 0.1, 100);
  const x = ndc(p, [1, 0, -2])[0];
  assert.ok(Math.abs(x - 0.5) < 1e-12);
});
