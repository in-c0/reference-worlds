export function projectionFromIntrinsics(intrinsics, width, height, near = 0.01, far = 1000) {
  if (!Array.isArray(intrinsics) || intrinsics.length !== 9) throw new Error("intrinsics must contain 9 values");
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error("positive integer width and height are required");
  }
  if (!(near > 0 && far > near)) throw new Error("invalid near/far clip planes");
  const [fx, skew, cx, , fy, cy, , , k22] = intrinsics.map(Number);
  if (![fx, skew, cx, fy, cy, k22].every(Number.isFinite) || fx <= 0 || fy <= 0) {
    throw new Error("invalid camera intrinsics");
  }
  if (Math.abs(skew) > 1e-9 || Math.abs(k22 - 1) > 1e-9) {
    throw new Error("v0 renderer requires zero skew and K[2,2] = 1");
  }
  return [
    2 * fx / width, 0, 1 - 2 * cx / width, 0,
    0, 2 * fy / height, 2 * cy / height - 1, 0,
    0, 0, (far + near) / (near - far), 2 * far * near / (near - far),
    0, 0, -1, 0,
  ];
}
