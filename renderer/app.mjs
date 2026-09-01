import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const params = new URLSearchParams(location.search);
const assetUrl = params.get("asset");
const cameraUrl = params.get("camera");
const width = Number(params.get("width"));
const height = Number(params.get("height"));
const near = Number(params.get("near") ?? "0.01");
const far = Number(params.get("far") ?? "1000");

if (!assetUrl || !cameraUrl) throw new Error("asset and camera query parameters are required");
if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
  throw new Error("positive integer width and height are required");
}
if (!(near > 0 && far > near)) throw new Error("invalid near/far clip planes");

const cameraSpec = await fetch(cameraUrl).then((response) => {
  if (!response.ok) throw new Error(`camera fetch failed: ${response.status}`);
  return response.json();
});
if (cameraSpec.convention !== "opengl-camera-to-world") {
  throw new Error(`unsupported camera convention: ${cameraSpec.convention}`);
}
if (!Array.isArray(cameraSpec.intrinsics) || cameraSpec.intrinsics.length !== 9) {
  throw new Error("camera intrinsics must contain 9 values");
}
if (!Array.isArray(cameraSpec.extrinsics) || cameraSpec.extrinsics.length !== 16) {
  throw new Error("camera extrinsics must contain 16 values");
}

const [fx, skew, cx, , fy, cy, , , k22] = cameraSpec.intrinsics.map(Number);
if (![fx, skew, cx, fy, cy, k22].every(Number.isFinite) || fx <= 0 || fy <= 0) {
  throw new Error("invalid camera intrinsics");
}
if (Math.abs(skew) > 1e-9 || Math.abs(k22 - 1) > 1e-9) {
  throw new Error("v0 renderer requires zero skew and K[2,2] = 1");
}

const canvas = document.getElementById("frame");
canvas.width = width;
canvas.height = height;
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  alpha: false,
  premultipliedAlpha: true,
  preserveDrawingBuffer: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(1);
renderer.setSize(width, height, false);
renderer.setClearColor(0x000000, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, width / height, near, far);

// Build an OpenGL projection matrix from K where pixel coordinates use
// +u right / +v down and camera space uses +X right / +Y up / -Z forward.
camera.projectionMatrix.set(
  2 * fx / width, 0, 1 - 2 * cx / width, 0,
  0, 2 * fy / height, 2 * cy / height - 1, 0,
  0, 0, (far + near) / (near - far), 2 * far * near / (near - far),
  0, 0, -1, 0,
);
camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

const c2w = new THREE.Matrix4();
c2w.set(...cameraSpec.extrinsics.map(Number));
c2w.decompose(camera.position, camera.quaternion, camera.scale);
camera.updateMatrix();
camera.updateMatrixWorld(true);

const spark = new SparkRenderer({ renderer });
scene.add(spark);
const splat = new SplatMesh({ url: assetUrl });
await splat.initialized;
scene.add(splat);

async function renderFrames(count = 12) {
  if (!Number.isInteger(count) || count < 1 || count > 240) throw new Error("invalid frame count");
  for (let i = 0; i < count; i += 1) {
    renderer.render(scene, camera);
    // Give Spark's sort/update jobs a frame boundary without introducing
    // arbitrary wall-clock sleeps into the benchmark protocol.
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}

window.__refworldRenderFrames = renderFrames;
window.__refworldMeta = {
  renderer: "spark",
  sparkVersion: "2.1.0",
  threeVersion: THREE.REVISION,
  width,
  height,
  near,
  far,
  devicePixelRatio,
  cameraConvention: cameraSpec.convention,
};
await renderFrames(12);
window.__refworldReady = true;
