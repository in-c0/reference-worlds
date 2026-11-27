import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");

function args(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error(`bad argument near ${key ?? "<end>"}`);
    out[key.slice(2)] = value;
  }
  return out;
}

function repoRelative(input) {
  const absolute = path.resolve(repoRoot, input);
  const prefix = repoRoot.endsWith(path.sep) ? repoRoot : `${repoRoot}${path.sep}`;
  if (absolute !== repoRoot && !absolute.startsWith(prefix)) throw new Error(`path escapes repository root: ${input}`);
  return absolute;
}

function contentType(file) {
  const ext = path.extname(file).toLowerCase();
  return {
    ".html": "text/html; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".spz": "application/octet-stream",
    ".wasm": "application/wasm",
  }[ext] ?? "application/octet-stream";
}

async function startServer() {
  const server = http.createServer((req, res) => {
    try {
      const pathname = decodeURIComponent(new URL(req.url, "http://127.0.0.1").pathname);
      const file = repoRelative(`.${pathname}`);
      if (!fs.statSync(file).isFile()) throw new Error("not a file");
      res.writeHead(200, { "Content-Type": contentType(file), "Cache-Control": "no-store" });
      fs.createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return server;
}

const opt = args(process.argv.slice(2));
for (const required of ["asset", "camera", "out", "width", "height"]) {
  if (!opt[required]) throw new Error(`--${required} is required`);
}
const width = Number(opt.width);
const height = Number(opt.height);
if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
  throw new Error("width and height must be positive integers");
}
const asset = repoRelative(opt.asset);
const cameraFile = repoRelative(opt.camera);
const output = path.resolve(repoRoot, opt.out);
if (!fs.statSync(asset).isFile()) throw new Error("asset does not exist");
if (!fs.statSync(cameraFile).isFile()) throw new Error("camera does not exist");
fs.mkdirSync(path.dirname(output), { recursive: true });

const server = await startServer();
const { port } = server.address();
const relativeAsset = path.relative(repoRoot, asset).split(path.sep).join("/");
const relativeCamera = path.relative(repoRoot, cameraFile).split(path.sep).join("/");
const url = new URL(`http://127.0.0.1:${port}/renderer/index.html`);
url.searchParams.set("asset", `/${relativeAsset}`);
url.searchParams.set("camera", `/${relativeCamera}`);
url.searchParams.set("width", String(width));
url.searchParams.set("height", String(height));
if (opt.near) url.searchParams.set("near", opt.near);
if (opt.far) url.searchParams.set("far", opt.far);

let browser;
try {
  browser = await chromium.launch({
    headless: true,
    args: [
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
    ],
  });
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  await page.goto(url.href, { waitUntil: "load", timeout: 120_000 });
  await page.waitForFunction(() => window.__refworldReady === true, null, { timeout: 120_000 });
  await page.evaluate(() => window.__refworldRenderFrames(8));
  await page.locator("canvas#frame").screenshot({ path: output });
  const meta = await page.evaluate(() => window.__refworldMeta);
  const result = {
    output: path.relative(repoRoot, output),
    browser: `Chromium ${browser.version()}`,
    ...meta,
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
