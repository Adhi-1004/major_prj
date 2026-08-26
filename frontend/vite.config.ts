import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The dev server proxies the four API paths to the Python service, so the
 * browser only ever talks to one origin and cross-origin rules never enter the
 * picture. In production the same-origin property is real rather than
 * simulated: the API serves the built assets itself.
 *
 * The target is `127.0.0.1`, not `localhost`, deliberately. Node 18+ resolves
 * `localhost` to `::1` (IPv6) first, while uvicorn's default `--host 0.0.0.0`
 * binds IPv4 only -- so `localhost` yields ECONNREFUSED against a server that
 * is plainly running.
 */
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/extract": API,
      "/jobs": API,
      "/frames": API,
      "/health": API,
    },
  },
  build: {
    // Read by the FastAPI static mount; see api/app.py::_mount_frontend.
    outDir: "dist",
    sourcemap: true,
  },
});
