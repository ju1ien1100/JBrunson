import { defineConfig } from "vite";

export default defineConfig({
  // Serve index.html from web/ root
  root: ".",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: {
      input: "index.html",
    },
  },
  server: {
    port: 5173,
    // Allow cross-origin WebSocket to localhost server
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
