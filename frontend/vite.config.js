import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API + figure requests to the FastAPI backend during development so the
// frontend can use relative URLs (no CORS juggling, no hardcoded host).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/figures": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
