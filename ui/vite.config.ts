import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (`npm run dev`), API calls are proxied to the FastAPI server started by `just dev-api`.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/analyze": api,
      "/tools": api,
      "/health": api,
      "/requests": api,
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
