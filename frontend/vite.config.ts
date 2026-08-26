import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    proxy: {
      // Phase 14 wires the real deployment; dev hits the local FastAPI.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // read-only artifact images (EDA / SHAP galleries)
      '/static': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
