import { defineConfig, loadEnv } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"

export default defineConfig(({ mode }) => {
  // Load .env from the repo root (two levels up) so a single .env file
  // covers both the Python backend and this Vite dev server.
  const env = loadEnv(mode, path.resolve(__dirname, '../..'), '')
  const apiTarget = env.AGENT_API_URL
  if (!apiTarget) {
    throw new Error('AGENT_API_URL is not set in src/clinical-rag-ui/.env (see README).')
  }

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      host: true,
      port: 5173,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
