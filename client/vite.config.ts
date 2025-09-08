import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig, loadEnv } from "vite"

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '');

  return {
  plugins: [
    react({
      include: "**/*.{jsx,js,ts,tsx}",
    })
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    extensions: ['.mjs', '.js', '.jsx', '.ts', '.tsx', '.json']
  },
  server: {
    host: '0.0.0.0', // Listen on all addresses
    port: 51273, // Custom development port
    strictPort: true,
    hmr: {
      host: 'localhost',
      clientPort: 51273
    },
    watch: {
      usePolling: true, // This can help in containerized environments
      interval: 1000, // Polling interval for better performance in containers
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    emptyOutDir: true,
  },
  preview: {
    port: 51273,
    host: '0.0.0.0'
  }
  };
});