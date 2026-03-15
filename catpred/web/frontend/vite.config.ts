import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: '/static/dist/',
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ready': 'http://localhost:8000',
      '/predict': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
