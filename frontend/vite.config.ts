import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/autocomplete': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/recommend_books': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/mood_recommend': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/wishlist': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/history': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/rate': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/book': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
      '/admin': {
        target: 'http://127.0.0.1:5001',
        changeOrigin: true,
      },
    },
  },
})
