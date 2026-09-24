import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Forward API calls to the Flask backend (Stage 6, backend/app.py) so the
    // browser never has to deal with cross-origin requests during dev.
    proxy: {
      '/api': 'http://127.0.0.1:5000',
    },
  },
})
