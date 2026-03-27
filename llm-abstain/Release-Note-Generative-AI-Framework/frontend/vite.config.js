import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/answer': 'http://localhost:3000',
      '/api': 'http://localhost:3000'
    }
  }
});

