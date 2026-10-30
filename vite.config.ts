import { defineConfig } from 'vite'

// dwi2trx deploys to the GitHub Pages project subpath
// https://neurolabusc.github.io/dwi2trx/ — base must match so assets resolve.
// Switch base to '/' (and add public/CNAME) if a custom apex domain is set up.
export default defineConfig({
  base: '/dwi2trx/',
  server: {
    open: true,
    port: 8091,
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
  },
  // Vite's dep prebundler moves these into .vite/deps/, where their
  // `new URL(..., import.meta.url)` workers and WASM 404.
  optimizeDeps: {
    exclude: ['@niivue/dcm2niix', '@niivue/niimath', '@brainchop/mindgrab'],
  },
})
