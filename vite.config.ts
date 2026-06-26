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
  // Vite's dep prebundler trips on the dynamic-import WASM workers used by
  // @niivue/niimath and @niivue/dcm2niix. Excluding them keeps the worker
  // scripts as standalone modules so their runtime URLs resolve.
  optimizeDeps: {
    exclude: ['@niivue/dcm2niix', '@niivue/niimath'],
  },
})
