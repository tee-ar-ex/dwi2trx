// Vite `?raw` imports of WGSL shader sources resolve to their text content.
declare module '*.wgsl?raw' {
  const source: string
  export default source
}
