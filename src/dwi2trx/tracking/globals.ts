/**
 * Tracking constants used host-side, from GPUStreamlines
 * `cuslines/cuda_python/_globals.py`. These must match the values baked into
 * `wgsl/globals.wgsl` (the GPU kernels and host buffer sizing depend on both).
 * Only the constants the TS host actually uses live here; the rest are in the
 * WGSL.
 */

export const MAX_SLINE_LEN = 501
export const REAL_SIZE = 4 // bytes per f32 (WGSL has no f64)
export const BLOCK_Y = 64 / 32 // THR_X_BL / THR_X_SL = seeds per workgroup = 2

export const divUp = (a: number, b: number): number =>
  Math.floor((a + b - 1) / b)
