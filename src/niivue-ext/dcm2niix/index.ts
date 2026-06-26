/**
 * @niivue/nv-ext-dcm2niix
 *
 * Browser-side DICOM-to-NIfTI conversion for NiiVue, wrapping the
 * `@niivue/dcm2niix` WebAssembly build of Chris Rorden's dcm2niix.
 *
 * {@link runDcm2niix} converts a `File[]` (from `<input webkitdirectory>` or a
 * folder walk) → NIfTI/bval/bvec `File[]`. The folder-drop walk itself lives in
 * `src/dwi2trx/files.ts` (kept apart so this DICOM WASM stays lazily imported).
 *
 * The underlying `Dcm2niix` class is re-exported for callers that need
 * full control over the command-line flags exposed by dcm2niix.
 *
 * Usage:
 * ```ts
 * import NiiVueGPU from '@niivue/niivue'
 * import { runDcm2niix } from '@niivue/nv-ext-dcm2niix'
 *
 * const nv = new NiiVueGPU()
 * await nv.attachTo('gl1')
 *
 * input.addEventListener('change', async () => {
 *   const niftiFiles = await runDcm2niix(input.files)
 *   await nv.loadVolumes([{ url: niftiFiles[0] }])
 * })
 * ```
 */

import { Dcm2niix } from '@niivue/dcm2niix'

// Re-export so callers can drop down to the raw API when they need flags
// like compression level, BIDS sidecars, etc.
export { Dcm2niix }

/** Options for {@link runDcm2niix}. */
export interface RunDcm2niixOptions {
  /**
   * Filter the result list down to NIfTI outputs (`.nii` and `.nii.gz`).
   * BIDS sidecars and other dcm2niix outputs are dropped. Default: `true`.
   */
  niftiOnly?: boolean
}

/**
 * Convert DICOM files to NIfTI by spinning up a fresh dcm2niix worker,
 * feeding it the files, waiting for the result, then terminating the
 * worker so the WASM heap is released.
 *
 * Each call boots its own worker — fine for one-off conversions; for
 * batch workflows, instantiate `Dcm2niix` once and reuse it (and call
 * `worker?.terminate()` yourself when finished).
 *
 * @param files       FileList from `<input webkitdirectory>` or File[]
 *                    from a drop event or a directory input.
 * @param options     See {@link RunDcm2niixOptions}.
 * @returns           Converted output files (NIfTI by default).
 */
export async function runDcm2niix(
  files: FileList | File[] | null | undefined,
  options: RunDcm2niixOptions = {},
): Promise<File[]> {
  const { niftiOnly = true } = options
  if (!files || files.length === 0) return []

  const dcm2niix = new Dcm2niix()
  try {
    await dcm2niix.init()
    const result = (await dcm2niix.input(files).run()) as File[]
    return niftiOnly
      ? result.filter((f) => /\.nii(\.gz)?$/i.test(f.name))
      : result
  } finally {
    dcm2niix.worker?.terminate()
  }
}
