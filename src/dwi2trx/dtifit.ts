/**
 * Tensor fit: DWI + bval/bvec → FA + V1 maps via niimath `--dtifit`, in native
 * space. Unmasked by default: the b0 is T2-weighted, so an intensity mask drops
 * white matter, and unmasked FA matches FSL in-brain. An optional mindgrab mask
 * gives a background-free fit.
 *
 * `--dtifit` has several inputs and outputs, which the @niivue/niimath worker
 * chain cannot express, so we drive the raw Emscripten module on the main thread.
 */

import type { NiimathModule } from '@niivue/niimath/niimath.js'
import type { DwiInput, TensorMaps } from './state'

let modulePromise: Promise<NiimathModule> | null = null

function getModule(): Promise<NiimathModule> {
  // Reset on failure so a one-off load error does not poison the session.
  modulePromise ??= import('@niivue/niimath/niimath.js')
    .then((m) => m.default())
    .catch((err) => {
      modulePromise = null
      throw err
    })
  return modulePromise
}

/**
 * Stage `inputs`, run niimath, return `outputs`, and unlink everything named in
 * `inputs`, `outputs` and `scratch`. Bytes are read before staging, so the FS
 * section has no await and concurrent calls cannot interleave.
 */
async function niimath(
  inputs: Record<string, File>,
  args: string[],
  outputs: string[],
  scratch: string[] = [],
): Promise<File[]> {
  const mod = await getModule()
  const staged = await Promise.all(
    Object.entries(inputs).map(
      async ([name, file]) =>
        [name, new Uint8Array(await file.arrayBuffer())] as const,
    ),
  )
  try {
    // splice(0) drops our copies before callMain; MEMFS keeps its own.
    for (const [name, data] of staged.splice(0))
      mod.FS_createDataFile('.', name, data, true, true)
    let code: number
    try {
      code = mod.callMain(args)
    } catch (err) {
      // A throw (not a nonzero exit) means the WASM aborted; it cannot be reused.
      modulePromise = null
      throw err
    }
    if (code !== 0) throw new Error(`niimath failed (exit ${code}).`)
    return outputs.map((n) => new File([mod.FS_readFile(n)], n))
  } finally {
    for (const n of [...Object.keys(inputs), ...outputs, ...scratch]) {
      try {
        mod.FS_unlink(n)
      } catch {} // not every output exists on failure
    }
  }
}

/** First volume (b0) of the DWI, the input mindgrab masks. */
export async function cropFirstVolume(input: DwiInput): Promise<File> {
  const [b0] = await niimath(
    { 'dwi.nii.gz': input.nifti },
    ['dwi.nii.gz', '-crop', '0', '1', 'b0.nii.gz'],
    ['b0.nii.gz'],
  )
  return b0
}

/**
 * Fit the tensor and return FA + V1. `mask` is a binary brain mask already on
 * the DWI grid (as mindgrab returns it). No dilation: scalp FA is noisy, so ask
 * mindgrab for `borderMm` if a looser mask is ever wanted.
 */
export async function fitTensor(
  input: DwiInput,
  mask?: File,
): Promise<TensorMaps> {
  const inputs: Record<string, File> = {
    'dwi.nii.gz': input.nifti,
    'dwi.bval': input.bval,
    'dwi.bvec': input.bvec,
  }
  const args = ['--dtifit', '-k', 'dwi', '-r', 'dwi.bvec', '-b', 'dwi.bval']
  if (mask) {
    inputs['mask.nii.gz'] = mask
    args.push('-m', 'mask')
  }
  args.push('-o', 'dti')
  const unread = ['MD', 'L1', 'L2', 'L3', 'V2', 'V3', 'S0', 'MO', 'tensor']
  const [fa, v1] = await niimath(
    inputs,
    args,
    ['dti_FA.nii.gz', 'dti_V1.nii.gz'],
    unread.map((s) => `dti_${s}.nii.gz`),
  )
  return { fa, v1 }
}
