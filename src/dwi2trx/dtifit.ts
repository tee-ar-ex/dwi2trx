/**
 * Tensor fit: DWI + bval/bvec → FA + V1 maps, entirely in the browser via the
 * locally-built niimath WASM (`--dtifit`).
 *
 * Mirrors commandline/commandline.txt but runs in native space. The fit is
 * UNMASKED by default: the b0 is T2-weighted, so an intensity (otsu) mask wrongly
 * drops white matter, and unmasked FA matches FSL in-brain (the FA floor hides
 * background noise). An optional mindgrab brain mask (`maskConf`) can be passed
 * for a background-free fit — see fitTensor.
 *
 * `--dtifit` is a CLI mode (multi-input, multi-output), not a chainable
 * operator, so we drive the raw module directly: stage files into the Emscripten
 * FS, callMain, read outputs. Runs on the main thread — fine for typical DWI;
 * move to a worker if a huge volume janks the UI.
 */

import type { NiimathModule } from '@niivue/niimath/niimath.js'
import type { DwiInput, TensorMaps } from './state'

let modulePromise: Promise<NiimathModule> | null = null

async function getModule(): Promise<NiimathModule> {
  // Reset the cache on failure so a later fit can retry — otherwise a one-off
  // load error (network/WASM hiccup) would be cached and poison the session.
  modulePromise ??= import('@niivue/niimath/niimath.js')
    .then((m) => m.default())
    .catch((err) => {
      modulePromise = null
      throw err
    })
  return modulePromise
}

let inFlight = false

/**
 * Extract the first volume (b0) of the DWI as a `.nii.gz` File — the input the
 * mindgrab brain mask runs on. Serialized like fitTensor (shared niimath FS).
 */
export async function cropFirstVolume(input: DwiInput): Promise<File> {
  if (inFlight) throw new Error('niimath is busy.')
  inFlight = true
  try {
    const mod = await getModule()
    try {
      mod.FS_createDataFile(
        '.',
        'dwi.nii.gz',
        await bytes(input.nifti),
        true,
        true,
      )
      run(mod, ['dwi.nii.gz', '-crop', '0', '1', 'b0.nii.gz'], 'extract b0')
      return new File([mod.FS_readFile('b0.nii.gz')], 'b0.nii.gz')
    } finally {
      for (const n of ['dwi.nii.gz', 'b0.nii.gz']) {
        try {
          mod.FS_unlink(n)
        } catch {}
      }
    }
  } finally {
    inFlight = false
  }
}

/**
 * Fit the diffusion tensor for a validated DWI and return the FA + V1 maps as
 * `.nii.gz` Files (ready for `nv.loadVolumes`). Serialized — the niimath module
 * is a single shared FS, so concurrent calls would collide; callers should also
 * disable the trigger UI while a fit runs.
 *
 * `maskConf` (optional): a mindgrab brain mask in *conformed* (256³) space. When
 * given, it is resliced to the native DWI grid (nearest-neighbour, no dilation
 * by default — see the inline comment), then dtifit is run masked — a
 * background-free fit. Without it, the fit is unmasked (the b0 is T2-weighted,
 * so an intensity mask wrongly drops white matter; unmasked FA matches FSL).
 */
export async function fitTensor(
  input: DwiInput,
  maskConf?: File,
): Promise<TensorMaps> {
  if (inFlight) throw new Error('A tensor fit is already running.')
  inFlight = true
  try {
    const mod = await getModule()
    const staged = [
      'dwi.nii.gz',
      'dwi.bval',
      'dwi.bvec',
      'b0.nii.gz',
      'maskconf.nii',
      'mask.nii.gz',
      'dti_FA.nii.gz',
      'dti_MD.nii.gz',
      'dti_L1.nii.gz',
      'dti_L2.nii.gz',
      'dti_L3.nii.gz',
      'dti_V1.nii.gz',
      'dti_V2.nii.gz',
      'dti_V3.nii.gz',
      'dti_S0.nii.gz',
      'dti_MO.nii.gz',
      'dti_tensor.nii.gz',
    ]
    try {
      mod.FS_createDataFile(
        '.',
        'dwi.nii.gz',
        await bytes(input.nifti),
        true,
        true,
      )
      mod.FS_createDataFile(
        '.',
        'dwi.bval',
        await bytes(input.bval),
        true,
        true,
      )
      mod.FS_createDataFile(
        '.',
        'dwi.bvec',
        await bytes(input.bvec),
        true,
        true,
      )

      const dtifitArgs = [
        '--dtifit',
        '-k',
        'dwi',
        '-r',
        'dwi.bvec',
        '-b',
        'dwi.bval',
        '-o',
        'dti',
      ]
      if (maskConf) {
        // Reslice the conformed mindgrab mask onto the native DWI grid (the b0
        // is the reference), nearest-neighbour. NO dilation by default: the FA is
        // very noisy at the scalp, so growing the mask outward pulls that noise
        // into the fit. To re-enable a brainchop-style border, append a
        // morphological close before the output, e.g.
        //   ['maskconf.nii', '-reslice_nn', 'b0.nii.gz', '-close', '1', '2', '0', 'mask.nii.gz']
        // (`-close 1 <border_mm> 0` = binarize at 1, dilate border_mm, erode 0).
        // The b0 is often qform-only (FSL-preprocessed DWI, sform_code=0); niimath
        // now fills a missing sform from the qform on read, so its reslice (which
        // reads the sform matrix) aligns correctly. See vendor/niimath.
        run(mod, ['dwi.nii.gz', '-crop', '0', '1', 'b0.nii.gz'], 'extract b0')
        mod.FS_createDataFile(
          '.',
          'maskconf.nii',
          await bytes(maskConf),
          true,
          true,
        )
        run(
          mod,
          ['maskconf.nii', '-reslice_nn', 'b0.nii.gz', 'mask.nii.gz'],
          'reslice mask',
        )
        dtifitArgs.splice(dtifitArgs.length - 2, 0, '-m', 'mask')
      }
      run(mod, dtifitArgs, 'tensor fit')

      return {
        fa: new File([mod.FS_readFile('dti_FA.nii.gz')], 'dti_FA.nii.gz'),
        v1: new File([mod.FS_readFile('dti_V1.nii.gz')], 'dti_V1.nii.gz'),
      }
    } finally {
      for (const name of staged) {
        try {
          mod.FS_unlink(name)
        } catch {
          // not all dti_* outputs are read; ignore missing.
        }
      }
    }
  } finally {
    inFlight = false
  }
}

function run(mod: NiimathModule, args: string[], step: string): void {
  const code = mod.callMain(args)
  if (code !== 0) {
    throw new Error(`niimath ${step} failed (exit ${code}).`)
  }
}

async function bytes(file: File): Promise<Uint8Array> {
  return new Uint8Array(await file.arrayBuffer())
}
