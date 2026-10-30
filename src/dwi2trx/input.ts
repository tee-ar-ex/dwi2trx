/**
 * Turn a drop (or the bundled sample) into a validated DWI triple.
 *
 * Impure layer: walks the DataTransfer, reads NIfTI headers + sidecar text, and
 * (only for DICOM) runs dcm2niix in its worker. Validation — including the NIfTI
 * volume count vs. bval/bvec direction count — finishes here, BEFORE main.ts
 * displays anything, so an invalid image is never loaded into the viewer.
 */

import {
  convert,
  filesFromDataTransfer,
  groupOutputs,
  type OutputFile,
  type Series,
} from '@niivue/dcm2niix'
import { assertInputSize } from './input-limits'
import type { DwiInput } from './state'
import { chooseBestSeries, countDirections, isNifti } from './validate'

/** Collect dropped files, recursing into folders. Call synchronously in the drop
 *  handler (items expire after the event). Browsers without `webkitGetAsEntry`
 *  yield no walk, so fall back to the flat file list. */
export async function collectFiles(dt: DataTransfer): Promise<File[]> {
  const walked = await filesFromDataTransfer(dt.items).catch(() => [])
  return walked.length > 0 ? walked : Array.from(dt.files)
}

/**
 * Resolve files into a validated DWI. NIfTI present → pair the dropped files by
 * stem; otherwise convert with dcm2niix (already paired). Picks the diffusion
 * series whose volume count matches its bval/bvec.
 */
export async function resolveInput(
  files: File[],
  signal?: AbortSignal,
): Promise<DwiInput> {
  assertInputSize(files)
  signal?.throwIfAborted()
  if (files.some((f) => isNifti(f.name))) {
    return pickDwi(await groupOutputs(files as OutputFile[]), 'nifti')
  }
  const { series, files: converted, log } = await convert(files, { signal })
  if (converted.length === 0) {
    throw new Error(
      `dcm2niix produced no output — not a valid DICOM set. ${log.at(-1) ?? ''}`.trim(),
    )
  }
  // DICOM is often compressed, so the converted NIfTI can exceed the input cap.
  assertInputSize(converted)
  return pickDwi(series, 'dicom')
}

async function pickDwi(
  series: Series[],
  source: 'nifti' | 'dicom',
): Promise<DwiInput> {
  const triples = series.filter(
    (s): s is Series & { bval: File; bvec: File } =>
      isNifti(s.image.name) && !!s.bval && !!s.bvec,
  )
  if (triples.length === 0) {
    throw new Error(
      source === 'dicom'
        ? 'No diffusion series found — the DICOMs emitted no bval/bvec.'
        : 'A DWI needs a NIfTI + bval + bvec sharing a basename, like dwi.nii.gz / dwi.bval / dwi.bvec.',
    )
  }

  const counted = await Promise.all(
    triples.map(async (s) => ({
      s,
      directions: countDirections(await s.bval.text(), await s.bvec.text()),
      volumes: await niftiVolumeCount(s.image),
    })),
  )
  const idx = chooseBestSeries(counted)
  if (idx < 0) {
    // None have matching counts — report the largest.
    const c = counted.reduce((a, b) => (b.directions > a.directions ? b : a))
    throw new Error(
      `Volume mismatch: ${c.s.image.name} has ${c.volumes} volume(s) but its bval/bvec list ${c.directions} direction(s). All must match.`,
    )
  }

  const { s, directions } = counted[idx]
  return { nifti: s.image, bval: s.bval, bvec: s.bvec, directions, source }
}

/**
 * Read a NIfTI-1 header's 4D volume count without loading the whole image — so
 * we validate before displaying. Handles gzip (`.nii.gz`) by streaming just the
 * 352-byte header, and either endianness. NIfTI-2 is rejected (dcm2niix and
 * typical DWI emit NIfTI-1).
 */
async function niftiVolumeCount(file: File): Promise<number> {
  const buf = await readHeaderBytes(file, 352)
  if (buf.length < 348) {
    throw new Error(`${file.name}: too small to be a NIfTI volume.`)
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  if (dv.getInt32(0, true) === 540 || dv.getInt32(0, false) === 540) {
    throw new Error(`${file.name}: NIfTI-2 is not supported yet.`)
  }
  // NIfTI-1 magic "n+1\0" at offset 344 — reject look-alikes before trusting dims.
  if (
    buf[344] !== 0x6e ||
    buf[345] !== 0x2b ||
    buf[346] !== 0x31 ||
    buf[347] !== 0x00
  ) {
    throw new Error(`${file.name}: not a NIfTI-1 volume (bad magic).`)
  }
  let le = true
  let ndim = dv.getInt16(40, true)
  if (ndim < 1 || ndim > 7) {
    le = false
    ndim = dv.getInt16(40, false)
  }
  if (ndim < 1 || ndim > 7) {
    throw new Error(`${file.name}: not a recognizable NIfTI-1 volume.`)
  }
  const volumes = ndim >= 4 ? dv.getInt16(48, le) : 1
  if (volumes < 1) {
    throw new Error(
      `${file.name}: unreadable volume count (dim[4]=${volumes}).`,
    )
  }
  return volumes
}

/** First `n` bytes of a file, transparently gunzipping a `.nii.gz`. */
async function readHeaderBytes(file: File, n: number): Promise<Uint8Array> {
  const sig = new Uint8Array(await file.slice(0, 2).arrayBuffer())
  const gzipped = sig[0] === 0x1f && sig[1] === 0x8b
  if (!gzipped) {
    return new Uint8Array(await file.slice(0, n).arrayBuffer())
  }
  const reader = file
    .stream()
    .pipeThrough(new DecompressionStream('gzip'))
    .getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    while (total < n) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      total += value.length
    }
  } finally {
    await reader.cancel()
  }
  const out = new Uint8Array(total)
  let offset = 0
  for (const c of chunks) {
    out.set(c, offset)
    offset += c.length
  }
  return out
}
