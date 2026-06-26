/**
 * Turn a drop (or the bundled sample) into a validated DWI triple.
 *
 * Impure layer: walks the DataTransfer, reads NIfTI headers + sidecar text, and
 * (only for DICOM) lazily loads dcm2niix. Validation — including the NIfTI
 * volume count vs. bval/bvec direction count — finishes here, BEFORE main.ts
 * displays anything, so an invalid image is never loaded into the viewer.
 */

import { traverseDataTransferItems } from './files'
import type { DwiInput } from './state'
import {
  baseName,
  chooseBestSeries,
  countDirections,
  isBval,
  isBvec,
  isJson,
  isNifti,
} from './validate'

// ponytail: sanity caps so a stray huge folder can't read gigabytes into JS/WASM
// memory before we refuse. Generous — real DWI studies are well under these.
// The byte cap is a backstop only; the real fix for large inputs janking the
// main-thread WASM is a worker (deferred).
const MAX_FILES = 20000
const MAX_BYTES = 2_000_000_000 // 2 GB total across the drop

/** Resolved, fully-validated triple (volume count already cross-checked). */
export type ResolvedInput = Omit<DwiInput, 'source'> & {
  source: 'nifti' | 'dicom'
}

/** Collect dropped files, recursing into folders, with a flat-files fallback. */
export async function collectFiles(dt: DataTransfer): Promise<File[]> {
  const items = dt.items
  let files: File[]
  if (
    items &&
    items.length > 0 &&
    typeof items[0].webkitGetAsEntry === 'function'
  ) {
    try {
      // Cap during the walk so a pathological tree aborts early, not after.
      const walked = await traverseDataTransferItems(items, MAX_FILES)
      files = walked.length > 0 ? walked : Array.from(dt.files)
    } catch (err) {
      // A limit overflow is a real refusal; surface it. Other failures
      // (Safari / older browsers) fall back to the flat file list.
      if (err instanceof Error && /too many files/i.test(err.message)) throw err
      files = Array.from(dt.files)
    }
  } else {
    files = Array.from(dt.files)
  }
  if (files.length > MAX_FILES) {
    throw new Error(
      `Too many files (${files.length}). Drop a single DWI study (limit ${MAX_FILES}).`,
    )
  }
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0)
  if (totalBytes > MAX_BYTES) {
    throw new Error(
      `Input too large (${Math.round(totalBytes / 1e6)} MB, limit ${MAX_BYTES / 1e6} MB).`,
    )
  }
  return files
}

/**
 * Resolve a list of files into a validated DWI triple. NIfTI present → treat as
 * a NIfTI+bval+bvec drop; otherwise convert with dcm2niix and pick the
 * diffusion series. Throws a caller-facing Error on any problem.
 */
export async function resolveInput(files: File[]): Promise<ResolvedInput> {
  if (files.some((f) => isNifti(f.name))) {
    return resolveTriple(files, 'nifti')
  }
  // DICOM path: pull in dcm2niix only now (keeps its worker/glue out of the
  // initial payload). niftiOnly:false keeps the bval/bvec/json sidecars.
  const { runDcm2niix } = await import('../niivue-ext/dcm2niix/index')
  const converted = await runDcm2niix(files, { niftiOnly: false })
  if (converted.length === 0) {
    throw new Error('dcm2niix produced no output — not a valid DICOM set.')
  }
  // The converted NIfTI can exceed the input bytes (DICOM is often compressed);
  // re-check the cap before this feeds display + the main-thread tensor fit.
  const convertedBytes = converted.reduce((sum, f) => sum + f.size, 0)
  if (convertedBytes > MAX_BYTES) {
    throw new Error(
      `Converted DICOM too large (${Math.round(convertedBytes / 1e6)} MB, limit ${MAX_BYTES / 1e6} MB).`,
    )
  }
  return resolveTriple(converted, 'dicom')
}

interface Series {
  nii?: File
  bval?: File
  bvec?: File
  json?: File
}

/** A Series with the three required files present — the type-guarded result of
 *  filtering the candidate groups, so callers don't need non-null assertions. */
type CompleteSeries = Series & Required<Pick<Series, 'nii' | 'bval' | 'bvec'>>

async function resolveTriple(
  files: File[],
  source: 'nifti' | 'dicom',
): Promise<ResolvedInput> {
  const groups = new Map<string, Series>()
  for (const f of files) {
    // Case-insensitive so DWI.nii.gz groups with dwi.bval / dwi.bvec.
    const key = baseName(f.name).toLowerCase()
    const g = groups.get(key) ?? {}
    if (isNifti(f.name)) g.nii = f
    else if (isBval(f.name)) g.bval = f
    else if (isBvec(f.name)) g.bvec = f
    else if (isJson(f.name)) g.json = f
    groups.set(key, g)
  }

  const triples = [...groups.values()].filter(
    (g): g is CompleteSeries => !!(g.nii && g.bval && g.bvec),
  )
  if (triples.length === 0) {
    if (source === 'dicom') {
      throw new Error(
        'No diffusion series found — the DICOMs emitted no bval/bvec (not a diffusion acquisition).',
      )
    }
    const has = (p: (n: string) => boolean) => files.some((f) => p(f.name))
    if (has(isNifti) && has(isBval) && has(isBvec)) {
      throw new Error(
        'Found a NIfTI, a bval and a bvec, but their names do not share a basename. Match them like dwi.nii.gz / dwi.bval / dwi.bvec.',
      )
    }
    const missing = [
      !has(isNifti) && 'NIfTI',
      !has(isBval) && 'bval',
      !has(isBvec) && 'bvec',
    ].filter(Boolean)
    throw new Error(
      `A DWI needs all three of NIfTI + bval + bvec. Missing: ${missing.join(', ')}.`,
    )
  }

  // Count directions (bval/bvec) and NIfTI volumes for every candidate; skip any
  // that fail to parse, remembering the error in case none survive.
  const counted: { g: CompleteSeries; directions: number; volumes: number }[] =
    []
  let lastError: Error | null = null
  for (const g of triples) {
    try {
      const directions = countDirections(
        await g.bval.text(),
        await g.bvec.text(),
      )
      const volumes = await niftiVolumeCount(g.nii)
      counted.push({ g, directions, volumes })
    } catch (err) {
      lastError = err as Error
    }
  }
  if (counted.length === 0) {
    throw lastError ?? new Error('No readable diffusion series found.')
  }

  const idx = chooseBestSeries(
    counted.map((c) => ({ directions: c.directions, volumes: c.volumes })),
  )
  if (idx < 0) {
    // Triples exist but none have matching counts — report the largest.
    const c = counted.reduce((a, b) => (b.directions > a.directions ? b : a))
    throw new Error(
      `Volume mismatch: ${c.g.nii.name} has ${c.volumes} volume(s) but its bval/bvec list ${c.directions} direction(s). All must match.`,
    )
  }

  const best = counted[idx]
  return {
    nifti: best.g.nii,
    bval: best.g.bval,
    bvec: best.g.bvec,
    json: best.g.json,
    directions: best.directions,
    source,
  }
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
