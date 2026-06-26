/**
 * Assemble the WebGPU tracker's inputs from the app's loaded data: read the DWI
 * and FA NIfTIs into the exact memory layout the WGSL kernels index, and wire
 * the Stage 3b host matrices (sphere SH sampling, OPDT H/R/delta_b/delta_q,
 * b0s_mask) into a `TrackingInputs`. Also seeds from the FA mask.
 *
 * Layouts (from tracking_helpers.wgsl):
 *   dataf      x*dimy*dimz*dimt + y*dimz*dimt + z*dimt + t   (t fastest)
 *   metric_map x*dimy*dimz       + y*dimz       + z          (z fastest)
 * NIfTI memory is x-fastest, so both are reordered on load.
 */

import { gradientTable } from './gradients'
import { readDataf, readMetric } from './nifti-read'
import { opdtMatrices } from './opdt'
import { realShDescoteaux } from './sh'
import { type Sphere, sphereThetaPhi } from './sphere'
import type { TrackingInputs } from './tracker'

// --- gzip-aware whole-file decompression (the NIfTI reader lives in nifti-read) ---

const tooLargeError = (bytes: number, maxBytes: number): Error =>
  new Error(
    `the DWI decompresses to ~${(bytes / 1e9).toFixed(1)} GB, over this GPU's ` +
      `${(maxBytes / 1e9).toFixed(1)} GB buffer limit — the DWI is too large for this GPU.`,
  )

/** A gzip member's trailer stores the uncompressed size mod 2³² (ISIZE, little-
 *  endian). Cheap to peek so we can reject an implausibly large input BEFORE
 *  decompressing, rather than OOMing mid-inflate. (Exact only below 4 GiB and for
 *  single-member gzip — fine as a guard, not a precise value.) */
async function gzipUncompressedSize(file: File): Promise<number> {
  const tail = new Uint8Array(await file.slice(file.size - 4).arrayBuffer())
  return (tail[0] | (tail[1] << 8) | (tail[2] << 16) | (tail[3] << 24)) >>> 0
}

async function gunzipAll(
  file: File,
  maxBytes = Number.POSITIVE_INFINITY,
): Promise<Uint8Array> {
  const sig = new Uint8Array(await file.slice(0, 2).arrayBuffer())
  if (!(sig[0] === 0x1f && sig[1] === 0x8b)) {
    if (file.size > maxBytes) throw tooLargeError(file.size, maxBytes)
    return new Uint8Array(await file.arrayBuffer())
  }
  // Peek the gzip trailer; reject up front if the decompressed image clearly
  // can't fit (only meaningful when it exceeds the budget — ISIZE wraps ≥ 4 GiB).
  const isize = await gzipUncompressedSize(file)
  if (isize > maxBytes) throw tooLargeError(isize, maxBytes)
  const reader = file
    .stream()
    .pipeThrough(new DecompressionStream('gzip'))
    .getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    total += value.length
    // Runtime backstop in case ISIZE wrapped (≥ 4 GiB) and slipped past the peek.
    if (total > maxBytes) {
      reader.cancel()
      throw tooLargeError(total, maxBytes)
    }
  }
  const out = new Uint8Array(total)
  let o = 0
  for (const c of chunks) {
    out.set(c, o)
    o += c.length
  }
  return out
}

// --- SH order: largest even order whose coeff count fits the direction count ---
// (nCoeff > nDir makes hat(B)=I so the residual
// bootstrap is a no-op; order 4 = 15 coeffs fits the 20-dir sample.)
const nCoeffForOrder = (order: number): number =>
  ((order + 1) * (order + 2)) / 2

export function chooseShOrder(nDir: number): number {
  if (nDir < nCoeffForOrder(2)) {
    throw new Error(
      `too few diffusion directions to fit an ODF (have ${nDir}, need ≥ ${nCoeffForOrder(2)}).`,
    )
  }
  let best = 2
  for (const o of [2, 4, 6, 8]) if (nCoeffForOrder(o) <= nDir) best = o
  return best
}

const f32 = (a: ArrayLike<number>): Float32Array => Float32Array.from(a)

/** Inputs for the GPU tracker plus the geometry needed to write a TRX. */
export interface AssembledInputs {
  inputs: TrackingInputs
  /** Voxel→world (RASMM) 4x4 row-major, for the TRX VOXEL_TO_RASMM. */
  voxelToRasmm: number[][]
  dims3: [number, number, number]
}

/** Build the full TrackingInputs for the Boot/OPDT tracker, plus TRX geometry.
 *  `maxBufferBytes` (the GPU storage-buffer limit) lets the DWI read reject a
 *  too-large volume up front rather than after a doomed multi-GB allocation. */
export async function assembleTrackingInputs(
  dwiFile: File,
  faFile: File,
  bvalText: string,
  bvecText: string,
  sphere: Sphere,
  maxBufferBytes = Number.POSITIVE_INFINITY,
): Promise<AssembledInputs> {
  const gt = gradientTable(bvalText, bvecText)
  const nDir = gt.dwiTheta.length
  const shOrder = chooseShOrder(nDir)

  const opdt = opdtMatrices(gt.dwiTheta, gt.dwiPhi, shOrder)
  const { theta, phi } = sphereThetaPhi(sphere)
  const sampling = realShDescoteaux(theta, phi, shOrder).B // nVerts × nCoeff

  // Read DWI then FA sequentially so their decompressed byte blobs don't pile up;
  // each gunzip result is only referenced for the duration of its read.
  const dwi = readDataf(
    await gunzipAll(dwiFile, maxBufferBytes),
    maxBufferBytes,
  )
  if (dwi.nt !== gt.b0sMask.length) {
    throw new Error(
      `tracking: DWI has ${dwi.nt} volumes but bval lists ${gt.b0sMask.length}.`,
    )
  }
  const fa = readMetric(await gunzipAll(faFile))
  if (fa.nx !== dwi.nx || fa.ny !== dwi.ny || fa.nz !== dwi.nz) {
    throw new Error(
      'tracking: FA map and DWI have different spatial dimensions.',
    )
  }

  return {
    inputs: {
      dataf: dwi.dataf,
      dims: [dwi.nx, dwi.ny, dwi.nz, dwi.nt],
      metricMap: fa.metric,
      sphereVertices: sphere.vertices,
      sphereEdges: sphere.edges,
      H: f32(opdt.H),
      R: f32(opdt.R),
      deltaB: f32(opdt.deltaB),
      deltaQ: f32(opdt.deltaQ),
      samplingMatrix: f32(sampling),
      b0sMask: Int32Array.from(gt.b0sMask, (b) => (b ? 1 : 0)),
      samplmNr: sphere.nVerts,
      nedges: sphere.nEdges,
      deltaNr: opdt.nCoeff,
      modelType: 0, // OPDT
    },
    voxelToRasmm: dwi.affine,
    dims3: [dwi.nx, dwi.ny, dwi.nz],
  }
}

/**
 * Seeds (VOX coords, flat x,y,z) from voxels whose metric (FA) ≥ `threshold`.
 * `perAxis`³ seeds per voxel on a regular sub-grid (deterministic — no RNG to
 * match), capped at `maxSeeds`. metric_map is in (x,y,z) z-fastest order.
 */
export function seedsFromMask(
  metricMap: Float32Array,
  dims: [number, number, number, number],
  threshold: number,
  perAxis = 1,
  maxSeeds = 100000,
): Float32Array {
  const [nx, ny, nz] = dims
  const offs: number[] = []
  for (let i = 0; i < perAxis; i++) offs.push((i + 0.5) / perAxis - 0.5) // voxel-centred sub-grid
  const seeds: number[] = []
  outer: for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        if (metricMap[(x * ny + y) * nz + z] < threshold) continue
        for (const dx of offs)
          for (const dy of offs)
            for (const dz of offs) {
              seeds.push(x + dx, y + dy, z + dz)
              if (seeds.length / 3 >= maxSeeds) break outer
            }
      }
    }
  }
  return Float32Array.from(seeds)
}
