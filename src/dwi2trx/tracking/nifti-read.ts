/**
 * Minimal NIfTI-1 image reader for the tracker — gzip handled by the caller.
 * Reads voxels straight into the WGSL kernel's target memory layout, sampling
 * x-fastest source voxels directly from the decompressed bytes (no intermediate
 * full-volume copy). Pure (only `readAffine` from nifti-geometry), so it
 * unit-tests in plain node (see nifti-read.test.ts).
 *
 * Layouts (from tracking_helpers.wgsl):
 *   dataf      x*ny*nz*nt + y*nz*nt + z*nt + t   (t fastest)
 *   metric_map x*ny*nz    + y*nz    + z          (z fastest)
 */

import { readAffine } from '../../lib/nifti-geometry.ts'

interface NiftiHeader {
  nx: number
  ny: number
  nz: number
  nt: number
  /** Voxel→world (RASMM) 4x4 row-major, from the sform (or qform/pixdim). */
  affine: number[][]
  /** Read+scale one voxel by its x-fastest linear index. */
  sample: (xFastIndex: number) => number
}

/** Bytes per voxel for each supported NIfTI-1 datatype code. */
const DT_BYTES: Record<number, number> = {
  2: 1,
  4: 2,
  8: 4,
  16: 4,
  64: 8,
  256: 1,
  512: 2,
  768: 4,
}

export function parseNifti(bytes: Uint8Array): NiftiHeader {
  if (bytes.byteLength < 348) {
    throw new Error('tracking: NIfTI file is truncated (header incomplete).')
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (dv.getInt32(0, true) !== 348) {
    throw new Error(
      'tracking: only NIfTI-1 (sizeof_hdr 348, little-endian) is supported.',
    )
  }
  const dim0 = dv.getInt16(40, true)
  const nx = dv.getInt16(42, true)
  const ny = dv.getInt16(44, true)
  const nz = dv.getInt16(46, true)
  const nt = dim0 >= 4 ? Math.max(1, dv.getInt16(48, true)) : 1
  if (nx <= 0 || ny <= 0 || nz <= 0 || nt <= 0) {
    throw new Error(
      `tracking: NIfTI has invalid dimensions ${nx}×${ny}×${nz}×${nt}.`,
    )
  }
  const datatype = dv.getInt16(70, true)
  let slope = dv.getFloat32(112, true)
  const inter = dv.getFloat32(116, true)
  if (slope === 0) slope = 1
  const voxOffset = Math.round(dv.getFloat32(108, true)) || 352

  const size = DT_BYTES[datatype]
  if (!size) {
    throw new Error(`tracking: unsupported NIfTI datatype ${datatype}.`)
  }
  // Verify the declared voxels actually fit in the (decompressed) buffer, so a
  // truncated/corrupt file fails with a clear message rather than a raw
  // out-of-bounds RangeError deep in the read loop.
  const need = voxOffset + nx * ny * nz * nt * size
  if (need > bytes.byteLength) {
    throw new Error(
      `tracking: NIfTI is truncated — needs ${need.toLocaleString()} bytes but the ` +
        `file has ${bytes.byteLength.toLocaleString()} (dims ${nx}×${ny}×${nz}×${nt}).`,
    )
  }

  const d = new DataView(bytes.buffer, bytes.byteOffset + voxOffset)
  const getters: Record<number, (o: number) => number> = {
    2: (o) => d.getUint8(o),
    4: (o) => d.getInt16(o, true),
    8: (o) => d.getInt32(o, true),
    16: (o) => d.getFloat32(o, true),
    64: (o) => d.getFloat64(o, true),
    256: (o) => d.getInt8(o),
    512: (o) => d.getUint16(o, true),
    768: (o) => d.getUint32(o, true),
  }
  const get = getters[datatype]
  return {
    nx,
    ny,
    nz,
    nt,
    affine: readAffine(dv),
    sample: (i) => get(i * size) * slope + inter,
  }
}

/**
 * Read the 4D DWI straight into the tracker's dataf layout (t fastest). When
 * `maxBytes` is given, reject BEFORE allocating if the target array would exceed
 * the GPU's buffer limit (it could never upload anyway) — an actionable error
 * instead of a doomed multi-GB allocation.
 */
export function readDataf(
  bytes: Uint8Array,
  maxBytes = Number.POSITIVE_INFINITY,
): {
  nx: number
  ny: number
  nz: number
  nt: number
  affine: number[][]
  dataf: Float32Array
} {
  const { nx, ny, nz, nt, affine, sample } = parseNifti(bytes)
  const want = nx * ny * nz * nt * 4
  if (want > maxBytes) {
    throw new Error(
      `the DWI needs a ${(want / 1e9).toFixed(1)} GB GPU buffer, over this GPU's ` +
        `${(maxBytes / 1e9).toFixed(1)} GB limit — the DWI is too large for this GPU.`,
    )
  }
  const dataf = new Float32Array(nx * ny * nz * nt)
  const sliceXY = nx * ny
  const sliceXYZ = sliceXY * nz
  for (let t = 0; t < nt; t++) {
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          dataf[((x * ny + y) * nz + z) * nt + t] = sample(
            x + y * nx + z * sliceXY + t * sliceXYZ,
          )
        }
      }
    }
  }
  return { nx, ny, nz, nt, affine, dataf }
}

/** Read the 3D FA straight into metric_map layout (z fastest). */
export function readMetric(bytes: Uint8Array): {
  nx: number
  ny: number
  nz: number
  metric: Float32Array
} {
  const { nx, ny, nz, sample } = parseNifti(bytes)
  const metric = new Float32Array(nx * ny * nz)
  const sliceXY = nx * ny
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        metric[(x * ny + y) * nz + z] = sample(x + y * nx + z * sliceXY)
      }
    }
  }
  return { nx, ny, nz, metric }
}
