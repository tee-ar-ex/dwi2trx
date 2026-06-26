/** NIfTI reader: correct target-layout reordering, scl scaling, and friendly
 *  errors on malformed/oversized files.
 *  Run: node --experimental-strip-types src/dwi2trx/tracking/nifti-read.test.ts */
import assert from 'node:assert/strict'
import { parseNifti, readDataf, readMetric } from './nifti-read.ts'

const LE = true

/** Build a NIfTI-1 with the given dims/datatype and x-fastest float voxels. */
function nifti(
  nx: number,
  ny: number,
  nz: number,
  nt: number,
  voxels: number[],
  opts: { slope?: number; inter?: number } = {},
): Uint8Array {
  const VOX = 352
  const buf = new ArrayBuffer(VOX + voxels.length * 4)
  const dv = new DataView(buf)
  dv.setInt32(0, 348, LE)
  dv.setInt16(40, nt > 1 ? 4 : 3, LE)
  dv.setInt16(42, nx, LE)
  dv.setInt16(44, ny, LE)
  dv.setInt16(46, nz, LE)
  dv.setInt16(48, nt, LE)
  for (const o of [50, 52, 54]) dv.setInt16(o, 1, LE)
  dv.setInt16(70, 16, LE) // DT_FLOAT32
  dv.setInt16(72, 32, LE)
  dv.setFloat32(108, VOX, LE) // vox_offset
  dv.setFloat32(112, opts.slope ?? 1, LE)
  dv.setFloat32(116, opts.inter ?? 0, LE)
  dv.setUint8(123, 2)
  dv.setUint8(344, 0x6e)
  dv.setUint8(345, 0x2b)
  dv.setUint8(346, 0x31)
  const v = new Float32Array(buf, VOX)
  v.set(voxels)
  return new Uint8Array(buf)
}

// 2×1×1×2 volume, x-fastest source order: t0=[10,11], t1=[20,21].
// dataf layout is t-fastest: index ((x*ny+y)*nz+z)*nt+t.
{
  const bytes = nifti(2, 1, 1, 2, [10, 11, 20, 21])
  const { nx, ny, nz, nt, dataf } = readDataf(bytes)
  assert.deepEqual([nx, ny, nz, nt], [2, 1, 1, 2])
  // voxel x=0 → dataf[0..1] = (t0,t1) = (10,20); x=1 → dataf[2..3] = (11,21)
  assert.deepEqual(Array.from(dataf), [10, 20, 11, 21], 'dataf t-fastest order')
}

// scl_slope/inter applied.
{
  const bytes = nifti(2, 1, 1, 1, [3, 4], { slope: 2, inter: 1 })
  const { metric } = readMetric(bytes)
  assert.deepEqual(Array.from(metric), [7, 9], 'scl_slope*v + inter')
}

// metric_map z-fastest: 1×1×2 → index z within (x,y).
{
  const bytes = nifti(1, 1, 2, 1, [5, 6])
  const { metric } = readMetric(bytes)
  assert.deepEqual(Array.from(metric), [5, 6], 'metric z-fastest order')
}

// Truncated voxel data → friendly error, not a raw RangeError.
{
  const good = nifti(2, 2, 2, 1, [0, 0, 0, 0, 0, 0, 0, 0])
  const truncated = good.slice(0, good.length - 8) // drop 2 voxels
  assert.throws(
    () => readMetric(truncated),
    /truncated/i,
    'truncated NIfTI → friendly error',
  )
}

// Header shorter than 348 bytes → friendly error.
assert.throws(
  () => parseNifti(new Uint8Array(100)),
  /truncated|NIfTI-1/i,
  'short header → friendly error',
)

// readDataf preflight: target array over the GPU limit → actionable error.
{
  const bytes = nifti(2, 2, 2, 2, new Array(16).fill(1))
  assert.throws(
    () => readDataf(bytes, 16), // 16 voxels × 4 = 64 bytes > 16
    /too large for this gpu/i,
    'oversized DWI → preflight error',
  )
}

console.log('nifti-read.test: OK — layout, scaling, truncation, preflight')
