/**
 * Minimal TRX writer (tee-ar-ex format) — packs tracked streamlines into a TRX
 * zip NiiVue can display and the user can save.
 *
 * TRX layout:
 *   header.json          { DIMENSIONS:[x,y,z], VOXEL_TO_RASMM:4x4 row-major,
 *                          NB_VERTICES, NB_STREAMLINES }
 *   positions.3.float32   flat xyz per vertex, in RASMM WORLD space
 *   offsets.uint32        cumulative start vertex per streamline, length
 *                         NB_STREAMLINES+1 (last entry == NB_VERTICES) — the
 *                         canonical trx-python layout
 * Per the TRX spec, positions are stored in RASMM (world) — NiiVue renders them
 * directly (it does NOT re-apply VOXEL_TO_RASMM). So the GPU's voxel-space
 * streamlines are transformed by the DWI affine here (like the reference's
 * `tractogram.to_world()` before save); VOXEL_TO_RASMM is kept as metadata.
 */

import { zipSync } from 'fflate'

/**
 * @param streamlines one Float32Array per streamline (flat x,y,z in VOXEL space)
 * @param voxelToRasmm 4x4 row-major voxel→world (the DWI sform); applied to each
 *   point so stored positions are RASMM, and kept as the header metadata
 * @param dims [nx, ny, nz]
 */
export function writeTrx(
  streamlines: Float32Array[],
  voxelToRasmm: number[][],
  dims: [number, number, number],
): Uint8Array {
  const nStreams = streamlines.length
  let nVerts = 0
  for (const s of streamlines) nVerts += s.length / 3
  if (nVerts > 0xffffffff)
    throw new Error('TRX: too many vertices for uint32 offsets.')

  const a = voxelToRasmm
  const positions = new Float32Array(nVerts * 3)
  const offsets = new Uint32Array(nStreams + 1)
  let pf = 0 // float cursor
  let pv = 0 // vertex cursor
  for (let i = 0; i < nStreams; i++) {
    offsets[i] = pv
    const s = streamlines[i]
    for (let j = 0; j < s.length; j += 3) {
      const x = s[j],
        y = s[j + 1],
        z = s[j + 2]
      positions[pf] = a[0][0] * x + a[0][1] * y + a[0][2] * z + a[0][3]
      positions[pf + 1] = a[1][0] * x + a[1][1] * y + a[1][2] * z + a[1][3]
      positions[pf + 2] = a[2][0] * x + a[2][1] * y + a[2][2] * z + a[2][3]
      pf += 3
    }
    pv += s.length / 3
  }
  offsets[nStreams] = pv

  const header = {
    DIMENSIONS: dims,
    VOXEL_TO_RASMM: voxelToRasmm,
    NB_VERTICES: nVerts,
    NB_STREAMLINES: nStreams,
  }
  return zipSync(
    {
      'header.json': new TextEncoder().encode(JSON.stringify(header)),
      'positions.3.float32': new Uint8Array(positions.buffer),
      'offsets.uint32': new Uint8Array(offsets.buffer),
    },
    { level: 0 }, // STORE — fast; NiiVue reads stored or deflated entries
  )
}
