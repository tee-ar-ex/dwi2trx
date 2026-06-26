/**
 * Golden tests for the tractography port (Stage 3), checked against DIPY
 * reference dumps in the committed goldens.json (from DIPY).
 * Run: node --experimental-strip-types src/dwi2trx/tracking/tracking.test.ts
 *
 * 3a: sphere asset + gradient table.
 */

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { gradientTable } from './gradients.ts'
import { invert, matmul } from './linalg.ts'
import { opdtMatrices } from './opdt.ts'
import { realShDescoteaux } from './sh.ts'
import { parseSphere, sphereThetaPhi } from './sphere.ts'

const g = JSON.parse(readFileSync('src/dwi2trx/tracking/goldens.json', 'utf8'))

/** Max abs diff between a typed array and a golden flat array. */
function maxDiff(a: ArrayLike<number>, b: number[]): number {
  assert.equal(a.length, b.length, `length ${a.length} vs ${b.length}`)
  let m = 0
  for (let i = 0; i < b.length; i++) {
    const d = Math.abs(a[i] - b[i])
    if (d > m) m = d
  }
  return m
}

/** Angular max diff (wraps at ±π). */
function maxAngleDiff(a: ArrayLike<number>, b: number[]): number {
  assert.equal(a.length, b.length, `length ${a.length} vs ${b.length}`)
  let m = 0
  for (let i = 0; i < b.length; i++) {
    let d = Math.abs(a[i] - b[i]) % (2 * Math.PI)
    if (d > Math.PI) d = 2 * Math.PI - d
    if (d > m) m = d
  }
  return m
}

const close = (name: string, d: number, tol: number) => {
  assert.ok(d <= tol, `${name}: max diff ${d.toExponential(2)} > ${tol}`)
}

// --- sphere asset ---
const bin = readFileSync('public/sphere.bin')
const sphere = parseSphere(
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
)
assert.equal(sphere.nVerts, g.sphere.vertices.shape[0], 'sphere vertex count')
assert.equal(sphere.nEdges, g.sphere.edges.shape[0], 'sphere edge count')
close('sphere vertices', maxDiff(sphere.vertices, g.sphere.vertices.data), 1e-5)
assert.deepEqual(
  Array.from(sphere.edges),
  g.sphere.edges.data,
  'sphere edges (exact)',
)

// cart2sphere on the sphere vertices reproduces DIPY's theta/phi (float32
// vertices → ~1e-6 deviation).
const { theta, phi } = sphereThetaPhi(sphere)
close('sphere theta', maxDiff(theta, g.sphere.theta.data), 1e-4)
close('sphere phi', maxAngleDiff(phi, g.sphere.phi.data), 1e-4)

// --- gradient table (full double precision from the bval/bvec text) ---
const gt = gradientTable(
  readFileSync('public/dwi.bval', 'utf8'),
  readFileSync('public/dwi.bvec', 'utf8'),
)
assert.deepEqual(gt.b0sMask, g.gtab.b0s_mask, 'b0 mask')
close('dwi theta', maxDiff(gt.dwiTheta, g.gtab.dwi_theta.data), 1e-9)
close('dwi phi', maxAngleDiff(gt.dwiPhi, g.gtab.dwi_phi.data), 1e-9)

console.log(
  `tracking.test.ts 3a: sphere (${sphere.nVerts} verts, ${sphere.nEdges} edges) + gtab (${gt.b0sMask.filter((b) => !b).length} dwi dirs) match DIPY ✓`,
)

// --- 3b: real SH basis (sampling_matrix at sphere directions, B at DWI dirs) ---
const sm = realShDescoteaux(
  g.sphere.theta.data,
  g.sphere.phi.data,
  g.sh_order_max,
)
close('sampling_matrix', maxDiff(sm.B, g.sampling_matrix.data), 1e-6)
const bDwi = realShDescoteaux(
  g.gtab.dwi_theta.data,
  g.gtab.dwi_phi.data,
  g.sh_order_max,
)
close('B_dwi', maxDiff(bDwi.B, g.B_dwi.data), 1e-9)

console.log(
  `tracking.test.ts 3b: real SH basis matches DIPY — sampling_matrix ${g.sampling_matrix.shape.join('×')}, B ${g.B_dwi.shape.join('×')} ✓`,
)

// --- linalg: invert round-trip (M · M⁻¹ = I) ---
{
  const M = Float64Array.from([4, 7, 2, 6]) // 2×2, det 10
  const inv = invert(M, 2)
  const prod = matmul(M, 2, 2, inv, 2)
  close('invert 2×2', maxDiff(prod, [1, 0, 0, 1]), 1e-12)
  const M3 = Float64Array.from([2, 1, 1, 1, 3, 2, 1, 0, 0]) // det -1
  const p3 = matmul(M3, 3, 3, invert(M3, 3), 3)
  close('invert 3×3', maxDiff(p3, [1, 0, 0, 0, 1, 0, 0, 0, 1]), 1e-12)
}

// --- 3b: OPDT fit matrices (delta_b, delta_q) + bootstrap matrices (H, R) ---
const opdt = opdtMatrices(
  g.gtab.dwi_theta.data,
  g.gtab.dwi_phi.data,
  g.sh_order_max,
)
close('delta_b', maxDiff(opdt.deltaB, g.delta_b.data), 1e-6)
close('delta_q', maxDiff(opdt.deltaQ, g.delta_q.data), 1e-6)
close('H (hat)', maxDiff(opdt.H, g.H.data), 1e-6)
close('R (lcr)', maxDiff(opdt.R, g.R.data), 1e-6)

console.log(
  `tracking.test.ts 3b: OPDT matrices match DIPY — delta_b/delta_q ${g.delta_b.shape.join('×')}, H/R ${g.H.shape.join('×')} ✓`,
)
