/**
 * Gradient table from bval/bvec text — the subset of DIPY's `gradient_table`
 * the tractography port needs: the b0 mask and the DWI gradient directions as
 * (theta, phi). Matches DIPY for the bundled sample (golden-tested).
 */

import { parseNumbers } from '../validate.ts'
import { cart2sphere } from './spherical.ts'

// DIPY's default b0 threshold. For single-shell data (b0 + one high shell) any
// value between 0 and the shell b-value gives the same mask.
const B0_THRESHOLD = 50

export interface GradientTable {
  bvals: Float64Array
  bvecs: Float64Array // n * 3 (x,y,z row-major)
  b0sMask: boolean[]
  /** (theta, phi) of the non-b0 gradient directions, in volume order. */
  dwiTheta: Float64Array
  dwiPhi: Float64Array
}

export function gradientTable(
  bvalText: string,
  bvecText: string,
): GradientTable {
  const bvals = Float64Array.from(parseNumbers(bvalText))
  if (bvals.length === 0) throw new Error('bval is empty.')
  const rows = bvecText
    .trim()
    .split(/\r?\n/)
    .map((r) => r.trim())
    .filter((r) => r.length > 0)
    .map(parseNumbers)
  if (rows.length !== 3) {
    throw new Error(`bvec must have 3 rows, found ${rows.length}.`)
  }
  const n = bvals.length
  // Defensive: this feeds NaN-silently-propagating SH math, so reject bad input
  // even though callers normally validate first.
  if (!bvals.every(Number.isFinite))
    throw new Error('bval has non-finite values.')
  for (const row of rows) {
    if (row.length !== n) {
      throw new Error(`bvec row has ${row.length} values but bval lists ${n}.`)
    }
    if (!row.every(Number.isFinite)) {
      throw new Error('bvec has non-finite values.')
    }
  }
  const bvecs = new Float64Array(n * 3)
  for (let i = 0; i < n; i++) {
    bvecs[i * 3] = rows[0][i]
    bvecs[i * 3 + 1] = rows[1][i]
    bvecs[i * 3 + 2] = rows[2][i]
  }

  const b0sMask = Array.from(bvals, (b) => b <= B0_THRESHOLD)
  const theta: number[] = []
  const phi: number[] = []
  for (let i = 0; i < n; i++) {
    if (b0sMask[i]) continue
    const s = cart2sphere(bvecs[i * 3], bvecs[i * 3 + 1], bvecs[i * 3 + 2])
    theta.push(s.theta)
    phi.push(s.phi)
  }

  return {
    bvals,
    bvecs,
    b0sMask,
    dwiTheta: Float64Array.from(theta),
    dwiPhi: Float64Array.from(phi),
  }
}
