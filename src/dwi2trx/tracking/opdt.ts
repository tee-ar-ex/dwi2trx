/**
 * OPDT bootstrap matrices — a port of DIPY's `OpdtModel._fit_matrix` plus the
 * `hat`/`lcr_matrix` residual-bootstrap matrices (boot_utils.prepare_opdt).
 * Golden-tested against DIPY (delta_b, delta_q, H, R in goldens.json).
 *
 * delta_b/delta_q map DWI signal → OPDT ODF SH coefficients (the model). H/R
 * drive the residual bootstrap used by the probabilistic direction getter.
 *
 * NOTE on the bundled sample: it has 20 DWI directions but order-6 SH has 28
 * coefficients (underdetermined), so the SH fit is exact, H is the identity,
 * and R is zero — the residual bootstrap is a no-op. For meaningful
 * probabilistic tracking the SH order must satisfy nCoeff ≤ nDir (e.g. order 4
 * = 15 coeffs for 20 directions). This port reproduces DIPY exactly either way.
 */

import { invert, matmul, transpose } from './linalg.ts'
import { legendrePlm, realShDescoteaux } from './sh.ts'

export interface OpdtMatrices {
  /** OPDT fit matrices (nCoeff × nDir): ODF coeffs from log-signal terms. */
  deltaB: Float64Array
  deltaQ: Float64Array
  /** Residual-bootstrap matrices (nDir × nDir). */
  H: Float64Array
  R: Float64Array
  nCoeff: number
  nDir: number
}

export function opdtMatrices(
  dwiTheta: ArrayLike<number>,
  dwiPhi: ArrayLike<number>,
  maxOrder = 6,
  smooth = 0.006,
): OpdtMatrices {
  if (dwiTheta.length !== dwiPhi.length) {
    throw new Error(
      `opdtMatrices: theta/phi length mismatch (${dwiTheta.length} vs ${dwiPhi.length}).`,
    )
  }
  if (dwiTheta.length === 0) {
    throw new Error('opdtMatrices: no DWI directions.')
  }
  const { B, l } = realShDescoteaux(dwiTheta, dwiPhi, maxOrder)
  const nDir = dwiTheta.length
  const nCoeff = l.length

  // smooth_pinv: invB = (BᵀB + diag(smooth·L²))⁻¹ Bᵀ   (nCoeff × nDir),
  // where L = -l(l+1) is the Laplace-Beltrami eigenvalue per coefficient.
  const Bt = transpose(B, nDir, nCoeff) // nCoeff × nDir
  const BtB = matmul(Bt, nCoeff, nDir, B, nCoeff) // nCoeff × nCoeff
  const L = l.map((li) => -li * (li + 1))
  for (let c = 0; c < nCoeff; c++) BtB[c * nCoeff + c] += smooth * L[c] * L[c]
  const invB = matmul(invert(BtB, nCoeff), nCoeff, nCoeff, Bt, nDir) // nCoeff × nDir

  // F = P_l(0) (Legendre at 0); delta_b = F·L·invB, delta_q = 4·F·invB (row-scale).
  const F = l.map((li) => legendrePlm(li, 0, 0))
  const deltaB = new Float64Array(nCoeff * nDir)
  const deltaQ = new Float64Array(nCoeff * nDir)
  for (let c = 0; c < nCoeff; c++) {
    const fb = F[c] * L[c]
    const fq = 4 * F[c]
    for (let j = 0; j < nDir; j++) {
      deltaB[c * nDir + j] = fb * invB[c * nDir + j]
      deltaQ[c * nDir + j] = fq * invB[c * nDir + j]
    }
  }

  const H = hat(B, nDir, nCoeff)
  const R = lcrMatrix(H, nDir)
  return { deltaB, deltaQ, H, R, nCoeff, nDir }
}

/** Hat (projection) matrix H = B·pinv(B), nDir × nDir. Shape-adaptive pinv
 *  (no SVD): tall B → (BᵀB)⁻¹Bᵀ, wide B → Bᵀ(BBᵀ)⁻¹. Matches DIPY's `hat`. */
function hat(B: Float64Array, m: number, n: number): Float64Array {
  const Bt = transpose(B, m, n) // n × m
  // DIPY uses SVD (handles rank deficiency); we use exact inversion, so a
  // rank-deficient gradient set (duplicate/collinear bvecs) makes the Gram
  // matrix singular — surface that as a clear, actionable error.
  let pinv: Float64Array // n × m
  try {
    if (m >= n) {
      const BtB = matmul(Bt, n, m, B, n) // n × n
      pinv = matmul(invert(BtB, n), n, n, Bt, m) // n × m
    } else {
      const BBt = matmul(B, m, n, Bt, m) // m × m
      pinv = matmul(Bt, n, m, invert(BBt, m), m) // n × m
    }
  } catch {
    throw new Error(
      'Gradient directions are rank-deficient (duplicate or collinear bvecs); cannot build the bootstrap matrices.',
    )
  }
  return matmul(B, m, n, pinv, m) // m × m
}

/** Leveraged centered residuals matrix R, port of DIPY `lcr_matrix`. */
function lcrMatrix(H: Float64Array, n: number): Float64Array {
  const R = new Float64Array(n * n)
  const lev = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const d = H[i * n + i]
    // Guard the d≥1 case (numerator (I-H) is 0 there, so any positive divisor
    // yields 0 — avoids 0/0). Matches DIPY's `where=H.diagonal()<1`.
    lev[i] = d < 1 ? Math.sqrt(1 - d) : 1
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const ident = i === j ? 1 : 0
      R[i * n + j] = (ident - H[i * n + j]) / lev[i]
    }
  }
  // Subtract the column mean (DIPY: R - R.mean(0)).
  for (let j = 0; j < n; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += R[i * n + j]
    const mean = s / n
    for (let i = 0; i < n; i++) R[i * n + j] -= mean
  }
  return R
}
