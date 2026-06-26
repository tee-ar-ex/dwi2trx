/**
 * Real spherical-harmonic basis, a direct port of DIPY's
 * `real_sh_descoteaux(..., legacy=True)` (Descoteaux 2007 legacy convention) —
 * the `sampling_matrix` the OPDT bootstrap getter needs. Golden-tested
 * term-by-term against DIPY.
 *
 * Basis (per (l, m), l even): start from the standard complex SH
 *   Y_l^|m|(θ,φ) = N · P_l^|m|(cosθ) · e^{i|m|φ},  N = √((2l+1)/4π · (l-|m|)!/(l+|m|)!)
 * (P with the Condon-Shortley (−1)^m phase, matching scipy.lpmv), then
 *   realSH = (m>0 ? Im : Re) · (m==0 ? 1 : √2).
 */

/** (m, l) index list for even orders 0..maxOrder, in DIPY's column order. */
function shIndexList(maxOrder: number): { m: number[]; l: number[] } {
  const m: number[] = []
  const l: number[] = []
  for (let ll = 0; ll <= maxOrder; ll += 2) {
    for (let mm = -ll; mm <= ll; mm++) {
      m.push(mm)
      l.push(ll)
    }
  }
  return { m, l }
}

/**
 * Associated Legendre P_l^m(x), m ≥ 0, including the Condon-Shortley (−1)^m
 * phase (scipy.special.lpmv convention). Numerical-Recipes recurrence.
 */
export function legendrePlm(l: number, m: number, x: number): number {
  let pmm = 1
  if (m > 0) {
    const somx2 = Math.sqrt((1 - x) * (1 + x)) // √(1−x²)
    let fact = 1
    for (let i = 1; i <= m; i++) {
      pmm *= -fact * somx2 // accumulates (−1)^m (2m−1)!! (1−x²)^{m/2}
      fact += 2
    }
  }
  if (l === m) return pmm
  let pmmp1 = x * (2 * m + 1) * pmm
  if (l === m + 1) return pmmp1
  let pll = 0
  for (let ll = m + 2; ll <= l; ll++) {
    pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
    pmm = pmmp1
    pmmp1 = pll
  }
  return pll
}

/** √((l−m)!/(l+m)!) for m ≥ 0, by direct product (l small). */
function normFactor(l: number, m: number): number {
  let ratio = 1
  for (let k = l - m + 1; k <= l + m; k++) ratio /= k
  return Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * ratio)
}

/**
 * Evaluate the real SH basis at the given directions. `theta` (polar) and `phi`
 * (azimuth) are equal-length. Returns a row-major (nPoints × nCoeffs) matrix
 * plus the (m, l) lists. Matches DIPY `real_sh_descoteaux(legacy=True)`.
 */
export function realShDescoteaux(
  theta: ArrayLike<number>,
  phi: ArrayLike<number>,
  maxOrder = 6,
): { B: Float64Array; m: number[]; l: number[] } {
  if (theta.length !== phi.length) {
    throw new Error(
      `realShDescoteaux: theta/phi length mismatch (${theta.length} vs ${phi.length}).`,
    )
  }
  const { m, l } = shIndexList(maxOrder)
  const nCoeff = m.length
  const nPts = theta.length
  const B = new Float64Array(nPts * nCoeff)
  const SQRT2 = Math.SQRT2
  for (let p = 0; p < nPts; p++) {
    const x = Math.cos(theta[p])
    const az = phi[p]
    const row = p * nCoeff
    for (let c = 0; c < nCoeff; c++) {
      const mm = Math.abs(m[c])
      const ll = l[c]
      const N = normFactor(ll, mm)
      const P = legendrePlm(ll, mm, x)
      const base = N * P
      // m>0 → imaginary part (sin), else real part (cos)
      let v = m[c] > 0 ? base * Math.sin(mm * az) : base * Math.cos(mm * az)
      if (m[c] !== 0) v *= SQRT2
      B[row + c] = v
    }
  }
  return { B, m, l }
}
