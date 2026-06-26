/**
 * Minimal dense linear algebra for the small matrices (≤28×28) the OPDT
 * bootstrap matrices need. Flat row-major Float64, explicit dims — no
 * dependency, no SVD/QR. The OPDT regularization (smooth=0.006) conditions the
 * matrices well within the golden tolerance, so Gauss-Jordan suffices.
 */

/** Transpose an r×c matrix → c×r. */
export function transpose(A: Float64Array, r: number, c: number): Float64Array {
  const T = new Float64Array(r * c)
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) T[j * r + i] = A[i * c + j]
  }
  return T
}

/** Multiply A (ar×ac) by B (ac×bc) → ar×bc. */
export function matmul(
  A: Float64Array,
  ar: number,
  ac: number,
  B: Float64Array,
  bc: number,
): Float64Array {
  const C = new Float64Array(ar * bc)
  for (let i = 0; i < ar; i++) {
    for (let k = 0; k < ac; k++) {
      const a = A[i * ac + k]
      if (a === 0) continue
      for (let j = 0; j < bc; j++) C[i * bc + j] += a * B[k * bc + j]
    }
  }
  return C
}

/** Inverse of an n×n matrix via Gauss-Jordan with partial pivoting. */
export function invert(M: Float64Array, n: number): Float64Array {
  const a = Float64Array.from(M)
  const inv = new Float64Array(n * n)
  for (let i = 0; i < n; i++) inv[i * n + i] = 1
  for (let col = 0; col < n; col++) {
    let piv = col
    let max = Math.abs(a[col * n + col])
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r * n + col])
      if (v > max) {
        max = v
        piv = r
      }
    }
    // `!(max >= …)` (not `max < …`) so a NaN pivot also throws rather than
    // silently propagating. Absolute threshold assumes O(1)-magnitude matrices
    // (true for the real-SH Gram matrices here).
    if (!(max >= 1e-12)) throw new Error('invert: matrix is singular.')
    if (piv !== col) {
      swapRows(a, n, col, piv)
      swapRows(inv, n, col, piv)
    }
    const d = a[col * n + col]
    for (let j = 0; j < n; j++) {
      a[col * n + j] /= d
      inv[col * n + j] /= d
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const f = a[r * n + col]
      if (f === 0) continue
      for (let j = 0; j < n; j++) {
        a[r * n + j] -= f * a[col * n + j]
        inv[r * n + j] -= f * inv[col * n + j]
      }
    }
  }
  return inv
}

function swapRows(a: Float64Array, n: number, r1: number, r2: number): void {
  for (let j = 0; j < n; j++) {
    const t = a[r1 * n + j]
    a[r1 * n + j] = a[r2 * n + j]
    a[r2 * n + j] = t
  }
}
