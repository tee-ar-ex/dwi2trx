/**
 * Spherical geometry shared by the tractography port. Matches DIPY's
 * conventions exactly (verified against golden dumps) so the SH machinery in
 * Stage 3b can be checked term-by-term.
 */

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v

/**
 * Cartesian → spherical, matching `dipy.core.geometry.cart2sphere`:
 * `theta = acos(z/r)` in [0, π] (0 when r=0), `phi = atan2(y, x)` in (-π, π].
 */
export function cart2sphere(
  x: number,
  y: number,
  z: number,
): { r: number; theta: number; phi: number } {
  const r = Math.sqrt(x * x + y * y + z * z)
  const theta = r > 0 ? Math.acos(clamp(z / r, -1, 1)) : 0
  const phi = Math.atan2(y, x)
  return { r, theta, phi }
}
