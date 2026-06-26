/**
 * The tracking sphere (DIPY `default_sphere`): unit-vector directions + their
 * Delaunay edges, used by the direction getter to pick streamline steps.
 *
 * Loaded from the committed `public/sphere.bin` (packed from DIPY):
 *   int32 nVerts, int32 nEdges,
 *   float32 vertices[nVerts*3] (x,y,z row-major),
 *   int32   edges[nEdges*2].
 */

import { cart2sphere } from './spherical.ts'

export interface Sphere {
  vertices: Float32Array // nVerts * 3 (x,y,z)
  edges: Int32Array // nEdges * 2
  nVerts: number
  nEdges: number
}

export function parseSphere(buf: ArrayBuffer): Sphere {
  if (buf.byteLength < 8) throw new Error('sphere.bin: truncated header.')
  const dv = new DataView(buf)
  const nVerts = dv.getInt32(0, true)
  const nEdges = dv.getInt32(4, true)
  const need = 8 + nVerts * 3 * 4 + nEdges * 2 * 4
  if (nVerts <= 0 || nEdges <= 0 || buf.byteLength < need) {
    throw new Error(
      `sphere.bin: corrupt or truncated (need ${need} bytes for ${nVerts} verts / ${nEdges} edges, got ${buf.byteLength}).`,
    )
  }
  let off = 8
  const vertices = new Float32Array(buf, off, nVerts * 3)
  off += nVerts * 3 * 4
  const edges = new Int32Array(buf, off, nEdges * 2)
  return { vertices, edges, nVerts, nEdges }
}

export async function loadSphere(
  url = `${import.meta.env.BASE_URL}sphere.bin`,
): Promise<Sphere> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not fetch sphere ${url} (${res.status}).`)
  return parseSphere(await res.arrayBuffer())
}

/** Per-vertex (theta, phi), matching DIPY's `sphere.theta` / `sphere.phi`. */
export function sphereThetaPhi(s: Sphere): {
  theta: Float64Array
  phi: Float64Array
} {
  const theta = new Float64Array(s.nVerts)
  const phi = new Float64Array(s.nVerts)
  for (let i = 0; i < s.nVerts; i++) {
    const sph = cart2sphere(
      s.vertices[i * 3],
      s.vertices[i * 3 + 1],
      s.vertices[i * 3 + 2],
    )
    theta[i] = sph.theta
    phi[i] = sph.phi
  }
  return { theta, phi }
}
