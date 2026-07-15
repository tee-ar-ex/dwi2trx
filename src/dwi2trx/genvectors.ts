/**
 * Generate uniform multi-shell diffusion gradient schemes and serialize them to
 * Siemens DVS files — a browser HEURISTIC inspired by the electrostatic-repulsion
 * model behind Emmanuel Caruyer's q-space sampling (Caruyer et al., MRM 2013) and
 * Anderson Winkler's `multishell.py`. It is NOT a faithful reimplementation of
 * their constrained (SLSQP) optimizer: it minimizes the same energy by projected-
 * gradient relaxation, which yields well-separated directions but not a certified
 * optimum. Treat the output as a good starting scheme to verify, not a validated
 * protocol.
 *
 * Directions are relaxed as charges on the unit sphere. Each pair (u, v) repels
 * through BOTH antipodes — cost v(u,v) = 1/|u−v|² + 1/|u+v|² — so an even number
 * of directions never collapses onto a single axis. The cost balances per-shell
 * uniformity against whole-scheme (all shells projected together) uniformity:
 * V = α·V₁ + (1−α)·V₂, α default 0.75, with each term size-normalized so α is
 * meaningful for unequal shells.
 *
 * Pure + DOM/NiiVue-free so it unit-tests in plain node (see genvectors.test.ts).
 * The optimiser is seeded (deterministic) so identical inputs give identical
 * schemes. The DVS scales each vector to |g| = √(b/b_max), the amplitude that
 * produces b-weighting b on a scanner running the max shell.
 */

/** One shell of the acquisition: how many directions, at what b-value. */
export interface ShellSpec {
  count: number
  bval: number
}

/** A single ordered sample: a unit direction + its shell b-value (a b0 is the
 *  zero vector with bval 0). */
export interface GenDir {
  x: number
  y: number
  z: number
  bval: number
}

export interface GenScheme {
  /** Ordered samples, b0s included. */
  dirs: GenDir[]
  /** Largest shell b-value (the DVS unit-length reference). */
  maxBval: number
  shells: ShellSpec[]
}

export interface GenOptions {
  /** Per-shell ↔ combined-shell uniformity weight (0..1, default 0.75). */
  alpha?: number
  /** Relaxation iterations (default 300). */
  iters?: number
  /** Insert a b0 before every N diffusion directions (0 = only a leading b0). */
  b0Every?: number
  /** PRNG seed for the initial layout (deterministic output). */
  seed?: number
}

const DEFAULT_ALPHA = 0.75
const DEFAULT_ITERS = 300
const DEFAULT_SEED = 42
// Guardrail: the optimiser is O(N²) per iteration on the main thread (~0.5 s at
// this cap; well under it for any real protocol, which rarely exceeds ~300). A
// Web Worker would remove the freeze entirely (deferred — see CLAUDE.md).
export const MAX_TOTAL_DIRECTIONS = 1000

/** Deterministic PRNG (mulberry32) so a scheme is reproducible from its inputs. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** A random point on the unit sphere (three Gaussians, normalized). */
function randUnit(rng: () => number): [number, number, number] {
  // Box–Muller for two standard normals; a third from a second draw.
  const g = () => {
    const u = Math.max(rng(), 1e-12)
    const v = rng()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  let x = g()
  let y = g()
  let z = g()
  let n = Math.hypot(x, y, z)
  if (n < 1e-9) {
    x = 1
    y = 0
    z = 0
    n = 1
  }
  return [x / n, y / n, z / n]
}

/**
 * Relax `shells` worth of directions on the sphere and return them as unit
 * vectors tagged with their shell b-value (no b0s, source order). Exposed for
 * testing; `generateScheme` wraps it with ordering + b0 insertion.
 *
 * A projected-gradient relaxation, NOT the exact SLSQP optimizer from the papers
 * — a fast in-browser heuristic on the same energy. Each pair's contribution is
 * normalized by its group size (per-shell for the α term, whole-scheme for the
 * 1−α term) so α balances the two objectives independently of unequal shell
 * sizes. Pairs are visited once and the equal/antipodal forces applied to both
 * endpoints (halving the inner loop).
 */
export function optimizeDirections(
  shells: ShellSpec[],
  opts: GenOptions = {},
): GenDir[] {
  const alpha = clamp(opts.alpha ?? DEFAULT_ALPHA, 0, 1)
  const iters = Math.max(1, Math.round(opts.iters ?? DEFAULT_ITERS))
  const rng = mulberry32(opts.seed ?? DEFAULT_SEED)

  const px: number[] = []
  const py: number[] = []
  const pz: number[] = []
  const shellOf: number[] = []
  shells.forEach((sh, si) => {
    for (let k = 0; k < sh.count; k++) {
      const [x, y, z] = randUnit(rng)
      px.push(x)
      py.push(y)
      pz.push(z)
      shellOf.push(si)
    }
  })
  const n = px.length

  // Size-normalized pair weights: an intra-shell pair carries α/(K_s−1), any pair
  // carries (1−α)/(N−1). So each point's total intra force is the *mean* over its
  // shell-mates and its combined force the *mean* over all others — independent of
  // how big each shell is.
  const combinedW = n > 1 ? (1 - alpha) / (n - 1) : 0
  const intraW = shells.map((s) => (s.count > 1 ? alpha / (s.count - 1) : 0))

  const fx = new Float64Array(n)
  const fy = new Float64Array(n)
  const fz = new Float64Array(n)

  for (let it = 0; it < iters; it++) {
    // Anneal the step so early chaos settles into a smooth final packing.
    const lr = 0.05 * (1 - it / iters) + 0.002
    fx.fill(0)
    fy.fill(0)
    fz.fill(0)
    // Visit each unordered pair once; accumulate forces on both endpoints.
    for (let i = 0; i < n; i++) {
      const uix = px[i]
      const uiy = py[i]
      const uiz = pz[i]
      const si = shellOf[i]
      for (let j = i + 1; j < n; j++) {
        const w = (si === shellOf[j] ? intraW[si] : 0) + combinedW
        // Direct term 2(u−v)/|u−v|⁴: equal and opposite on the two endpoints.
        const dx = uix - px[j]
        const dy = uiy - py[j]
        const dz = uiz - pz[j]
        let d2 = dx * dx + dy * dy + dz * dz
        if (d2 < 1e-4) d2 = 1e-4
        const a = (2 * w) / (d2 * d2)
        fx[i] += dx * a
        fy[i] += dy * a
        fz[i] += dz * a
        fx[j] -= dx * a
        fy[j] -= dy * a
        fz[j] -= dz * a
        // Antipodal term 2(u+v)/|u+v|⁴: SAME direction on both endpoints (the
        // gradient of 1/|u+v|² w.r.t. u and w.r.t. v is identical).
        const ex = uix + px[j]
        const ey = uiy + py[j]
        const ez = uiz + pz[j]
        let e2 = ex * ex + ey * ey + ez * ez
        if (e2 < 1e-4) e2 = 1e-4
        const b = (2 * w) / (e2 * e2)
        fx[i] += ex * b
        fy[i] += ey * b
        fz[i] += ez * b
        fx[j] += ex * b
        fy[j] += ey * b
        fz[j] += ez * b
      }
    }
    for (let i = 0; i < n; i++) {
      // Keep the force tangent to the sphere at u, then cap its magnitude so a
      // near-collision can't fling a point across the sphere (step ≤ lr).
      const dot = fx[i] * px[i] + fy[i] * py[i] + fz[i] * pz[i]
      const tx = fx[i] - dot * px[i]
      const ty = fy[i] - dot * py[i]
      const tz = fz[i] - dot * pz[i]
      const mag = Math.hypot(tx, ty, tz)
      const s = mag > 1 ? 1 / mag : 1
      const x = px[i] + lr * tx * s
      const y = py[i] + lr * ty * s
      const z = pz[i] + lr * tz * s
      const nn = Math.hypot(x, y, z) || 1
      px[i] = x / nn
      py[i] = y / nn
      pz[i] = z / nn
    }
  }

  const out: GenDir[] = []
  for (let i = 0; i < n; i++) {
    out.push({ x: px[i], y: py[i], z: pz[i], bval: shells[shellOf[i]].bval })
  }
  return out
}

/** Validate + normalize a shell list: round counts/b-values to positive integers,
 *  drop empties, merge shells that share a b-value, and return FRESH objects
 *  (never aliasing the caller's array) sorted ascending by b-value. */
export function normalizeShells(shells: ShellSpec[]): ShellSpec[] {
  const byBval = new Map<number, number>()
  for (const s of shells) {
    const count = Math.round(s.count)
    const bval = Math.round(s.bval)
    if (!Number.isFinite(count) || !Number.isFinite(bval)) continue
    if (count <= 0 || bval <= 0) continue
    byBval.set(bval, (byBval.get(bval) ?? 0) + count)
  }
  return [...byBval.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([bval, count]) => ({ count, bval }))
}

/**
 * Generate a full ordered scheme: relax the directions, spread each shell across
 * the sequence by phase (avoiding a long single-shell tail), and insert b0s (one
 * leading, then one before every `b0Every` directions). This ordering balances
 * shell counts; it does not optimize the angular coverage of arbitrary prefixes.
 */
export function generateScheme(
  shells: ShellSpec[],
  opts: GenOptions = {},
): GenScheme {
  const clean = normalizeShells(shells)
  if (clean.length === 0) {
    throw new Error('Add at least one shell with a positive count and b-value.')
  }
  const total = clean.reduce((s, sh) => s + sh.count, 0)
  if (total > MAX_TOTAL_DIRECTIONS) {
    throw new Error(
      `Too many directions (${total}). Keep the total under ${MAX_TOTAL_DIRECTIONS}.`,
    )
  }

  const relaxed = optimizeDirections(clean, opts)

  // Group by shell (source order matches `clean`), then interleave by fractional
  // phase so each shell is spread uniformly through the acquisition order.
  const byShell: GenDir[][] = clean.map(() => [])
  let idx = 0
  clean.forEach((sh, si) => {
    for (let k = 0; k < sh.count; k++) byShell[si].push(relaxed[idx++])
  })
  const phased: Array<{ dir: GenDir; phase: number }> = []
  for (const shellDirs of byShell) {
    const c = shellDirs.length
    for (let k = 0; k < c; k++) {
      phased.push({ dir: shellDirs[k], phase: (k + 0.5) / c })
    }
  }
  phased.sort((a, b) => a.phase - b.phase)
  const interleaved = phased.map((p) => p.dir)

  const b0Every = Math.max(0, Math.round(opts.b0Every ?? 0))
  const b0: GenDir = { x: 0, y: 0, z: 0, bval: 0 }
  const dirs: GenDir[] = [b0] // always a leading b0
  interleaved.forEach((d, i) => {
    if (b0Every > 0 && i > 0 && i % b0Every === 0) dirs.push(b0)
    dirs.push(d)
  })

  const maxBval = Math.max(...clean.map((s) => s.bval))
  return { dirs, maxBval, shells: clean }
}

/**
 * Serialize to a Siemens DVS (the `multishell.py` variant: `[directions=N]`,
 * `coordinatesystem=xyz`, `Vector[i] = (x, y, z)`). Each direction is scaled to
 * |g| = √(b/b_max) so its length encodes the shell's diffusion weighting.
 */
export function schemeToDvs(scheme: GenScheme): string {
  const bmax = scheme.maxBval > 0 ? scheme.maxBval : 1
  const shellDesc = scheme.shells
    .map((s) => `${s.count}×b=${s.bval}`)
    .join(', ')
  const nB0 = scheme.dirs.filter((d) => d.bval <= 0).length
  const lines: string[] = [
    `# Diffusion vector set generated by dwi2trx — ${shellDesc}${nB0 ? `, ${nB0} b0` : ''}.`,
    '# Directions from an antipodal electrostatic-repulsion relaxation (heuristic,',
    '# after Caruyer et al. 2013). Verify coverage before scanner use.',
    `[directions=${scheme.dirs.length}]`,
    'coordinatesystem=xyz',
    'normalisation = none',
  ]
  scheme.dirs.forEach((d, i) => {
    const g = d.bval > 0 ? Math.sqrt(d.bval / bmax) : 0
    const x = (d.x * g).toFixed(6)
    const y = (d.y * g).toFixed(6)
    const z = (d.z * g).toFixed(6)
    lines.push(`Vector[${i}]  = (${x}, ${y}, ${z})`)
  })
  return `${lines.join('\n')}\n`
}

/** A filename-safe descriptor of the scheme, e.g. `30x1000_30x2000`. */
export function schemeBaseName(shells: ShellSpec[]): string {
  return shells.map((s) => `${s.count}x${s.bval}`).join('_')
}

/** Round to the nearest multiple of 3 (min 3). Diffusion protocols favour
 *  direction counts divisible by 3, and it reproduces the √b reference tables. */
function roundDirs(x: number): number {
  return Math.max(3, 3 * Math.round(x / 3))
}

/**
 * Square-root rule: scale the previous shell's count by √(newBval/lastBval), a
 * common starting heuristic for allocating more directions at higher b-values.
 * It compounds shell-to-shell (24@1000 → 33@2000 → 39@3000 → 45@4000).
 */
export function sqrtRuleCount(
  lastCount: number,
  lastBval: number,
  newBval: number,
): number {
  if (lastBval <= 0 || lastCount <= 0 || newBval <= 0) return 24
  return roundDirs(lastCount * Math.sqrt(newBval / lastBval))
}

/** The shell to append when the user adds one: b-value + 1000, with directions
 *  set by the √b rule off the last shell (a sensible starting point to edit). */
export function suggestNextShell(shells: ShellSpec[]): ShellSpec {
  const last = shells[shells.length - 1]
  if (!last || last.bval <= 0) return { count: 24, bval: 1000 }
  const bval = last.bval + 1000
  return { count: sqrtRuleCount(last.count, last.bval, bval), bval }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}
