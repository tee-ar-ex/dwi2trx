/**
 * Golden checks for the pure DVS generator. No framework — run with:
 *   node --experimental-strip-types src/dwi2trx/genvectors.test.ts
 */

import assert from 'node:assert/strict'
import {
  generateScheme,
  normalizeShells,
  optimizeDirections,
  schemeBaseName,
  schemeToDvs,
  sqrtRuleCount,
  suggestNextShell,
} from './genvectors.ts'

const approx = (a: number, b: number, eps = 1e-6) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b}`)

// --- optimizeDirections: right count, unit length, deterministic ---
{
  const a = optimizeDirections([{ count: 20, bval: 1000 }], { iters: 50 })
  assert.equal(a.length, 20)
  for (const d of a) approx(Math.hypot(d.x, d.y, d.z), 1, 1e-6)
  // Same seed ⇒ identical output.
  const b = optimizeDirections([{ count: 20, bval: 1000 }], { iters: 50 })
  assert.deepEqual(a, b)
  // Different seed ⇒ different layout.
  const c = optimizeDirections([{ count: 20, bval: 1000 }], {
    iters: 50,
    seed: 7,
  })
  assert.notDeepEqual(a, c)
}

// --- relaxation actually spreads points: min pairwise angle beats random ---
{
  const dirs = optimizeDirections([{ count: 30, bval: 1000 }], { iters: 300 })
  // Minimum angular separation (antipodal-aware) should be comfortably large.
  let minAng = Math.PI
  for (let i = 0; i < dirs.length; i++) {
    for (let j = i + 1; j < dirs.length; j++) {
      const dot = Math.abs(
        dirs[i].x * dirs[j].x + dirs[i].y * dirs[j].y + dirs[i].z * dirs[j].z,
      )
      minAng = Math.min(minAng, Math.acos(Math.min(1, dot)))
    }
  }
  // 30 antipodal directions on a hemisphere pack to ~20°+; assert a safe floor.
  assert.ok(
    minAng > (15 * Math.PI) / 180,
    `min separation ${(minAng * 180) / Math.PI}° too small`,
  )
}

// --- unequal shells exercise per-shell and combined-shell force weights ---
{
  const dirs = optimizeDirections([
    { count: 12, bval: 1000 },
    { count: 18, bval: 2000 },
  ])
  const minSeparation = (set: typeof dirs): number => {
    let min = Math.PI
    for (let i = 0; i < set.length; i++) {
      for (let j = i + 1; j < set.length; j++) {
        const dot = Math.abs(
          set[i].x * set[j].x + set[i].y * set[j].y + set[i].z * set[j].z,
        )
        min = Math.min(min, Math.acos(Math.min(1, dot)))
      }
    }
    return min
  }
  assert.ok(minSeparation(dirs) > (15 * Math.PI) / 180)
  assert.ok(
    minSeparation(dirs.filter((d) => d.bval === 1000)) > (25 * Math.PI) / 180,
  )
  assert.ok(
    minSeparation(dirs.filter((d) => d.bval === 2000)) > (25 * Math.PI) / 180,
  )
}

// --- generateScheme: leading b0, interleaved shells, correct total ---
{
  const s = generateScheme(
    [
      { count: 6, bval: 1000 },
      { count: 6, bval: 2000 },
    ],
    { iters: 50 },
  )
  assert.equal(s.maxBval, 2000)
  // 12 directions + 1 leading b0.
  assert.equal(s.dirs.length, 13)
  assert.equal(s.dirs[0].bval, 0) // leading b0 at origin
  approx(s.dirs[0].x, 0)
  // First two diffusion samples come from different shells (interleaved).
  assert.notEqual(s.dirs[1].bval, s.dirs[2].bval)
  // Counts per shell preserved.
  assert.equal(s.dirs.filter((d) => d.bval === 1000).length, 6)
  assert.equal(s.dirs.filter((d) => d.bval === 2000).length, 6)
}

// --- b0Every inserts interspersed b0s ---
{
  const s = generateScheme([{ count: 8, bval: 1000 }], {
    iters: 20,
    b0Every: 4,
  })
  // leading b0 + 8 dirs + a b0 before dir 5 (index 4) = 10.
  assert.equal(s.dirs.length, 10)
  assert.equal(s.dirs.filter((d) => d.bval === 0).length, 2)
}

// --- DVS: header + √(b/bmax) scaling + b0 line ---
{
  const s = generateScheme(
    [
      { count: 4, bval: 1000 },
      { count: 4, bval: 4000 },
    ],
    { iters: 40 },
  )
  const dvs = schemeToDvs(s)
  assert.match(dvs, /\[directions=9\]/) // 8 + leading b0
  assert.match(dvs, /coordinatesystem=xyz/)
  assert.match(dvs, /normalisation = none/)
  assert.match(dvs, /Vector\[0\] {2}= \(0\.000000, 0\.000000, 0\.000000\)/)
  // A b=1000 vector against bmax 4000 has length √(1/4) = 0.5.
  const b1000 = s.dirs.find((d) => d.bval === 1000)
  const len = Math.sqrt(1000 / 4000)
  approx(len, 0.5)
  // Every non-b0 line's parsed length equals √(bval/bmax) for its shell.
  const lines = dvs.split('\n').filter((l) => /^Vector\[/.test(l))
  assert.equal(lines.length, 9)
  for (let i = 0; i < s.dirs.length; i++) {
    const m = lines[i].match(/\(([^,]+), ([^,]+), ([^)]+)\)/)
    assert.ok(m)
    const vlen = Math.hypot(Number(m[1]), Number(m[2]), Number(m[3]))
    const expect = s.dirs[i].bval > 0 ? Math.sqrt(s.dirs[i].bval / 4000) : 0
    approx(vlen, expect, 1e-5)
  }
  assert.ok(b1000)
}

// --- normalizeShells: round, drop empties, merge duplicate b-values, clone ---
// sorted ascending by b-value
assert.deepEqual(
  normalizeShells([
    { count: 30, bval: 2000 },
    { count: 24, bval: 1000 },
  ]),
  [
    { count: 24, bval: 1000 },
    { count: 30, bval: 2000 },
  ],
)
// Duplicate b-values merge (counts sum).
assert.deepEqual(
  normalizeShells([
    { count: 10, bval: 1000 },
    { count: 15, bval: 1000 },
  ]),
  [{ count: 25, bval: 1000 }],
)
// Fractional counts round; non-positive / non-finite drop.
assert.deepEqual(
  normalizeShells([
    { count: 12.4, bval: 1000 },
    { count: 0, bval: 2000 },
    { count: 5, bval: -1 },
    { count: 5, bval: Number.NaN },
  ]),
  [{ count: 12, bval: 1000 }],
)

// --- generateScheme returns FRESH shells (no aliasing the caller's array) ---
{
  const input = [{ count: 6, bval: 1000 }]
  const s = generateScheme(input, { iters: 20 })
  input[0].count = 999 // mutate the caller's array afterwards
  assert.equal(s.shells[0].count, 6) // scheme unaffected
  assert.notEqual(s.shells[0], input[0]) // distinct object
}

// --- phase-interleaving spreads a small shell across the sequence (no tail) ---
{
  const s = generateScheme(
    [
      { count: 2, bval: 1000 },
      { count: 8, bval: 2000 },
    ],
    { iters: 20 },
  )
  const diff = s.dirs.filter((d) => d.bval > 0) // drop the leading b0
  const half = diff.length / 2
  const lowIdx = diff.flatMap((d, i) => (d.bval === 1000 ? [i] : []))
  // The two b=1000 volumes land on opposite halves — not clustered at the front
  // (round-robin used to leave a 6-long b=2000 tail).
  assert.ok(
    lowIdx.some((i) => i < half) && lowIdx.some((i) => i >= half),
    `b=1000 not spread across halves: ${lowIdx}`,
  )
}

// --- square-root rule: 24@1000 → 33@2000 → 39@3000 → 45@4000 ---
assert.equal(sqrtRuleCount(24, 1000, 2000), 33)
assert.equal(sqrtRuleCount(33, 2000, 3000), 39)
assert.equal(sqrtRuleCount(39, 3000, 4000), 45)
// suggestNextShell compounds off the last shell, bumping b by 1000.
assert.deepEqual(suggestNextShell([{ count: 24, bval: 1000 }]), {
  count: 33,
  bval: 2000,
})
assert.deepEqual(
  suggestNextShell([
    { count: 24, bval: 1000 },
    { count: 33, bval: 2000 },
  ]),
  { count: 39, bval: 3000 },
)
// Degenerate last shell (e.g. cleared b) falls back to a sane default.
assert.deepEqual(suggestNextShell([{ count: 5, bval: 0 }]), {
  count: 24,
  bval: 1000,
})
assert.deepEqual(suggestNextShell([]), { count: 24, bval: 1000 })

// --- schemeBaseName ---
assert.equal(
  schemeBaseName([
    { count: 30, bval: 1000 },
    { count: 30, bval: 2000 },
  ]),
  '30x1000_30x2000',
)

// --- validation errors ---
const throws = (fn: () => void, re: RegExp) =>
  assert.throws(fn, (e: unknown) => re.test((e as Error).message))
throws(() => generateScheme([], {}), /at least one shell/)
throws(
  () => generateScheme([{ count: 0, bval: 1000 }], {}),
  /at least one shell/,
)
throws(() => generateScheme([{ count: 2000, bval: 1000 }], {}), /Too many/)

console.log('genvectors.test.ts: all assertions passed ✓')
