/**
 * Forced-backoff harness for the tracker's adaptive batch logic — the
 * memory/cancellation edge cases the happy-path WebGPU smoke can't reach. Tests
 * the pure decision pieces (`isOomError`, `nextChunkSize`, `planBatchError`) plus
 * a simulated batch run that drives shrink → salvage → fail without a GPU.
 *
 * Run: node --experimental-strip-types src/dwi2trx/tracking/backoff.test.ts
 */
import assert from 'node:assert/strict'
import {
  growChunkSize,
  isOomError,
  nextChunkSize,
  planBatchError,
} from './backoff.ts'

// --- isOomError classification (broadened forms) ---
assert.ok(
  isOomError(new RangeError('Array buffer allocation failed')),
  'RangeError',
)
assert.ok(isOomError(new Error('Out of memory')), 'out of memory')
assert.ok(
  isOomError(new Error('Failed to allocate buffer')),
  'failed to allocate',
)
assert.ok(isOomError(new Error('out of device memory')), 'device memory')
assert.ok(
  isOomError(new Error('insufficient memory for allocation')),
  'insufficient',
)
assert.ok(
  !isOomError(new Error('Tint WGSL validation error')),
  'validation ≠ oom',
)
assert.ok(!isOomError(new TypeError('x is not a function')), 'TypeError ≠ oom')

// --- nextChunkSize halves, floored at MIN_CHUNK (500) ---
assert.equal(nextChunkSize(10000), 5000, 'halve 10k')
assert.equal(nextChunkSize(1000), 500, 'halve to floor')
assert.equal(nextChunkSize(500), 500, 'already at floor stays')
assert.equal(nextChunkSize(700), 500, 'clamp up to floor')

// --- growChunkSize grows ~50% toward the cap, never past it ---
assert.equal(growChunkSize(2500, 10000), 3750, 'grow 50%')
assert.equal(growChunkSize(8000, 10000), 10000, 'clamp to cap')
assert.equal(growChunkSize(10000, 10000), 10000, 'at cap stays')

// --- planBatchError decision matrix ---
const plan = (
  oom: boolean,
  lost: boolean,
  chunkSize: number,
  hasOutput: boolean,
) => planBatchError({ oom, lost, chunkSize, hasOutput })
assert.equal(
  plan(true, false, 10000, false),
  'shrink',
  'oom above floor → shrink',
)
assert.equal(
  plan(true, false, 10000, true),
  'shrink',
  'oom above floor → shrink even with output',
)
assert.equal(
  plan(true, false, 500, true),
  'salvage',
  'oom at floor + output → salvage',
)
assert.equal(
  plan(true, false, 500, false),
  'rethrow',
  'oom at floor, nothing → rethrow',
)
assert.equal(
  plan(false, true, 10000, true),
  'salvage',
  'device lost + output → salvage (no shrink)',
)
assert.equal(
  plan(false, true, 10000, false),
  'rethrow',
  'device lost, nothing → rethrow',
)
assert.equal(
  plan(false, false, 10000, true),
  'rethrow',
  'non-memory error → rethrow',
)

// --- Simulated adaptive run: drive the same loop logic the tracker uses ---
// A batch "fits" only when its seed count is ≤ capacity; otherwise it OOMs.
// Mirrors trackStreamlines' start/chunkSize/processed bookkeeping.
function simulate(opts: {
  totalSeeds: number
  capacity: number // max seeds a batch can read back before OOM
  startChunk: number
  produced: (n: number) => number // streamlines a batch of n seeds yields
}): {
  processed: number
  truncated: boolean
  failed: boolean
  collected: number
} {
  let start = 0
  let chunk = opts.startChunk
  let collected = 0
  let truncated = false
  while (start < opts.totalSeeds) {
    const n = Math.min(chunk, opts.totalSeeds - start)
    const oom = n > opts.capacity // this batch can't be read back
    if (oom) {
      const p = planBatchError({
        oom: true,
        lost: false,
        chunkSize: chunk,
        hasOutput: collected > 0,
      })
      if (p === 'shrink') {
        chunk = nextChunkSize(chunk)
        continue
      }
      if (p === 'salvage') {
        truncated = true
        break
      }
      return { processed: start, truncated, failed: true, collected }
    }
    collected += opts.produced(n)
    start += n
    chunk = growChunkSize(chunk, opts.startChunk) // grow cautiously after success
  }
  return { processed: start, truncated, failed: false, collected }
}

// Capacity above startChunk → completes in full-size batches, no truncation.
{
  const r = simulate({
    totalSeeds: 30000,
    capacity: 100000,
    startChunk: 10000,
    produced: (n) => n,
  })
  assert.equal(r.failed, false)
  assert.equal(r.truncated, false)
  assert.equal(r.processed, 30000)
  assert.equal(r.collected, 30000)
}

// Capacity below startChunk but above the floor → shrinks, then completes fully.
{
  const r = simulate({
    totalSeeds: 30000,
    capacity: 4000,
    startChunk: 10000,
    produced: (n) => n,
  })
  assert.equal(r.failed, false, 'shrinks to a fitting batch')
  assert.equal(r.truncated, false, 'completes, not truncated')
  assert.equal(r.processed, 30000)
}

// First batch already produced output, then capacity collapses below the floor →
// salvage a partial instead of failing (the round-3 minimum-batch bug).
{
  let calls = 0
  const r = simulate({
    totalSeeds: 30000,
    startChunk: 1000,
    produced: (n) => n,
    // first batch fits (cap 1000), afterwards nothing fits even at the floor
    get capacity() {
      return calls++ === 0 ? 1000 : 0
    },
  })
  assert.equal(r.failed, false, 'does not fail the whole run')
  assert.equal(r.truncated, true, 'returns a truncated partial')
  assert.ok(r.collected > 0, 'kept the first batch')
  assert.ok(r.processed < 30000, 'did not finish')
}

// Nothing ever fits and no output → hard failure (surfaces the real error).
{
  const r = simulate({
    totalSeeds: 30000,
    capacity: 0,
    startChunk: 10000,
    produced: (n) => n,
  })
  assert.equal(r.failed, true, 'no salvage possible → fail')
  assert.equal(r.collected, 0)
}

console.log(
  'backoff.test: OK — isOomError, nextChunkSize, planBatchError, adaptive run',
)
