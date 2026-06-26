/**
 * Pure helpers for the tracker's adaptive memory backoff — no GPU/shader deps,
 * so they unit-test in plain node (see backoff.test.ts). The tracker re-exports
 * `isOomError` for the UI.
 */

/** Smallest seed batch the adaptive loop will shrink to before giving up. */
export const MIN_CHUNK = 500

/** Next (smaller) batch size after an OOM, floored at MIN_CHUNK. */
export const nextChunkSize = (chunkSize: number): number =>
  Math.max(MIN_CHUNK, chunkSize >> 1)

/** Batch size after a SUCCESS: grow ~50% toward `max` rather than jumping back
 *  to full, so a dataset/GPU where only a reduced batch fits doesn't re-attempt
 *  (and re-fail) the known-bad full size on every batch. */
export const growChunkSize = (chunkSize: number, max: number): number =>
  Math.min(max, chunkSize + (chunkSize >> 1))

/** Whether an error is an out-of-memory failure that adaptive backoff / partial
 *  salvage should react to — covers host RangeError, GPUOutOfMemoryError, and the
 *  common browser/WebGPU "allocation failed" / "out of (device) memory" strings.
 *  (Device-loss is handled separately: shrinking is futile, but salvage applies.) */
export function isOomError(err: unknown): boolean {
  if (err instanceof RangeError) return true
  if (
    typeof GPUOutOfMemoryError !== 'undefined' &&
    err instanceof GPUOutOfMemoryError
  ) {
    return true
  }
  const msg = (err as Error)?.message ?? String(err)
  return /allocation failed|failed to allocate|out of memory|out of device memory|insufficient memory|memory limit|exceeded.*memory|too large for this gpu/i.test(
    msg,
  )
}

/** Device-loss (often downstream of OOM): can't retry, but host-side streamlines
 *  already read back are safe to keep. */
export function isDeviceLost(err: unknown): boolean {
  return /device.*lost|lost.*device/i.test((err as Error)?.message ?? '')
}

/**
 * Pure decision for how an over-budget batch should be handled — shared by the
 * "streamlines exceed the GPU buffer" guard and the catch block:
 *  - `shrink`: retry the SAME seeds with a halved batch (still above MIN_CHUNK)
 *  - `salvage`: at the floor but we already have streamlines → return partial
 *  - `rethrow`: nothing to salvage and can't shrink → surface the error
 */
export function planBatchError(o: {
  oom: boolean
  lost: boolean
  chunkSize: number
  hasOutput: boolean
}): 'shrink' | 'salvage' | 'rethrow' {
  if (o.oom && o.chunkSize > MIN_CHUNK) return 'shrink'
  if ((o.oom || o.lost) && o.hasOutput) return 'salvage'
  return 'rethrow'
}
