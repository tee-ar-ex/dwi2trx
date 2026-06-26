# CLAUDE.md — dwi2trx project brief

## Project

Browser-only diffusion-MRI pipeline: drop in a DWI, fit the diffusion tensor, track white-matter streamlines on the GPU, and save a `.trx` tractogram. Everything runs client-side — no data leaves the machine, no server. It combines WASM tools (niimath for tensor fitting + reslicing, dcm2niix for in-browser DICOM→NIfTI) with WebGPU (NiiVue rendering, mindgrab brain masking, a GPUStreamlines port for tractography). The pipeline stages live in `src/main.ts` (`runFit` = mask + tensor fit, `runTrack` = streamline tracking + TRX + 3D preview); the tensor fit is in `src/dwi2trx/dtifit.ts`; tracking is in `src/dwi2trx/tracking/` (`tracker.ts` GPU orchestration, `inputs.ts` NIfTI read + seed generation).

## Build & test

- `npm run dev` (Vite dev server), `npm run build` (`tsc --noEmit && vite build` → `dist/`), `npm run lint` (Biome), `npm run typecheck` (`tsc --noEmit`), `npm test`, `npm run test:e2e`.
- `test:e2e` builds then runs `e2e/smoke.mjs` (nominal pipeline) — needs a real GPU + Chrome (`CHROME_PATH`); local-only. `test:stress` runs `e2e/stress.mjs`: aggressive seeding (Seed FA 0.1, Density 4) through the real multi-batch tracking + windowed-readback + preview + save path; it waits for the PREVIEW phase to finish (a "saved as TRX" status), not just the Save button, and writes `e2e/large-data-report.md`. Exit codes: 0 pass, 1 fail, 75 skip (no Chrome/GPU) — distinct so "not run" ≠ "passed".
- Both e2e scripts wait for a *complete* download (no `.crdownload`, non-zero size stable across two polls) before parsing the TRX — reading mid-flush was an intermittent "invalid zip data" failure.
- Tests are plain `node --experimental-strip-types` files (no framework). The full list lives in `package.json` `"test"`: `validate.test.ts`, `dtifit.test.ts`, `lib/nifti-geometry.test.ts` (qformAffineRows + readAffine), `niimath-sform.test.ts` (vendored-WASM sform sync), `tracking/backoff.test.ts` (adaptive backoff / OOM classification / salvage — forced-backoff harness), `tracking/nifti-read.test.ts` (layout, scaling, truncation, size preflight), `tracking/tracking.test.ts`.
- Node can't run anything that transitively imports the GPU/shader graph (extensionless runtime imports). Keep unit-testable logic in dependency-free leaf modules (e.g. `tracking/backoff.ts`, `lib/nifti-geometry.ts`) and import those from the test.

## Vendored niimath (CRITICAL provenance)

- The niimath WASM is **vendored** at `vendor/niimath/dist/`, wired as the `file:./vendor/niimath` dependency `@niivue/niimath`. The SOURCE repo is the rordenlab/niimath checkout (maintainer-local; on the current maintainer's machine `/Users/chris/src/niimath` — adjust to wherever yours lives).
- The app depends on a niimath fix: `nifti_image_read()` now fills a MISSING/INVALID sform from a valid qform on read (`src/nifti_io.c`, function `nifti_sync_sform_from_qform`, condition `q_ok && !s_ok`, with a scale-invariant degeneracy check in `nifti_dmat44_spatial_ok`). This is why niimath's `-reslice_nn` correctly aligns the mindgrab mask for qform-only DWI.
- To rebuild + re-vendor (paths are maintainer-local; substitute your emsdk + niimath checkout): activate emsdk (`source <emsdk>/emsdk_env.sh`, built with emsdk 6.0.1), then in `<niimath>/js` run `bun run prebuild && bun run build`, then copy `<niimath>/js/dist/*` into this repo's `vendor/niimath/dist/`.
- `src/dwi2trx/niimath-sform.test.ts` proves the vendored WASM syncs the sform from the qform — it guards against a future re-vendor regressing this.

## The qform-only bug class (what was fixed, why it matters)

- Symptoms on FSL-preprocessed qform-only DWI (`sform_code = 0`): (1) the masked fit produced an empty brain mask → "No seed voxels"; (2) fibers were misaligned with the image; (3) "Array buffer allocation failed".
- Root cause: both niimath's reslice AND the tracker's `readAffine` read only the sform.
- Fixes: the niimath sform sync (above), plus `readAffine()` + `qformAffineRows()` in `src/lib/nifti-geometry.ts` (a neutral geometry module) which the tracker (`tracking/inputs.ts`) uses for voxel→RASMM: sform → qform → pixdim. NOTE: the tracker reads the RAW uploaded DWI, which niimath never processes, so the JS qform fallback is REQUIRED independently of the niimath fix.
- `qformAffineRows` follows `nifti1.h`, including over-unit quaternion renormalization (b²+c²+d² > 1 → a = 0, renormalize) and the qfac sign on the z column.

## Memory / large datasets (tracker, `src/dwi2trx/tracking/tracker.ts`)

- The GPU streamline buffer reserves `MAX_SLINE_LEN` (501) points × streamlines-in-batch even though most streamlines are far shorter — so the readback is the bottleneck. It is read back in **fixed windows** (`READBACK_WINDOW_BYTES = 128 MB`). NOTE the true peak per window is staging buffer + the `getMappedRange().slice(0)` host copy + the retained streamline slices, i.e. somewhat above 128 MB — the constant bounds the dominant term, not the exact total.
- **Adaptive batching** (pure logic in `tracking/backoff.ts`, unit-tested in `backoff.test.ts`): starts at `DEFAULT_PARAMS.chunkSize = 10000` seeds; on an OOM (host RangeError, `GPUOutOfMemoryError`, broadened message match in `isOomError`, or a batch whose streamline buffer exceeds the GPU binding limit) it halves the batch and retries the SAME seed range down to `MIN_CHUNK = 500`. `planBatchError()` makes the shrink/salvage/rethrow decision (shared by the over-limit guard and the catch). A batch's streamlines are staged locally and merged only after it fully reads back, so a retry never double-counts. The over-limit-at-floor case now salvages a partial too (was a hard error — round-3 fix).
- Last resort: at the floor with output (or device-loss), `trackStreamlines` returns `{ lines, truncated: true, processedSeeds, totalSeeds }`; `runTrack` reports "P of N seeds". Static GPU input buffers are validated BEFORE allocation: `storageFromParts()` checks combined size and writes parts straight into the mapped buffer (no host concat), with an actionable "DWI too large for this GPU" error.
- Abort: `trackStreamlines` takes `shouldStop()` (main.ts passes `() => seq !== loadSeq`), polled between batches AND between readback windows, so a new DWI dropped mid-track stops within a window and frees GPU buffers. (niimath WASM `callMain` is synchronous and cannot be aborted mid-op; the `loadSeq` guard discards its stale result.) The mindgrab transient b0 load is serialized on the canvas chain via `runOnCanvas()` so it can't interleave with a nav swap.
- `runTrack` builds + stores the TRX, then frees the voxel-space `lines` (`lines.length = 0`) BEFORE the 3D preview (a separate, often-larger NiiVue cylinder-mesh allocation). A preview failure is caught separately and classified OOM-vs-other (`isOomError`), so the TRX stays downloadable either way.
- `assembleTrackingInputs` (`tracking/inputs.ts`) reads the DWI/FA **straight into** the tracker layout via `parseNifti` + `readDataf`/`readMetric` (sampling source voxels directly from the decompressed bytes) — no intermediate x-fastest `raw` copy. Peak is now decompressed bytes + one reordered copy, read sequentially so the two files' blobs don't pile up.

## Deferred / known limitations (do NOT silently drop)

- `assembleTrackingInputs` still holds the decompressed DWI bytes + the reordered float copy simultaneously (down from +raw, but still ~2× the volume). `gunzipAll` now peeks the gzip ISIZE trailer and rejects up front (plus a runtime backstop) when the decompressed image exceeds the GPU buffer limit, but a volume *under* that limit still allocates bytes+reordered at once — eliminating the last copy needs streaming/windowed decode (deferred).
- In-batch cancellation is observed between readback windows, not mid-kernel: a single very large batch still runs pass 1 + pass 2 to completion before the next `shouldStop` check. Acceptable; finer granularity would need splitting the dispatch.
- niimath `callMain` (fit/mask) is synchronous and non-abortable; a changed input during fit burns WASM time until completion (the `loadSeq` guard discards the stale result). Moving it to a worker is the real fix — deferred.
- The wired `test:stress` proves the real multi-batch + windowed-readback + preview + save path, but on an ample-RAM machine it does NOT force a true host OOM — so the *truncated/preview-OOM* branches are exercised in `backoff.test.ts` (logic) but not yet end-to-end on real hardware. Forcing that needs the manual protocol below on a memory-constrained machine.

## Manual large-data / forced-memory protocol (run on constrained hardware before broad release)

`test:stress` is the automated baseline (records `e2e/large-data-report.md`). To cover what ample RAM can't force, also run by hand on a memory-constrained GPU/machine with a big DWI (e.g. the ~800 MB / 129-dir HBN sample): Fit, then Generate with Seed FA 0.1 / dense seeding. Expect + record: tracking completes or reports a PARTIAL "P of N seeds" (not a hard failure); if the 3D preview OOMs the status says so and **Save TRX stays enabled** and downloads a valid `.trx`; dropping a new DWI mid-track stops the run promptly; a moderate run still aligns (fibers over FA).

## Conventions

- Markdown: no hard-wrapping (one paragraph per line).
- Commits: validate (lint + typecheck + test + build) before committing. User is neurolabusc / Chris Rorden.
