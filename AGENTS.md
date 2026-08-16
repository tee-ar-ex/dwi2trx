# dwi2trx contributor guide

`CLAUDE.md` points here. Keep this file limited to durable architecture, invariants, and non-obvious traps.

## Purpose and shape

dwi2trx is a browser-only diffusion-MRI demonstration: load NIfTI/bval/bvec or DICOM, fit a tensor, track streamlines with WebGPU, and save TRX. Data never leaves the browser. Prefer clear examples and small pure functions over production-style abstraction or repeated defensive layers.

- `src/main.ts`: UI and pipeline orchestration.
- `src/dwi2trx/dtifit.ts`: niimath/WASM tensor operations.
- `src/dwi2trx/tracking/`: WebGPU tracking, NIfTI input preparation, and adaptive batching.
- `src/dwi2trx/vectors.ts`: loaded-scheme parsing and visualization data.
- `src/dwi2trx/genvectors.ts`: pure vector generation, conditioning, QC, ordering, and scanner serializers.

## Brain masking (@brainchop/mindgrab)

The mask is entirely the package's: `segment(b0, { model: 'mindgrab', mask: true, worker: true, backend: 'webgpu', assetPath })` returns a binary mask already on the input grid, so this repo holds no model, no conform step, and no mask GPU state. `dtifit.ts` stages the mask straight into niimath — do not reslice it, and do not dilate it (scalp FA is noisy; ask the package for `borderMm` instead). The worker acquires and releases its own device per call, so nothing has to be freed before the tracker allocates.

The wasm module loads its emscripten glue by a URL computed at run time, and the glue finds its own `.wasm` through `import.meta.url`, so Vite can neither rewrite the import nor emit the assets. `scripts/copy-brainchop.mjs` stages them into gitignored `public/brainchop/` on every `dev`/`build`, and `assetPath` points there. Only the WebGPU pair is copied, so `backend: 'webgpu'` is pinned — `auto` would reach for WebGL2/CPU files this repo does not ship. That is safe because dwi2trx already requires WebGPU for tracking; a GPU without `shader-f16` or 512 MiB buffers throws a `BrainchopError`, which `runFit` catches and falls back to an unmasked fit.

## Commands and style

- Validate with `npm run lint`, `npm run typecheck`, `npm test`, and `npm run build`. Use `npm run test:e2e` when browser initialization, rendering, downloads, or the full pipeline changes; it requires Chrome and a real GPU.
- Tests are plain `node --experimental-strip-types` scripts listed in `package.json`. Keep unit-testable code in dependency-light leaf modules; Node cannot execute the WebGPU/shader dependency graph.
- Biome owns TypeScript formatting. Markdown uses one paragraph per line rather than hard wrapping.
- Keep changes local and proportionate. Preserve user changes in a dirty tree. Avoid refactors that only move complexity.

## Core lifecycle rules

- `loadSeq` is the identity of the active dataset. Async work must check it before publishing state or touching a viewer.
- Every new input goes through `beginLoad()`, which aborts the preceding load before incrementing `loadSeq`. This is required to terminate obsolete dcm2niix workers and avoid overlapping WASM heaps.
- The bundled sample loads only when no explicit user load began during WebGPU initialization.
- Generator work uses a Web Worker and `genRevision`; edits, close, and replacement generation invalidate stale results and Save buttons.
- Viewer creation is cached as a promise, not only as a resolved instance, so rapid opens cannot attach multiple NiiVue controls to one canvas. Canvas mutations are serialized.
- Blob download URLs are revoked on a later task; synchronous revocation can cancel Safari downloads.
- A `<dialog>` ID rule with `display:flex` overrides the browser's closed-dialog rule. Apply layout display only to `[open]` dialogs.

## NIfTI geometry and vendored niimath

- `vendor/niimath/dist/` is the application dependency via `file:./vendor/niimath`; do not silently replace it with an npm build.
- The vendored build repairs a missing/invalid sform from a valid qform. `src/dwi2trx/niimath-sform.test.ts` protects this behavior. Everything downstream inherits the repair: the b0 niimath crops for mindgrab, and the fit outputs NiiVue displays.
- Tracking reads the original uploaded DWI, not a niimath output, so it needs its own `readAffine()` in `src/lib/nifti-geometry.ts`: sform, then qform, then pixdim fallback. Removing either qform path breaks alignment for FSL-style qform-only images.
- niimath `callMain` is synchronous. If it throws, `dtifit.ts` invalidates the cached module because an aborted Emscripten runtime cannot safely be reused.

## Memory and cancellation

- Browser/WASM arrays have practical contiguous-memory limits regardless of physical RAM. `input-limits.ts` enforces the 2 GB input cap on every entry path and again after DICOM conversion.
- A compressed NIfTI below the cap can still inflate beyond wasm32 or require too many fit intermediates. The failure is reported clearly and the poisoned niimath module is reset; exact preflight would require duplicate decompression and a peak-allocation model.
- Tracking reads GPU output in 128 MB windows and adaptively halves seed batches on OOM down to `MIN_CHUNK`. Batch results merge only after complete readback, so retries must not duplicate streamlines.
- Cancellation is observed between batches and readback windows, not inside a submitted GPU kernel, a synchronous niimath call, or a running mindgrab segmentation. Stale results must still be discarded through `loadSeq`.
- Tracking retains decompressed DWI bytes plus one reordered float copy. Removing the remaining copy requires streaming decode and is intentionally deferred.
- Free large voxel-space streamline arrays before constructing the NiiVue preview. A preview failure must not invalidate an already-created TRX download.

## Vector generation and export

- The generator has two deliberately different methods:
  - **Simultaneous/Winkler:** optimizes the complete set, applies alpha, then uses duty-cycle ordering.
  - **Incremental/Caruyer:** adds directions with earlier samples fixed, ignores alpha, and preserves construction order so truncated prefixes remain useful. Do not pass it through the duty-cycle sorter.
- Both methods balance polarity per shell and rotate the complete set so the first direction is `[1,1,1]/sqrt(3)`. Antipodal nodes are visualization-only and must never enter exported schemes.
- `MIN_DIRECTIONS_PER_SHELL` is 6. Generation runs off the main thread; QC metrics are logged to the console rather than expanding the UI.
- Plot radius defaults to `sqrt(b/bmax)`, while saved formats have different semantics:
  - **Siemens DVS:** vector magnitude is `sqrt(b/bmax)`; comments and LF endings.
  - **GE DAT:** vector magnitude is `sqrt(b/bmax)` because effective b-value is encoded by magnitude; space-delimited, LF endings, documented 6–300-volume range.
  - **Philips `dti_vectors_input.txt`:** unit x/y/z directions plus an explicit raw b-value column. Do not apply Siemens/GE amplitude scaling — Philips carries b in its own column. Rules enforced in code, from dcm2niix `Philips/README.md`: a b=0 row must be FIRST, and repeated b0s must each carry a UNIQUE direction (we space them around the XY circle) — a zero-vector b0 is not a documented form. A header is optional and omitted (if present it must not start with a digit), so no comments. The `Opt x` 6–128 range applies to `Opt`, NOT to `From File` — this export is deliberately ungated, unlike GE's 6–300. UNVERIFIED: we emit 3 spaces + CRLF while the README's only example is TAB-separated and LF-terminated; no source states a separator/EOL rule (the console is Windows, so CRLF is a safe guess, not a spec). Also per that README, custom files only work with Philips FiberTrak when the same directions are used for every b-value — this generator optimizes directions per shell, so output is FiberTrak-incompatible by construction (fine for offline pipelines).
- Scanner exports are examples and should be verified on the target platform before acquisition, especially the GE single-count-block convention.

## Residual limitations

- The vector optimizer is O(N²), bounded to 1000 directions, and runs in a worker. Further module splitting or optimizer replacement is not justified for typical schemes.
- niimath fitting cannot be interrupted mid-call; a superseded fit may consume CPU until return, after which `loadSeq` discards it.
- A single large tracking kernel completes before cancellation is observed. Finer cancellation requires smaller dispatches.
- Automated stress testing exercises batching and readback but may not force a true host OOM on high-memory hardware; OOM decision logic is covered separately in `tracking/backoff.test.ts`.
