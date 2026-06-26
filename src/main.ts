/**
 * dwi2trx — browser-only diffusion MRI pipeline (WASM + WebGPU; no data leaves
 * the machine). Tabbed wizard: 1 select/view a DWI, 2 fit the tensor + view the
 * FA-modulated V1 map (+ FA floor, Generate streamlines), 3 streamlines + Save
 * TRX. A tab unlocks once its inputs exist; dropping a new DWI relocks tabs 2–3.
 * Inspired by brain2print, but its own project.
 */

import NiiVueGPU, { SHOW_RENDER, SLICE_TYPE } from '@niivue/niivue'
import { cropFirstVolume, fitTensor } from './dwi2trx/dtifit'
import { collectFiles, type ResolvedInput, resolveInput } from './dwi2trx/input'
// mindgrab + conform are lazily imported on first "Mask + fit" (keeps the
// ~250 KB tinygrad model + gl-matrix out of the initial bundle, like
// dcm2niix/niimath). Only the type is imported eagerly (erased at build).
import type { MindgrabInferer } from './dwi2trx/mindgrab'
import {
  type InputSource,
  type Step,
  state,
  type TensorMaps,
} from './dwi2trx/state'
import { baseName } from './dwi2trx/validate'

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id)
  if (!el) throw new Error(`missing #${id}`)
  return el as T
}

const maskFitBtn = $<HTMLButtonElement>('maskFitBtn')
const chooseBtn = $<HTMLButtonElement>('chooseBtn')
const filePicker = $<HTMLInputElement>('filePicker')
const trackBtn = $<HTMLButtonElement>('trackBtn')
const saveBtn = $<HTMLButtonElement>('saveBtn')
const saveMapsBtn = $<HTMLButtonElement>('saveMapsBtn')
const fiberColor = $<HTMLSelectElement>('fiberColor')
const displayMode = $<HTMLSelectElement>('displayMode')
const sliceTypeSel = $<HTMLSelectElement>('sliceType')
const seedFaIn = $<HTMLInputElement>('seedFa')
const stopFaIn = $<HTMLInputElement>('stopFa')
const stepSizeIn = $<HTMLInputElement>('stepSize')
const maxAngleIn = $<HTMLInputElement>('maxAngle')
const seedDensityIn = $<HTMLInputElement>('seedDensity')
const aboutBtn = $<HTMLButtonElement>('aboutBtn')
const aboutDlg = $<HTMLDialogElement>('aboutDlg')
const faSlider = $<HTMLInputElement>('faSlider')
const statusEl = $<HTMLDivElement>('status')
const spinnerEl = $<HTMLSpanElement>('spinner')
const locationEl = $<HTMLDivElement>('location')
const dropOverlay = $<HTMLDivElement>('dropOverlay')
const tabEls = Array.from(document.querySelectorAll<HTMLButtonElement>('.tab'))

// Which volumes the canvas currently shows, so tab navigation only reloads
// when the view actually needs to change.
let shownView: 'input' | 'maps' | 'tracts' | null = null

// Monotonic load token: each drop/sample load claims a sequence number; a load
// (or the fit it feeds) only mutates the viewer/state if it's still the latest,
// so a slow async result can't clobber a newer input. `loadSeq` is input
// identity (bumped per new DWI). Canvas swaps are serialized separately via
// `viewChain` (see syncView) so overlapping loadVolumes can't fight.
let loadSeq = 0

function setStatus(msg: string, error = false): void {
  statusEl.textContent = msg
  statusEl.classList.toggle('error', error)
}

/** Show/hide the spinning busy indicator beside the status text during slow
 *  work (mindgrab, the dtifit fit, DICOM conversion). */
function busy(on: boolean): void {
  spinnerEl.classList.toggle('hidden', !on)
}

/** Read a numeric input, falling back to `def` (incl. for an empty/blank field —
 *  `valueAsNumber` is NaN there, unlike `Number('')` which is 0) and clamping. */
function num(
  el: HTMLInputElement,
  def: number,
  min: number,
  max: number,
): number {
  const v = el.valueAsNumber
  return Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : def
}

/** Reflect state in the tab bar: active tab highlighted; tabs gated by readiness. */
function render(): void {
  document.body.dataset.tab = String(state.step)
  maskFitBtn.disabled = !state.input
  saveMapsBtn.disabled = !state.maps
  saveBtn.disabled = !state.tracts
  for (const t of tabEls) {
    const n = Number(t.dataset.tab)
    t.disabled = (n === 2 && !state.maps) || (n === 3 && !state.tracts)
  }
}

/** Switch the active tab (caller triggers the matching view via syncView). */
function gotoTab(step: Step): void {
  state.step = step
  render()
}

for (const t of tabEls) {
  t.addEventListener('click', () => {
    if (t.disabled) return
    gotoTab(Number(t.dataset.tab) as Step)
    navSync()
  })
}

chooseBtn.addEventListener('click', () => filePicker.click())
maskFitBtn.addEventListener('click', () => {
  void runFit()
})
trackBtn.addEventListener('click', () => {
  void runTrack()
})
fiberColor.addEventListener('change', () => {
  void applyFiberColor() // self-guards on a loaded tract
})
displayMode.addEventListener('change', () => {
  void applyDisplayMode() // self-guards on the FA+V1 maps being loaded
})
sliceTypeSel.addEventListener('change', () => {
  nv.sliceType = Number(sliceTypeSel.value)
  nv.drawScene()
})
aboutBtn.addEventListener('click', () => aboutDlg.showModal())
function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  // Revoke on the next macrotask, not synchronously — Safari/WebKit can drop
  // the download if the blob URL is revoked before the fetch is queued (and a
  // "Save maps" click fires this twice back-to-back).
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
saveBtn.addEventListener('click', () => {
  if (state.tracts) download(state.tracts, state.tracts.name)
})
saveMapsBtn.addEventListener('click', () => {
  if (!state.maps || !state.input) return
  const base = baseName(state.input.nifti.name) || 'dti'
  download(state.maps.fa, `${base}_FA.nii.gz`)
  download(state.maps.v1, `${base}_V1.nii.gz`)
})
faSlider.addEventListener('input', () => {
  if (shownView === 'maps') setFaFloor() // ignore while the raw DWI is shown
})

// --- Drag & drop ---
// Prevent the browser's default "navigate to dropped file" on the WHOLE window —
// a drop anywhere outside the canvas would otherwise blow away the app.
window.addEventListener('dragover', (e) => e.preventDefault())
window.addEventListener('drop', (e) => e.preventDefault())

const main = $<HTMLElement>('canvas-container')
main.addEventListener('dragover', () => dropOverlay.classList.remove('hidden'))
main.addEventListener('dragleave', (e) => {
  // Only hide when the cursor actually leaves the container, not on child crossings.
  if (!main.contains(e.relatedTarget as Node))
    dropOverlay.classList.add('hidden')
})
main.addEventListener('drop', (e) => {
  void handleDrop(e as DragEvent)
})

function handleDrop(e: DragEvent): void {
  dropOverlay.classList.add('hidden')
  const dt = e.dataTransfer
  if (!dt) return
  // collectFiles must read dt.items synchronously (they expire after the event),
  // so call it now and let loadInputFiles await the result.
  void loadInputFiles(collectFiles(dt))
}

// File-picker fallback: drag-drop is finicky on touch / some Linux file
// managers / Safari folder-drop, so a plain <input type=file multiple> routes
// the same way (select a NIfTI+bval+bvec triple, or the DICOM files).
filePicker.addEventListener('change', () => {
  if (filePicker.files?.length) {
    void loadInputFiles(Promise.resolve(Array.from(filePicker.files)))
  }
  filePicker.value = '' // let the user re-pick the same files
})

/** Load a DWI from a promise of File[] (drop walk or file picker), through the
 *  same validate-then-display path. */
async function loadInputFiles(filesPromise: Promise<File[]>): Promise<void> {
  const seq = ++loadSeq
  // A new load relocks tabs 2–3 and invalidates any downstream progress.
  state.input = undefined
  state.maps = undefined
  state.tracts = undefined
  gotoTab(1)
  busy(true)
  setStatus('Loading…')
  try {
    const resolved = await resolveInput(await filesPromise)
    await loadInput(
      resolved,
      resolved.source,
      seq,
      resolved.source === 'dicom' ? 'DICOM → DWI' : 'DWI',
    )
  } catch (err) {
    if (seq === loadSeq) setStatus((err as Error).message, true)
  } finally {
    if (seq === loadSeq) busy(false)
  }
}

// --- NiiVue (WebGPU) ---
// Init inside try/catch: a browser without WebGPU throws here, and we want a
// clear message instead of a blank page with a stale "Loading…" status.
let nv: NiiVueGPU
try {
  nv = new NiiVueGPU({
    isDragDropEnabled: false, // we handle drops to drive the tabs
    backgroundColor: [0, 0, 0, 1],
    isSnapToVoxelCenters: true, // crisp V1 direction lines (per vox.modulate)
  })
  await nv.attachTo('gl1')
  nv.sliceType = SLICE_TYPE.MULTIPLANAR
  nv.showRender = SHOW_RENDER.AUTO
  // Live voxel readout in the footer (location + per-volume intensity).
  nv.addEventListener('locationChange', (loc) => {
    locationEl.textContent = (loc as { string?: string })?.string ?? ''
  })
} catch (err) {
  setStatus(
    `WebGPU unavailable — dwi2trx needs a recent desktop Chrome or Edge. (${(err as Error).message})`,
    true,
  )
  throw err
}

// Extension context for the mindgrab conform step (256³ FreeSurfer-canonical).
// The `conform` transform + the mindgrab model are registered/loaded lazily on
// first "Mask + fit" so the heavy code stays out of the initial bundle.
const maskCtx = nv.createExtensionContext()

// mindgrab device + model, loaded once on first "Mask + fit". `maskDevice` is
// undefined until tried, null if WebGPU/shader-f16 is unavailable.
let maskDevice: GPUDevice | null | undefined
let maskInferer: MindgrabInferer | null = null
let conformRegistered = false

async function getMaskInferer(): Promise<MindgrabInferer | null> {
  const { getBrainGPUDevice, loadMindgrab } = await import('./dwi2trx/mindgrab')
  if (maskDevice === undefined) maskDevice = await getBrainGPUDevice()
  if (!maskDevice) return null
  if (!conformRegistered) {
    const { conform } = await import('./niivue-ext/image-processing/transforms')
    maskCtx.registerVolumeTransform(conform)
    conformRegistered = true
  }
  if (!maskInferer) {
    maskInferer = await loadMindgrab(
      maskDevice,
      `${import.meta.env.BASE_URL}models/net_mindgrab.safetensors`,
    )
  }
  return maskInferer
}

/** Release mindgrab's ~1.4 GB GPU device + model buffers so the tracker can
 *  allocate its own (a second resident device risks OOM on single-GPU
 *  machines). Re-loaded lazily on the next "Mask + fit". `conformRegistered`
 *  stays true — the conform transform runs in a worker, not on this device. */
async function freeMaskGpu(): Promise<void> {
  if (maskInferer) {
    await maskInferer.dispose()
    maskInferer = null
  }
  if (maskDevice) maskDevice.destroy()
  maskDevice = undefined // re-request from getBrainGPUDevice() next time
}

/** Run mindgrab on the b0 → a brain mask in conformed space, or null if WebGPU
 *  is unavailable (caller fits unmasked). `seq` is the input identity this mask
 *  belongs to; we bail if a newer input arrives so a superseded mask neither
 *  wastes GPU work nor mutates the canvas under the new input. */
async function makeBrainMask(
  input: NonNullable<typeof state.input>,
  seq: number,
): Promise<File | null> {
  const inferer = await getMaskInferer()
  if (!inferer || seq !== loadSeq) return null
  setStatus('Brain extraction (mindgrab)…')
  const b0 = await cropFirstVolume(input)
  if (seq !== loadSeq) return null
  const { prepareInput, buildMaskNifti } = await import('./dwi2trx/mindgrab')
  // Load the transient b0 (just to get a parsed NVImage for conform) serialized
  // on the canvas chain, so it can't interleave with a navigation swap if a new
  // DWI is dropped mid-fit. Capture the NVImage inside the serialized step; the
  // reference stays valid even after a later swap replaces nv.volumes.
  const b0Img = await runOnCanvas(async () => {
    await nv.loadVolumes([{ url: b0, name: 'b0.nii.gz' }])
    shownView = null
    return nv.volumes[0]
  })
  if (seq !== loadSeq) return null // superseded during the transient load
  const { conformed, img32 } = await prepareInput(maskCtx, b0Img)
  const [labels] = await inferer(img32)
  return new File([buildMaskNifti(conformed, labels)], 'maskconf.nii')
}

// The input is already fully validated by resolveInput (volume count cross-
// checked against bval/bvec before we get here), so this just displays it.
async function loadInput(
  r: ResolvedInput,
  source: InputSource,
  seq: number,
  label: string,
): Promise<void> {
  if (seq !== loadSeq) return // superseded by a newer load
  state.input = { ...r, source }
  state.maps = undefined // new input invalidates any prior tensor fit
  state.tracts = undefined
  shownView = null // force the input view to (re)load through the chain
  gotoTab(1) // enables the Fit button, relocks tabs 2–3
  await syncView() // serialized canvas swap (not a bare nv.loadVolumes)
  if (seq !== loadSeq) return
  setStatus(
    `${label}: ${r.directions} volumes / directions. Press “Fit tensor”.`,
  )
}

// --- Stage 2: tensor fit + display ---

let fitting = false

async function runFit(): Promise<void> {
  const input = state.input
  if (!input || fitting) return // local guard: ignore a queued duplicate click
  const seq = loadSeq // the input identity this fit belongs to
  fitting = true
  maskFitBtn.disabled = true
  busy(true)
  setStatus('Fitting the diffusion tensor (niimath dtifit)…')
  try {
    // Brain-mask with mindgrab. Non-fatal: null = WebGPU can't run it, a throw =
    // a runtime mindgrab error — either way fall back to an unmasked fit rather
    // than failing the whole tensor fit.
    let maskConf: File | undefined
    let maskErr: string | undefined
    try {
      maskConf = (await makeBrainMask(input, seq)) ?? undefined
    } catch (err) {
      maskErr = (err as Error).message // a runtime failure (vs. null = no GPU)
      console.warn('[dwi2trx] mindgrab mask failed; fitting unmasked:', err)
    }
    if (seq !== loadSeq) return
    if (!maskConf) {
      setStatus(
        maskErr
          ? `Brain mask failed (${maskErr}) — fitting without a mask.`
          : 'Brain mask unavailable (needs a WebGPU GPU with shader-f16) — fitting without a mask.',
      )
    }
    const maps = await fitTensor(input, maskConf)
    if (seq !== loadSeq) return // a newer input superseded this fit — discard
    state.maps = maps
    state.tracts = undefined // a new fit invalidates the old TRX
    shownView = null // force showMaps to (re)load
    gotoTab(2) // unlock + switch to the Tensor maps tab
    await syncView()
    if (seq !== loadSeq) return // re-check: a new input may have arrived during the swap
    setStatus(
      `Tensor fit complete${maskConf ? ' (brain-masked)' : ''} — V1 modulated by FA.`,
    )
  } catch (err) {
    if (seq === loadSeq) {
      setStatus(`Tensor fit failed: ${(err as Error).message}`, true)
    }
  } finally {
    fitting = false
    maskFitBtn.disabled = !state.input
    if (seq === loadSeq) busy(false) // don't clear a newer load/fit's spinner
  }
}

// --- Stage 3: WebGPU streamline tracking (Boot/OPDT) ---

let tracking = false

/** Track streamlines on the GPU from the fitted FA/DWI, write a TRX, show it
 *  over the FA in a clipped 3D render, and unlock tab 3 + Save. Browser-only:
 *  needs WebGPU with `subgroups` (getTrackingDevice throws a clear reason if
 *  not). Seeds from FA ≥ 0.25, stops below FA 0.1. */
async function runTrack(): Promise<void> {
  const input = state.input
  const maps = state.maps
  if (!input || !maps || tracking) return
  const seq = loadSeq
  tracking = true
  trackBtn.disabled = true
  busy(true)
  setStatus('Generating streamlines (WebGPU)…')
  let device: GPUDevice | null = null
  try {
    const [
      { getTrackingDevice, trackStreamlines, DEFAULT_PARAMS, isOomError },
      inputsMod,
      { loadSphere },
      { writeTrx },
    ] = await Promise.all([
      import('./dwi2trx/tracking/tracker'),
      import('./dwi2trx/tracking/inputs'),
      import('./dwi2trx/tracking/sphere'),
      import('./dwi2trx/tracking/trx'),
    ])
    await freeMaskGpu() // free mindgrab's ~1.4 GB GPU before the tracker allocates
    device = await getTrackingDevice() // throws a specific reason if unsupported
    const [sphere, bvalText, bvecText] = await Promise.all([
      loadSphere(),
      input.bval.text(),
      input.bvec.text(),
    ])
    const {
      inputs: tInputs,
      voxelToRasmm,
      dims3,
    } = await inputsMod.assembleTrackingInputs(
      input.nifti,
      maps.fa,
      bvalText,
      bvecText,
      sphere,
      Math.min(
        device.limits.maxStorageBufferBindingSize,
        device.limits.maxBufferSize,
      ),
    )
    if (seq !== loadSeq) return
    // Tracking knobs from the UI (clamped to the input ranges).
    const MAX_SEEDS = 100000
    const seedFa = num(seedFaIn, 0.25, 0, 1)
    const density = Math.round(num(seedDensityIn, 1, 1, 4)) // whole seeds per axis
    const seeds = inputsMod.seedsFromMask(
      tInputs.metricMap,
      tInputs.dims,
      seedFa,
      density,
      MAX_SEEDS,
    )
    const nSeeds = seeds.length / 3
    if (nSeeds === 0) {
      setStatus(
        `No seed voxels with FA ≥ ${seedFa}. Lower the Seed FA threshold.`,
        true,
      )
      return
    }
    // The seed list is built in voxel order, so hitting the cap biases toward
    // one side of the brain — tell the user rather than silently truncating.
    const capped = nSeeds >= MAX_SEEDS
    const stepSize = num(stepSizeIn, 0.5, 0.1, 2)
    const params = {
      ...DEFAULT_PARAMS,
      tcThreshold: num(stopFaIn, 0.1, 0, 1),
      stepSize,
      maxAngle: (num(maxAngleIn, 60, 10, 90) * Math.PI) / 180,
      // Drop sub-streamline stubs: keep only tracts ≳ 5 voxels long (a 1-2 point
      // fragment is noise that bloats the TRX and clutters the render).
      minPts: Math.max(2, Math.ceil(5 / stepSize)),
    }
    const { lines, truncated, processedSeeds } = await trackStreamlines(
      device,
      tInputs,
      seeds,
      params,
      (done, total) => {
        if (seq === loadSeq)
          setStatus(
            `Tracking… ${done.toLocaleString()} / ${total.toLocaleString()} seeds`,
          )
      },
      () => seq !== loadSeq, // stop promptly if a new DWI was dropped mid-track
    )
    if (seq !== loadSeq) return
    if (lines.length === 0) {
      setStatus(
        'No streamlines survived — lower the Seed/Stop FA thresholds or the step size.',
        true,
      )
      return
    }
    const trxName = `${baseName(input.nifti.name) || 'streamlines'}.trx`
    const totalPts = lines.reduce((s, l) => s + l.length / 3, 0)
    const meanLen = (totalPts / lines.length).toFixed(1)
    const count = lines.length.toLocaleString()
    state.tracts = new File([writeTrx(lines, voxelToRasmm, dims3)], trxName)
    // Free the voxel-space lines now that the TRX is serialized — the 3D preview
    // below allocates a large cylinder mesh, and there is no need to hold both.
    lines.length = 0
    const note = truncated
      ? ' PARTIAL (out of memory) — raise the Seed/Stop FA thresholds or lower Density for the full set.'
      : capped
        ? ' Seeds capped at 100,000 — lower Density for full coverage.'
        : ''
    const seedNote = truncated
      ? `${processedSeeds.toLocaleString()} of ${nSeeds.toLocaleString()} seeds`
      : `${nSeeds.toLocaleString()} seeds`
    const summary =
      `${count} streamlines from ${seedNote} ` +
      `(mean ${meanLen} pts) — saved as TRX.${note}`
    gotoTab(3)
    shownView = null // force the tract render to load
    // The TRX is already built and saveable. The 3D preview is separate: NiiVue
    // turns every streamline into a cylinder mesh (millions of vertices), which
    // can exhaust memory on a big tractogram even though the tracking itself
    // succeeded. Catch that so a render OOM doesn't masquerade as a tracking
    // failure and the user can still download their TRX.
    try {
      await syncView()
      if (seq !== loadSeq) return
      setStatus(summary)
    } catch (renderErr) {
      if (seq !== loadSeq) return
      console.warn('[dwi2trx] tract render failed:', renderErr)
      render() // keep “Save TRX” enabled (state.tracts is set)
      // Distinguish an out-of-memory preview (the expected failure on a huge
      // tractogram) from any other render error, so a real bug isn't mislabeled
      // as OOM. Either way the TRX is already built and downloadable.
      const msg = (renderErr as Error)?.message ?? String(renderErr)
      const oom = isOomError(renderErr)
      setStatus(
        oom
          ? `${summary} The 3D preview ran out of memory — click “Save TRX” to download it, or raise the Seed/Stop FA thresholds to render fewer streamlines.`
          : `${summary} The 3D preview failed (${msg}) — your TRX is saved; click “Save TRX” to download it.`,
        true,
      )
    }
  } catch (err) {
    if (seq === loadSeq)
      setStatus(
        `Streamline tracking failed: ${(err as Error).message} ` +
          '(hint: raise the Seed and Stop FA thresholds, or lower Density/step size, to use less memory).',
        true,
      )
  } finally {
    device?.destroy() // free the WebGPU device on every path (incl. errors)
    tracking = false
    trackBtn.disabled = false
    if (seq === loadSeq) busy(false) // don't clear a newer load/fit's spinner
  }
}

// Canvas swaps run one-at-a-time through this chain so two overlapping
// nv.loadVolumes() calls (rapid Back/Next, or a fit completing mid-nav) can't
// leave shownView disagreeing with what's actually on the canvas.
let viewChain: Promise<void> = Promise.resolve()

/** Load the volumes the current step should show, serialized + skipping redundant reloads. */
function syncView(): Promise<void> {
  viewChain = viewChain.then(doSyncView, doSyncView)
  return viewChain
}

/** Run a canvas operation serialized on the same chain as syncView, so a
 *  transient load (e.g. the mindgrab b0) can't interleave with a navigation
 *  swap and corrupt nv.volumes. Returns the op's result. */
function runOnCanvas<T>(fn: () => Promise<T>): Promise<T> {
  const run = viewChain.then(fn, fn)
  viewChain = run.then(
    () => {},
    () => {},
  )
  return run
}

/** Fire-and-forget syncView for navigation, surfacing any display error. */
function navSync(): void {
  syncView().catch((err) =>
    setStatus(`Display failed: ${(err as Error).message}`, true),
  )
}

async function doSyncView(): Promise<void> {
  // A new input bumps loadSeq; a swap that finishes after that must NOT record
  // its (now-stale) view, or the string dedup would skip the new input's load.
  const seq = loadSeq
  if (state.step >= 3 && state.tracts) {
    if (shownView === 'tracts') return
    await showTracts(state.tracts)
    if (seq !== loadSeq) return
    shownView = 'tracts'
  } else if (state.step >= 2 && state.maps) {
    if (shownView === 'maps') return
    await clearTractScene()
    await showMaps(state.maps)
    if (seq !== loadSeq) return
    shownView = 'maps'
  } else if (state.input) {
    if (shownView === 'input') return
    await clearTractScene()
    await nv.loadVolumes([
      { url: state.input.nifti, name: state.input.nifti.name },
    ])
    if (seq !== loadSeq) return
    shownView = 'input'
  }
}

/** Tear down the 3D tract render (meshes, clip planes, render view) when
 *  returning to the 2D slice views. */
async function clearTractScene(): Promise<void> {
  if (nv.meshes.length) await nv.removeAllMeshes()
  nv.setClipPlanes([]) // slice type stays under the global View dropdown
}

/**
 * Tab 3: the tracked streamlines (TRX) over the FA volume, in a clipped 3D
 * render with direction-encoded colour — modelled on niivue's tract.groups
 * demo (clip planes + volume illumination so the FA slices show through).
 */
async function showTracts(trx: File): Promise<void> {
  if (state.maps) {
    await nv.loadVolumes([
      { url: state.maps.fa, name: 'FA.nii.gz', opacity: 1 },
    ])
  }
  nv.volumeIsV1SliceShader = false // plain FA backdrop, not the V1 colour shader
  await nv.loadMeshes([
    { url: trx, name: 'streamlines.trx', rgba255: TRACT_RGBA },
  ])
  if (nv.meshes.length) await applyFiberColor()
  nv.setClipPlanes([
    [0.1, 180, 20],
    [0.1, 0, -20],
  ])
  nv.volumeIllumination = 0.5 // slice type is the global View dropdown's choice
}

const TRACT_RGBA: [number, number, number, number] = [0, 142, 200, 255]

/** Apply the "Fiber color" dropdown to the loaded tract: local/global direction
 *  colouring, or a fixed colour. */
async function applyFiberColor(): Promise<void> {
  if (!nv.meshes.length) return // no tract loaded yet
  const mode = fiberColor.value
  await nv.setTractOptions(
    0,
    mode === 'fixed'
      ? { colorBy: 'fixed', fixedColor: TRACT_RGBA }
      : { colorBy: mode },
  )
}

/**
 * Display the principal eigenvector V1 as directionally-encoded colour,
 * modulated by FA — niivue's vox.modulate "V1 modulated by FA (isV1SliceShader)"
 * mode. Volume ORDER matters: FA must be volume 0, V1 (the 3-frame vector) must
 * be volume 1, both opacity 1. `volumeIsV1SliceShader` is what renders V1 as
 * colour rather than a grayscale 4D scalar.
 */
async function showMaps(maps: TensorMaps): Promise<void> {
  await nv.loadVolumes([
    { url: maps.fa, name: 'FA.nii.gz', opacity: 1 },
    { url: maps.v1, name: 'V1.nii.gz', opacity: 1 },
  ])
  nv.volumeIsNearestInterpolation = true // crisp V1 direction lines
  await applyDisplayMode() // honour the "Display" dropdown
  setFaFloor() // apply the initial slider value (volumes are the maps here)
}

/**
 * Apply the "Display" dropdown (FA / V1 / V1×FA / isV1SliceShader variants),
 * ported from niivue's vox.modulate demo. Operates on the loaded FA (volume 0)
 * + V1 (volume 1); self-guards so it's a no-op outside the maps view.
 */
async function applyDisplayMode(): Promise<void> {
  if (nv.volumes.length < 2) return // only meaningful for the FA+V1 maps
  const idx = Number(displayMode.value)
  const fa = nv.volumes[0]
  const v1 = nv.volumes[1]
  fa.opacity = idx === 0 || idx > 2 ? 1 : 0
  v1.opacity = idx === 0 ? 0 : 1
  const modulate = idx === 2 || idx === 4
  await nv.setModulationImage(v1.id ?? '', modulate ? (fa.id ?? '') : '')
  nv.volumeIsV1SliceShader = idx > 2
  nv.updateGLVolume()
}

/** Slider [0..100] → FA floor [0..1] on volume 0 (FA): hides low-anisotropy voxels. */
function setFaFloor(): void {
  const fa = nv.volumes[0]
  if (!fa) return
  fa.calMin = Number(faSlider.value) / 100
  fa.calMax = 1
  nv.updateGLVolume()
}

// --- Default sample (validated through the same path as a user drop) ---
async function fetchAsFile(url: string): Promise<File> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not fetch ${url} (${res.status}).`)
  return new File([await res.blob()], url.split('/').pop() ?? 'file')
}

async function loadSample(): Promise<void> {
  const seq = ++loadSeq
  busy(true)
  setStatus('Loading sample…')
  try {
    const base = import.meta.env.BASE_URL
    const [nii, bval, bvec] = await Promise.all([
      fetchAsFile(`${base}dwi.nii.gz`),
      fetchAsFile(`${base}dwi.bval`),
      fetchAsFile(`${base}dwi.bvec`),
    ])
    const resolved = await resolveInput([nii, bval, bvec])
    await loadInput(resolved, 'sample', seq, 'Sample DWI')
  } catch (err) {
    if (seq === loadSeq)
      setStatus(`Failed to load sample: ${(err as Error).message}`, true)
  } finally {
    if (seq === loadSeq) busy(false)
  }
}

await loadSample()
render()
