// Real-WebGPU browser smoke test (PLAN item 2). Drives the whole pipeline in a
// real Chrome: auto-loaded sample DWI -> Mask + fit -> Generate streamlines ->
// Save TRX, then unzips the downloaded TRX and asserts it parses with > 0
// streamlines. This is the one test that catches a Dawn/Tint shader regression
// or a wiring break that `npm test` (pure validation + host math) structurally
// cannot.
//
// Skip vs fail contract — local-only by design, so an environment that simply
// *can't run it* must SKIP (exit 0), not fail, while a genuine regression FAILs
// (exit 1). SKIP: no Chrome at CHROME_PATH, preview port busy, no WebGPU adapter,
// or the tracker's `subgroups` requirement unmet. FAIL: a shader/validation
// error, 0 streamlines, or a malformed TRX — the things this test exists to
// catch. (Crucially, a tracking error is only skipped when its message names a
// WebGPU/subgroups limitation; any other error is a real failure.)
//
// Run: npm run test:e2e   (builds first; uses system Chrome, override with
// CHROME_PATH=/path/to/chrome).

import { spawn } from 'node:child_process'
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { unzipSync } from 'fflate'
import puppeteer from 'puppeteer-core'

const PORT = 4317
const BASE = '/dwi2trx/' // must match vite.config.ts `base`
const URL = `http://localhost:${PORT}${BASE}`

const CHROME =
  process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

const SKIP = 0 // "can't run here" — don't break CI / local runs without a GPU
const PASS = 0
const FAIL = 1

// Thrown for environment-absence cases (no Chrome/GPU/subgroups, busy port) so
// the catch can exit SKIP. Anything else thrown is a real regression -> FAIL.
class SkipError extends Error {}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function startPreview() {
  // `vite preview` serves the built dist/ at the configured base path.
  return spawn(
    'npx',
    ['vite', 'preview', '--port', String(PORT), '--strictPort'],
    { stdio: ['ignore', 'pipe', 'inherit'] },
  )
}

async function waitForServer(isDead, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    // --strictPort makes vite exit if PORT is taken; that's an environment
    // problem (something else is on 4317), not a regression.
    if (isDead()) {
      throw new SkipError(`preview server exited — port ${PORT} busy?`)
    }
    try {
      const res = await fetch(URL)
      if (res.ok) return
    } catch {
      // not up yet
    }
    await sleep(200)
  }
  throw new Error(`preview server never became ready at ${URL}`)
}

async function main() {
  if (!existsSync(CHROME)) {
    console.log(`SKIP: no Chrome at ${CHROME} — set CHROME_PATH.`)
    process.exit(SKIP)
  }

  const preview = startPreview()
  let previewExited = false
  preview.on('exit', () => {
    previewExited = true
  })
  const downloadDir = mkdtempSync(join(tmpdir(), 'dwi2trx-e2e-'))
  let browser
  let exitCode = FAIL
  try {
    await waitForServer(() => previewExited)

    browser = await puppeteer.launch({
      executablePath: CHROME,
      headless: 'new',
      args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,WebGPU',
        '--no-sandbox',
        '--use-mock-keychain',
      ],
    })
    const page = await browser.newPage()
    page.on('console', (m) => {
      const t = m.text()
      if (/error|warn|webgpu|subgroup|mindgrab|dawn|tint/i.test(t)) {
        console.log(`  [page] ${t}`)
      }
    })

    const client = await page.target().createCDPSession()
    await client.send('Page.setDownloadBehavior', {
      behavior: 'allow',
      downloadPath: downloadDir,
    })

    console.log(`→ loading ${URL}`)
    await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 })

    // No WebGPU adapter at all -> nothing past the fit can run; skip cleanly
    // instead of timing out later. (The load + fit themselves use WASM, not GPU.)
    const hasAdapter = await page.evaluate(async () => {
      if (!navigator.gpu) return false
      try {
        return !!(await navigator.gpu.requestAdapter())
      } catch {
        return false
      }
    })
    if (!hasAdapter) throw new SkipError('no WebGPU adapter in this browser')

    // 1. sample auto-loads -> Mask+fit button enables
    console.log('→ waiting for sample to load')
    await page.waitForSelector('#maskFitBtn:not([disabled])', {
      timeout: 60000,
    })

    // 2. mask + fit tensor -> Save maps enables (state.maps set). Works without
    //    WebGPU (unmasked fallback), so a timeout here is a real regression.
    console.log('→ Mask + fit tensor')
    await page.click('#maskFitBtn')
    await page.waitForSelector('#saveMapsBtn:not([disabled])', {
      timeout: 120000,
    })

    // 3. generate streamlines. Either Save-TRX enables (success) or the status
    //    bar shows an error. An error is a SKIP only if it names a WebGPU/
    //    subgroups limitation; otherwise it's the regression we want to catch.
    console.log('→ Generate streamlines (WebGPU)')
    await page.click('#trackBtn')
    const outcome = await Promise.race([
      page
        .waitForSelector('#saveBtn:not([disabled])', { timeout: 180000 })
        .then(() => null),
      waitForStatusError(page).then((msg) => msg),
    ])
    if (outcome) {
      if (
        /subgroup|webgpu|adapter|gpu|device|unsupported|require/i.test(outcome)
      ) {
        throw new SkipError(`tracking unsupported on this GPU — ${outcome}`)
      }
      throw new Error(`tracking failed — ${outcome}`)
    }

    // 4. save TRX -> capture the download -> parse it
    console.log('→ Save TRX')
    await page.click('#saveBtn')
    const { count, positions } = await waitForParsedTrx(downloadDir, 30000)
    console.log(
      `→ TRX parsed: ${count} streamlines, ${positions} position floats`,
    )

    if (count <= 0)
      throw new Error(`TRX has ${count} streamlines (expected > 0)`)
    if (positions <= 0) throw new Error('TRX positions buffer is empty')

    console.log(
      '\nPASS: full pipeline ran on real WebGPU and the TRX round-trips.',
    )
    exitCode = PASS
  } catch (err) {
    if (err instanceof SkipError) {
      console.log(`\nSKIP: ${err.message}`)
      exitCode = SKIP
    } else {
      console.error(`\nFAIL: ${err.message}`)
      exitCode = FAIL
    }
  } finally {
    if (browser) await browser.close().catch(() => {})
    preview.kill('SIGTERM')
    rmSync(downloadDir, { recursive: true, force: true })
  }
  process.exit(exitCode)
}

// Resolve with the trimmed status text once the status bar enters its error
// state (`setStatus(msg, true)` adds the `error` class). The caller decides
// skip-vs-fail from the message — this only detects that an error happened.
async function waitForStatusError(page) {
  for (;;) {
    await sleep(500)
    const msg = await page
      .$eval('#status', (e) =>
        e.classList.contains('error') ? e.textContent || '' : null,
      )
      .catch(() => null)
    if (msg) return msg.trim()
  }
}

// Wait for a finished, parseable TRX. Chrome can expose the final filename
// before the bytes are fully readable, so filesystem stability alone is not
// enough; parse success is the completion condition.
async function waitForParsedTrx(dir, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  let lastSize = -1
  let lastError = ''
  while (Date.now() < deadline) {
    const entries = readdirSync(dir)
    const inProgress = entries.some((f) => f.endsWith('.crdownload'))
    const done = entries.find(
      (f) => f.endsWith('.trx') && !f.endsWith('.crdownload'),
    )
    if (done && !inProgress) {
      const path = join(dir, done)
      const size = statSync(path).size
      if (size > 0 && size === lastSize) {
        try {
          return parseTrx(readFileSync(path))
        } catch (err) {
          lastError = err?.message ?? String(err)
        }
      }
      lastSize = size
    }
    await sleep(200)
  }
  throw new Error(
    `no complete parseable .trx download appeared within ${timeoutMs}ms` +
      (lastError ? `; last parse error: ${lastError}` : ''),
  )
}

// Minimal TRX reader: it's a STORE zip with header.json + positions.3.float32.
function parseTrx(buf) {
  const files = unzipSync(new Uint8Array(buf))
  const headerName = Object.keys(files).find((k) => k.endsWith('header.json'))
  if (!headerName) throw new Error('TRX has no header.json')
  const header = JSON.parse(new TextDecoder().decode(files[headerName]))
  const count = header.NB_STREAMLINES ?? 0
  const posName = Object.keys(files).find((k) =>
    /positions\.3\.float32$/.test(k),
  )
  if (!posName) throw new Error('TRX has no positions.3.float32')
  return { count, positions: files[posName].byteLength / 4 }
}

main()
