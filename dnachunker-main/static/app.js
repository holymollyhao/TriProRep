// ===== API endpoint =====
// On GitHub Pages this is a different origin; the backend (FastAPI in server/)
// must be reachable at this URL with CORS allowing the Pages origin.
// For local dev with `cd server && uvicorn app:app`, leave API_BASE = "" so
// fetch() uses the same origin that served the page.
const API_BASE =
  window.location.hostname.endsWith("github.io")
    ? "https://nyc-nine-marine-match.trycloudflare.com"   // Cloudflare quick tunnel (ephemeral)
    : "";

const SAMPLE_BASES = "ACGT";

function randomSample(len = 256) {
  let out = "";
  for (let i = 0; i < len; i++) {
    out += SAMPLE_BASES[Math.floor(Math.random() * 4)];
  }
  return out;
}

const MAX_LEN = 8192;
const VIZ_CAP = 256;          // bp shown in the animated visualization
const BASE_STAGGER_MS = 8;    // per-base delay for the base-pill pop-in
const S1_STAGGER_MS = 20;
const S2_STAGGER_MS = 40;
const PHASE_GAP_MS = 350;     // gap between phases (bases -> S1 -> S2)

const seqEl = document.getElementById("seq");
const runBtn = document.getElementById("run-btn");
const sampleBtn = document.getElementById("sample-btn");
const replayBtn = document.getElementById("replay-btn");
const staticBtn = document.getElementById("static-btn");
const lengthHint = document.getElementById("length-hint");
const status = document.getElementById("status");
const resultsEl = document.getElementById("results");
const errorEl = document.getElementById("error");
const metricsEl = document.getElementById("metrics");
const vizEl = document.getElementById("viz");
const capNotice = document.getElementById("cap-notice");

let lastResult = null;
let staticMode = false;

function sanitize(s) {
  return s.toUpperCase().replace(/\s+/g, "").replace(/[^ACGTN]/g, "N").slice(0, MAX_LEN);
}

function updateLengthHint() {
  const clean = sanitize(seqEl.value);
  lengthHint.textContent = `${clean.length.toLocaleString()} bp` + (clean.length === MAX_LEN ? " (capped)" : "");
}

seqEl.addEventListener("input", updateLengthHint);
sampleBtn.addEventListener("click", () => {
  seqEl.value = randomSample(256);
  updateLengthHint();
});
replayBtn.addEventListener("click", () => {
  if (lastResult) {
    staticMode = false;
    drawViz(lastResult);
  }
});
staticBtn.addEventListener("click", () => {
  if (lastResult) {
    staticMode = !staticMode;
    staticBtn.textContent = staticMode ? "Animated view" : "Static view";
    drawViz(lastResult);
  }
});

async function pingHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) throw new Error();
    const j = await r.json();
    status.textContent = `model ready · ${j.device} · ${j.dtype}`;
    status.classList.add("ok");
  } catch (e) {
    status.textContent = "model failed to load";
    status.classList.add("err");
  }
}
pingHealth();

runBtn.addEventListener("click", async () => {
  const sequence = sanitize(seqEl.value);
  errorEl.hidden = true;
  if (!sequence) {
    errorEl.hidden = false;
    errorEl.textContent = "Please enter a DNA sequence.";
    return;
  }
  runBtn.disabled = true;
  runBtn.textContent = "Chunking…";
  try {
    const r = await fetch(`${API_BASE}/api/chunk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`;
      throw new Error(detail);
    }
    const data = await r.json();
    lastResult = data;
    render(data);
  } catch (e) {
    errorEl.hidden = false;
    errorEl.textContent = `Inference failed: ${e.message}`;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Chunk it";
  }
});

function metric(label, value, sub) {
  const el = document.createElement("div");
  el.className = "metric";
  el.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>` +
    (sub ? `<div class="sub">${sub}</div>` : "");
  return el;
}

function render(data) {
  resultsEl.hidden = false;
  metricsEl.innerHTML = "";

  const cr1 = (data.compression_ratio_stage1 * 100).toFixed(1);
  const cr2 = (data.compression_ratio_stage2 * 100).toFixed(1);
  const total = (data.compression_ratio_stage1 * data.compression_ratio_stage2 * 100).toFixed(1);

  metricsEl.appendChild(metric("Input length", `${data.length.toLocaleString()} bp`));
  metricsEl.appendChild(metric("Stage-1 chunks", data.n_chunks_stage1.toLocaleString(), `${cr1}% of bases`));
  metricsEl.appendChild(metric("Stage-2 chunks", data.n_chunks_stage2.toLocaleString(), `${cr2}% of S1 chunks`));
  metricsEl.appendChild(metric("Total compression", `${total}%`, "of original length"));

  drawViz(data);
}

function drawViz(data) {
  vizEl.innerHTML = "";

  const total = data.length;
  const shown = Math.min(total, VIZ_CAP);
  capNotice.hidden = (total <= VIZ_CAP);
  capNotice.textContent = `Visualizing first ${shown} bp of ${total} bp (animation kept readable). ` +
    `Inference and metrics are computed over the full input.`;

  // Slice all per-base arrays to the visible portion.
  const bases = data.bases.slice(0, shown);
  const preds = data.predicted_bases.slice(0, shown);
  const s1ids = data.stage1_chunk_id.slice(0, shown);
  const s2ids = data.stage2_chunk_id.slice(0, shown);
  const p1    = data.p_stage1.slice(0, shown);
  const p2    = data.p_stage2.slice(0, shown);

  // Build chunk spans (contiguous runs of identical chunk_id) for S1 and S2.
  const s1chunks = buildChunkSpans(s1ids, p1);
  const s2chunks = buildChunkSpans(s2ids, p2);

  // Pick a cell width so the whole shown range tries to fit in the viewport.
  const desiredW = Math.min(Math.max(window.innerWidth - 140, 600), 1400);
  const cellW = Math.max(12, Math.min(28, Math.floor(desiredW / shown)));
  vizEl.style.setProperty("--cell-w", `${cellW}px`);

  const grid = document.createElement("div");
  grid.className = "viz-grid" + (staticMode ? " static" : "");
  grid.style.gridTemplateColumns = `repeat(${shown}, var(--cell-w))`;
  grid.style.minWidth = `${shown * cellW + 16}px`;

  // ---- Track 1: bases ----
  const baseLabel = document.createElement("div");
  baseLabel.className = "track-label";
  baseLabel.textContent = "Input DNA";
  grid.appendChild(baseLabel);

  const baseAnimEndMs = shown * BASE_STAGGER_MS + 380;
  for (let i = 0; i < shown; i++) {
    const cell = document.createElement("div");
    cell.className = `base-pill c${i % 4}`;
    cell.style.gridColumn = `${i + 1} / span 1`;
    cell.style.setProperty("--i", i);
    cell.textContent = cellW >= 14 ? bases[i] : "";
    cell.title = `pos ${i} · base ${bases[i]} · S1 chunk ${s1ids[i]} · prediction ${preds[i]}`;
    grid.appendChild(cell);
  }

  // ---- Track 2: Stage 1 chunks ----
  const s1Label = document.createElement("div");
  s1Label.className = "track-label";
  s1Label.textContent = `Stage 1 chunks · router #1 (${s1chunks.length} shown of ${data.n_chunks_stage1})`;
  grid.appendChild(s1Label);

  const s1BaseDelay = staticMode ? 0 : (baseAnimEndMs + PHASE_GAP_MS);
  s1chunks.forEach((c, idx) => {
    const el = document.createElement("div");
    el.className = `chunk-box c${idx % 4}`;
    el.style.gridColumn = `${c.start + 1} / span ${c.span}`;
    el.style.animationDelay = `${s1BaseDelay + idx * S1_STAGGER_MS}ms`;
    el.title = `S1 chunk ${idx} · ${c.span} bp · p_boundary≈${c.prob.toFixed(3)}`;
    const label = bases.slice(c.start, c.start + c.span).join("");
    el.textContent = (c.span * cellW >= label.length * 7 + 8) ? label : "";
    if (!el.textContent) el.classList.add("compact");
    grid.appendChild(el);
  });

  // ---- Track 3: Stage 2 chunks ----
  const s2Label = document.createElement("div");
  s2Label.className = "track-label";
  s2Label.textContent = `Stage 2 chunks · router #2 (${s2chunks.length} shown of ${data.n_chunks_stage2})`;
  grid.appendChild(s2Label);

  const s1AnimEndMs = staticMode ? 0 : (s1BaseDelay + s1chunks.length * S1_STAGGER_MS + 420);
  const s2BaseDelay = staticMode ? 0 : (s1AnimEndMs + PHASE_GAP_MS);
  s2chunks.forEach((c, idx) => {
    const el = document.createElement("div");
    el.className = `chunk-box c${idx % 4}`;
    el.style.gridColumn = `${c.start + 1} / span ${c.span}`;
    el.style.animationDelay = `${s2BaseDelay + idx * S2_STAGGER_MS}ms`;
    el.title = `S2 chunk ${idx} · ${c.span} bp · p_boundary≈${c.prob.toFixed(3)}`;
    const label = bases.slice(c.start, c.start + c.span).join("");
    el.textContent = (c.span * cellW >= label.length * 7 + 8) ? label : "";
    if (!el.textContent) el.classList.add("compact");
    grid.appendChild(el);
  });

  vizEl.appendChild(grid);
}

// Build {start, span, prob} runs from a per-position chunk_id array.
function buildChunkSpans(chunkIds, probs) {
  const out = [];
  if (chunkIds.length === 0) return out;
  let start = 0;
  for (let i = 1; i <= chunkIds.length; i++) {
    if (i === chunkIds.length || chunkIds[i] !== chunkIds[start]) {
      out.push({ start, span: i - start, prob: probs[start] });
      start = i;
    }
  }
  return out;
}

updateLengthHint();

// ===== Benchmark tab toggle =====
document.querySelectorAll(".benchmark-tabs").forEach((container) => {
  const buttons = container.querySelectorAll(".tab-btn");
  const panels = container.querySelectorAll(".tab-panel");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      buttons.forEach((b) => b.classList.toggle("active", b === btn));
      panels.forEach((p) => p.classList.toggle("active", p.dataset.tab === target));
    });
  });
});
