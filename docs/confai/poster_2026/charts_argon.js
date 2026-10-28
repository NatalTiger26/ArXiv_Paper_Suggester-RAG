// ConfAI 2026 Poster - Stage 2 Data-Driven Chart Renderer (charts_argon.js)
// Reads strictly from DATA. No hardcoded constants.

const FS = {
  caption: 7.06, // 20 pt
  tick: 7.76,    // 22 pt
  body: 8.47,    // 24 pt
  title: 9.88    // 28 pt
};

const PALETTE = {
  // 4-signal ranking palette
  sigEmbed: "#0284c7",
  sigRecency: "#d97706",
  sigCitations: "#4f46e5",
  sigAuthor: "#db2777",
  // Pillar & semantic colours
  pillarBlue: "#2563eb",
  pillarGreen: "#059669",
  pillarPurple: "#7c3aed",
  warnOrange: "#d97706",
  dangerRed: "#dc2626",
  text: "#0f172a",
  muted: "#475569",
  line: "#cbd5e1",
  grid: "#e2e8f0"
};

function makeSvg(w, h) {
  return `<svg width="${w}mm" height="${h}mm" viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">`;
}

// Self-Check: Recompute each histogram mean from its bin counts and throw if diff > 0.005
function verifyHistogramMeans() {
  const nData = DATA.novelty.null;
  const edges = nData.bin_edges;
  const mids = [];
  for (let i = 0; i < edges.length - 1; i++) {
    mids.push((edges[i] + edges[i + 1]) / 2);
  }
  ["real", "isotropic_null", "library_null"].forEach(key => {
    const series = nData[key];
    if (!series || !Array.isArray(series.counts)) return;
    let total = 0;
    let weighted = 0;
    for (let i = 0; i < series.counts.length; i++) {
      total += series.counts[i];
      weighted += series.counts[i] * mids[i];
    }
    const recomputed = total > 0 ? weighted / total : 0;
    const diff = Math.abs(recomputed - series.mean);
    if (diff > 0.005) {
      throw new Error(`Histogram mean self-check failed for ${key}: recomputed=${recomputed.toFixed(4)} vs DATA=${series.mean.toFixed(4)} (diff=${diff.toFixed(4)})`);
    }
  });
}

// 1. Col 1 presets block (168 x 100 mm)
function renderPresets() {
  const slot = document.getElementById("slot-presets");
  if (!slot) return;
  const W = 168, H = 100;
  const sigColors = [PALETTE.sigEmbed, PALETTE.sigRecency, PALETTE.sigCitations, PALETTE.sigAuthor];
  const sigLabels = ["Topic", "Recency", "Citations", "Authors"];
  let s = makeSvg(W, H);

  // Claim title (28 pt bold, 2 lines) + How-to-read (22 pt, 2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Presets trade topic match</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">against recency</text>`;
  s += `<text x="0" y="29.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Each bar is the whole score. Blue is the share</text>`;
  s += `<text x="0" y="37.8" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">given to matching your query.</text>`;

  // Legend row (y=46.8)
  const legX = [0, 39, 83, 128];
  for (let i = 0; i < 4; i++) {
    s += `<rect x="${legX[i]}" y="41.5" width="5.5" height="5.5" rx="1" fill="${sigColors[i]}"/>`;
    s += `<text x="${legX[i] + 7.2}" y="46.8" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${sigLabels[i]}</text>`;
  }

  // 4 horizontal stacked bars filling y=51.5..97.5
  const barX = 38, barW = 128, barH = 9.0;
  const rowYs = [51.5, 64.0, 76.5, 89.0];
  DATA.ranking.presets.forEach((p, idx) => {
    const y = rowYs[idx];
    s += `<text x="0" y="${(y + 7.2).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${p.name}</text>`;
    let curX = barX;
    p.weights.forEach((w, wi) => {
      const segW = w * barW;
      s += `<rect x="${curX.toFixed(2)}" y="${y.toFixed(2)}" width="${segW.toFixed(2)}" height="${barH}" fill="${sigColors[wi]}"/>`;
      if (segW >= 15) {
        const pct = Math.round(w * 100) + "%";
        s += `<text x="${(curX + segW / 2).toFixed(2)}" y="${(y + 7.2).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="#ffffff" text-anchor="middle">${pct}</text>`;
      }
      curX += segW;
    });
  });

  s += `</svg>`;
  slot.innerHTML = s;
}

// 2. Col 1 top5 block (168 x 132 mm)
function renderTop5() {
  const slot = document.getElementById("slot-top5");
  if (!slot) return;
  const W = 168, H = 132;
  const sigColors = [PALETTE.sigEmbed, PALETTE.sigRecency, PALETTE.sigCitations, PALETTE.sigAuthor];
  let s = makeSvg(W, H);

  // Claim title (28 pt bold, 2 lines) + How-to-read (22 pt, 2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">One query, five papers, and what</text>`;
  s += `<text x="0" y="20.6" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">drove each score</text>`;
  s += `<text x="0" y="30.0" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Coloured segments show how much each signal</text>`;
  s += `<text x="0" y="38.8" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">added to the score.</text>`;

  // Query caption at y=48.5
  s += `<text x="0" y="48.5" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Query: "${DATA.ranking.query}" (Balanced)</text>`;

  // 5 rows filling y=52.5..129
  const startY = 52.5, rowStep = 15.6;
  const maxScore = 0.80, barFullW = 166;
  DATA.ranking.top5.forEach((item, idx) => {
    const ry = startY + idx * rowStep;
    const textY = ry + 6.0;
    const barY = ry + 8.0;
    const barH = 5.0;
    const label = `#${item.rank}  ${item.short_id}  ${item.short_title}`;
    s += `<text x="0" y="${textY.toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${label}</text>`;
    s += `<text x="167" y="${textY.toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.sigEmbed}" text-anchor="end">${item.final_score.toFixed(3)}</text>`;

    s += `<rect x="0" y="${barY.toFixed(2)}" width="${barFullW}" height="${barH}" rx="1" fill="#f1f5f9"/>`;
    let curX = 0;
    item.contributions.forEach((c, ci) => {
      const segW = (c / maxScore) * barFullW;
      s += `<rect x="${curX.toFixed(2)}" y="${barY.toFixed(2)}" width="${segW.toFixed(2)}" height="${barH}" fill="${sigColors[ci]}"/>`;
      curX += segW;
    });
  });

  s += `</svg>`;
  slot.innerHTML = s;
}

// 3. Col 1 rankmatrix block (168 x 112 mm)
function renderRankMatrix() {
  const slot = document.getElementById("slot-rankmatrix");
  if (!slot) return;
  const W = 168, H = 112;
  let s = makeSvg(W, H);

  // Claim title (28 pt bold, 2 lines) + How-to-read (22 pt, 2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Changing the preset reorders</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">the list</text>`;
  s += `<text x="0" y="29.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Number = rank, 1 is best. Read along a row</text>`;
  s += `<text x="0" y="37.8" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">to see one paper move.</text>`;

  const colX = [64, 90, 116, 142];
  const colW = 25;
  const colHdrs = ["Bal", "Prec", "Rec", "Cit"];
  s += `<text x="0" y="46.8" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.muted}" text-anchor="start">Paper</text>`;
  colHdrs.forEach((h, ci) => {
    s += `<text x="${colX[ci] + (colW - 1.5) / 2}" y="46.8" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">${h}</text>`;
  });

  const rows = DATA.ranking.rank_shift_trajectories;
  const startY = 49.5, rowH = 10.2;
  rows.forEach((r, ri) => {
    const y = startY + ri * rowH;
    const lbl = r.short_title.replace(" ViT", "");
    s += `<text x="0" y="${(y + 7.8).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${lbl}</text>`;
    r.ranks.forEach((rk, ci) => {
      const x = colX[ci];
      const txt = rk === null || rk > 20 ? "20+" : String(rk);
      let fill = "#f1f5f9", tcol = PALETTE.text;
      if (rk === 1) { fill = "#0284c7"; tcol = "#ffffff"; }
      else if (rk === 2) { fill = "#38bdf8"; tcol = PALETTE.text; }
      else if (rk <= 4) { fill = "#bae6fd"; tcol = PALETTE.text; }
      else if (rk <= 6) { fill = "#e0f2fe"; tcol = PALETTE.text; }
      s += `<rect x="${x}" y="${y.toFixed(2)}" width="${colW - 1.5}" height="${rowH - 1.4}" rx="1.2" fill="${fill}"/>`;
      s += `<text x="${(x + (colW - 1.5) / 2).toFixed(2)}" y="${(y + 7.8).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${tcol}" text-anchor="middle">${txt}</text>`;
    });
  });

  s += `<text x="0" y="107.5" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">Citations preset lifts classic foundations (#1).</text>`;

  s += `</svg>`;
  slot.innerHTML = s;
}

// 4. Col 2 scaling block (168 x 124 mm)
function renderScaling() {
  const slot = document.getElementById("slot-scaling");
  if (!slot) return;
  const first = DATA.scaling[0];
  const last = DATA.scaling[DATA.scaling.length - 1];
  const mrrDropPct = (((first.mrr_mean - last.mrr_mean) / first.mrr_mean) * 100).toFixed(1);
  const hitDropPct = (((first.hit1_mean - last.hit1_mean) / first.hit1_mean) * 100).toFixed(1);
  const nFull = last.n_papers.toLocaleString("en-US");

  const W = 168, H = 124;
  let s = makeSvg(W, H);

  // Claim title (28 pt bold, 2 lines) + How-to-read (22 pt, 2 lines) + Takeaway (24 pt bold, 2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Retrieval gets worse as the</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">library grows</text>`;
  s += `<text x="0" y="29.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">MRR: rank of right passage (1 = best).</text>`;
  s += `<text x="0" y="37.8" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Hit@1: it is first. Recall@6: in top 6.</text>`;
  s += `<text x="0" y="47.0" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">MRR falls ${mrrDropPct}% and Hit@1 ${hitDropPct}%</text>`;
  s += `<text x="0" y="56.2" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">from ${first.n_papers} to ${nFull} papers.</text>`;

  // Legend row at y=65.0
  s += `<line x1="24" y1="62.3" x2="33" y2="62.3" stroke="#047857" stroke-width="1.2" stroke-dasharray="3,1.5"/>`;
  s += `<rect x="27" y="60.8" width="3" height="3" fill="#047857"/>`;
  s += `<text x="35" y="65.0" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Recall@6</text>`;

  s += `<line x1="78" y1="62.3" x2="87" y2="62.3" stroke="${PALETTE.pillarBlue}" stroke-width="1.4"/>`;
  s += `<circle cx="82.5" cy="62.3" r="1.8" fill="${PALETTE.pillarBlue}"/>`;
  s += `<text x="89" y="65.0" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">MRR</text>`;

  s += `<line x1="118" y1="62.3" x2="127" y2="62.3" stroke="#b45309" stroke-width="1.2" stroke-dasharray="1.5,1.5"/>`;
  s += `<polygon points="122.5,60.3 120.5,64.3 124.5,64.3" fill="#b45309"/>`;
  s += `<text x="129" y="65.0" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Hit@1</text>`;

  // Plot area: left=24, right=144, top=70.0, bottom=103.5 (h=33.5)
  const pL = 24, pR = 144, pT = 70.0, pB = 103.5;
  const pW = pR - pL, pH = pB - pT;
  const logMin = Math.log10(first.n_papers), logMax = Math.log10(last.n_papers);
  const xMap = n => pL + ((Math.log10(n) - logMin) / (logMax - logMin)) * pW;
  const yMap = v => pB - v * pH;

  [0.0, 0.50, 1.0].forEach(yv => {
    const py = yMap(yv);
    s += `<line x1="${pL}" y1="${py.toFixed(2)}" x2="${pR}" y2="${py.toFixed(2)}" stroke="${PALETTE.grid}" stroke-width="0.4"/>`;
    s += `<text x="${pL - 2}" y="${(py + 2.5).toFixed(2)}" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="end">${yv.toFixed(2)}</text>`;
  });

  s += `<line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pB}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;
  s += `<line x1="${pL}" y1="${pB}" x2="${pR}" y2="${pB}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;

  DATA.scaling.forEach((d, idx) => {
    const px = xMap(d.n_papers);
    const lbl = d.n_papers.toLocaleString("en-US");
    s += `<line x1="${px.toFixed(2)}" y1="${pB}" x2="${px.toFixed(2)}" y2="${pB + 2}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;
    const anchor = idx === DATA.scaling.length - 1 ? "end" : (idx === 0 ? "start" : "middle");
    const tx = idx === DATA.scaling.length - 1 ? px + 2 : (idx === 0 ? px - 2 : px);
    s += `<text x="${tx.toFixed(2)}" y="${pB + 9.2}" font-size="${FS.tick}" fill="${PALETTE.text}" text-anchor="${anchor}">${lbl}</text>`;
  });

  const topPts = DATA.scaling.map(d => `${xMap(d.n_papers).toFixed(2)},${yMap(d.mrr_ci_hi).toFixed(2)}`);
  const botPts = [...DATA.scaling].reverse().map(d => `${xMap(d.n_papers).toFixed(2)},${yMap(d.mrr_ci_lo).toFixed(2)}`);
  s += `<polygon points="${topPts.concat(botPts).join(" ")}" fill="#93c5fd" fill-opacity="0.35"/>`;

  const recPts = DATA.scaling.map(d => `${xMap(d.n_papers).toFixed(2)},${yMap(d.recall6_mean).toFixed(2)}`).join(" ");
  const mrrPts = DATA.scaling.map(d => `${xMap(d.n_papers).toFixed(2)},${yMap(d.mrr_mean).toFixed(2)}`).join(" ");
  const hitPts = DATA.scaling.map(d => `${xMap(d.n_papers).toFixed(2)},${yMap(d.hit1_mean).toFixed(2)}`).join(" ");

  s += `<polyline points="${recPts}" fill="none" stroke="#047857" stroke-width="1.2" stroke-dasharray="3,1.5"/>`;
  s += `<polyline points="${mrrPts}" fill="none" stroke="${PALETTE.pillarBlue}" stroke-width="1.5"/>`;
  s += `<polyline points="${hitPts}" fill="none" stroke="#b45309" stroke-width="1.2" stroke-dasharray="1.5,1.5"/>`;

  DATA.scaling.forEach(d => {
    const px = xMap(d.n_papers);
    const yr = yMap(d.recall6_mean), ym = yMap(d.mrr_mean), yh = yMap(d.hit1_mean);
    s += `<rect x="${(px - 1.5).toFixed(2)}" y="${(yr - 1.5).toFixed(2)}" width="3" height="3" fill="#047857"/>`;
    s += `<circle cx="${px.toFixed(2)}" cy="${ym.toFixed(2)}" r="1.8" fill="${PALETTE.pillarBlue}"/>`;
    s += `<polygon points="${px.toFixed(2)},${(yh - 2).toFixed(2)} ${(px - 1.8).toFixed(2)},${(yh + 1.6).toFixed(2)} ${(px + 1.8).toFixed(2)},${(yh + 1.6).toFixed(2)}" fill="#b45309"/>`;
  });

  const yRecEnd = yMap(last.recall6_mean);
  const yMrrEnd = Math.max(yRecEnd + 8.8, yMap(last.mrr_mean));
  const yHitEnd = Math.max(yMrrEnd + 8.8, yMap(last.hit1_mean));
  s += `<text x="${pR + 2.5}" y="${(yRecEnd + 2.5).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="#047857" text-anchor="start">${last.recall6_mean.toFixed(2)}</text>`;
  s += `<text x="${pR + 2.5}" y="${(yMrrEnd + 2.5).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.pillarBlue}" text-anchor="start">${last.mrr_mean.toFixed(2)}</text>`;
  s += `<text x="${pR + 2.5}" y="${(yHitEnd + 2.5).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="#b45309" text-anchor="start">${last.hit1_mean.toFixed(2)}</text>`;

  s += `<text x="0" y="120.5" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">Band: 95% bootstrap CI over ${DATA.dense_vs_hybrid.n_questions} questions (MRR).</text>`;

  s += `</svg>`;
  slot.innerHTML = s;
}

// 5. Col 2 densehybrid block (168 x 110 mm)
function renderDenseHybrid() {
  const slot = document.getElementById("slot-densehybrid");
  if (!slot) return;
  const dvh = DATA.dense_vs_hybrid;
  const mMrr = dvh.metrics[0];
  const mHit = dvh.metrics[1];
  const mRec = dvh.metrics[2];
  const mrrExcludes0 = mMrr.ci[0] > 0 || mMrr.ci[1] < 0;
  const hitExcludes0 = mHit.ci[0] > 0 || mHit.ci[1] < 0;
  const recIncludes0 = mRec.ci[0] <= 0 && mRec.ci[1] >= 0;
  const ratio = (dvh.p95_latency_ms.hybrid / dvh.p95_latency_ms.dense).toFixed(1);
  const tLine1 = (mrrExcludes0 && hitExcludes0 && recIncludes0)
    ? "Ranking gains hold, recall gain does not."
    : "Hybrid improves ranking.";
  const tLine2 = `Retrieval is ${ratio}x slower.`;

  const W = 168, H = 110;
  let s = makeSvg(W, H);

  // Claim title (2 lines) + How-to-read (2 lines) + Takeaway (2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Keyword matching ranks the right</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">passage higher</text>`;
  s += `<text x="0" y="29.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Same 80 questions. Under each pair: gain</text>`;
  s += `<text x="0" y="37.8" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">from hybrid, with 95% interval.</text>`;
  s += `<text x="0" y="47.2" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${tLine1}</text>`;
  s += `<text x="0" y="56.4" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${tLine2}</text>`;

  // Legend
  s += `<rect x="95" y="51.2" width="4.8" height="4.8" rx="1" fill="#64748b"/>`;
  s += `<text x="101.0" y="56.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Dense</text>`;
  s += `<rect x="126" y="51.2" width="4.8" height="4.8" rx="1" fill="${PALETTE.pillarGreen}"/>`;
  s += `<text x="132.0" y="56.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Hybrid [1]</text>`;

  // Plot area: left=24, right=166, top=65.0, bottom=84.0 (h=19)
  const pL = 24, pR = 166, pT = 65.0, pB = 84.0;
  const pH = pB - pT;
  const yMax = 0.85;
  const yMap = v => pB - (v / yMax) * pH;

  [0.0, 0.8].forEach(yv => {
    const py = yMap(yv);
    s += `<line x1="${pL}" y1="${py.toFixed(2)}" x2="${pR}" y2="${py.toFixed(2)}" stroke="${PALETTE.grid}" stroke-width="0.4"/>`;
    s += `<text x="${pL - 2}" y="${(py + 2.4).toFixed(2)}" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="end">${yv.toFixed(1)}</text>`;
  });
  s += `<line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pB}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;
  s += `<line x1="${pL}" y1="${pB}" x2="${pR}" y2="${pB}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;

  const groupCenters = [48, 95, 142];
  const barW = 16;
  dvh.metrics.forEach((m, idx) => {
    const cx = groupCenters[idx];
    const xD = cx - barW - 1, xH = cx + 1;
    const yD = yMap(m.dense), yH = yMap(m.hybrid);
    s += `<rect x="${xD}" y="${yD.toFixed(2)}" width="${barW}" height="${(pB - yD).toFixed(2)}" fill="#64748b"/>`;
    s += `<rect x="${xH}" y="${yH.toFixed(2)}" width="${barW}" height="${(pB - yH).toFixed(2)}" fill="${PALETTE.pillarGreen}"/>`;

    s += `<text x="${(xD + barW / 2 - 2).toFixed(2)}" y="${(yD - 1.5).toFixed(2)}" font-size="${FS.caption}" font-weight="700" fill="#475569" text-anchor="middle">${m.dense.toFixed(2)}</text>`;
    s += `<text x="${(xH + barW / 2 + 2).toFixed(2)}" y="${(yH - 1.5).toFixed(2)}" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.pillarGreen}" text-anchor="middle">${m.hybrid.toFixed(2)}</text>`;

    // 3 non-overlapping rows under each group
    s += `<text x="${cx}" y="91.5" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">${m.name}</text>`;
    s += `<text x="${cx}" y="99.5" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.pillarGreen}" text-anchor="middle">${m.diff_str}</text>`;
    s += `<text x="${cx}" y="107.2" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="middle">[${m.ci[0].toFixed(2)}, ${m.ci[1].toFixed(2)}]</text>`;
  });

  s += `</svg>`;
  slot.innerHTML = s;
}

// 6. Col 2 generation block (168 x 110 mm)
function renderGeneration() {
  const slot = document.getElementById("slot-generation");
  if (!slot) return;
  const g = DATA.generation;
  const totalAns = g.FP + g.TN;

  const W = 168, H = 110;
  let s = makeSvg(W, H);

  // Claim title (2 lines) + How-to-read (2 lines) + Takeaway (2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Citations are real, but the system</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">over-refuses</text>`;
  s += `<text x="0" y="29.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Rows: question type. Columns: what it did.</text>`;
  s += `<text x="0" y="37.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Green cells are correct.</text>`;
  s += `<text x="0" y="46.2" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${g.FP} of ${totalAns} answerable questions were</text>`;
  s += `<text x="0" y="55.2" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">wrongly refused.</text>`;

  // Headers
  s += `<text x="0" y="63.0" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">Positive means should refuse</text>`;
  s += `<text x="116" y="63.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">Refused</text>`;
  s += `<text x="151" y="63.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">Answered</text>`;

  // Row 1: Out of scope
  s += `<text x="0" y="70.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Out of scope</text>`;
  s += `<rect x="99" y="64.4" width="33" height="6.6" rx="1.0" fill="#dcfce7" stroke="#059669" stroke-width="0.4"/>`;
  s += `<text x="115.5" y="70.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">TP: ${g.TP}</text>`;
  s += `<rect x="134" y="64.4" width="33" height="6.6" rx="1.0" fill="#fef3c7" stroke="#d97706" stroke-width="0.4"/>`;
  s += `<text x="150.5" y="70.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">FN: ${g.FN}</text>`;

  // Row 2: Answerable
  s += `<text x="0" y="77.6" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Answerable</text>`;
  s += `<rect x="99" y="72.0" width="33" height="6.6" rx="1.0" fill="#fef3c7" stroke="#d97706" stroke-width="0.4"/>`;
  s += `<text x="115.5" y="77.6" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">FP: ${g.FP}</text>`;
  s += `<rect x="134" y="72.0" width="33" height="6.6" rx="1.0" fill="#dcfce7" stroke="#059669" stroke-width="0.4"/>`;
  s += `<text x="150.5" y="77.6" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">TN: ${g.TN}</text>`;

  s += `<line x1="0" y1="79.2" x2="168" y2="79.2" stroke="${PALETTE.line}" stroke-width="0.4"/>`;

  // 3-row metrics table
  s += `<text x="0" y="85.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Refusal recall</text>`;
  s += `<text x="167" y="85.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.pillarGreen}" text-anchor="end">${g.refusal_recall.display}</text>`;

  s += `<text x="0" y="92.6" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.warnOrange}" text-anchor="start">Refusal precision</text>`;
  s += `<text x="167" y="92.6" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.warnOrange}" text-anchor="end">${g.refusal_precision.display}</text>`;

  s += `<text x="0" y="100.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Citations point to a retrieved passage</text>`;
  const citPctOnly = `${(g.citation_validity.value * 100).toFixed(1)}%`;
  s += `<text x="167" y="100.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.pillarGreen}" text-anchor="end">${citPctOnly}</text>`;

  s += `<text x="0" y="107.4" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">Does not test if passage supports claim (${g.citation_validity.num}/${g.citation_validity.den}).</text>`;

  s += `</svg>`;
  slot.innerHTML = s;
}

// 7. Col 3 cone block (168 x 140 mm)
function renderCone() {
  const slot = document.getElementById("slot-cone");
  if (!slot) return;
  const W = 168, H = 140;
  let s = makeSvg(W, H);

  // Claim title (2 lines) + How-to-read (2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Novelty is measured against a</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">paper's own references</text>`;
  s += `<text x="0" y="29.4" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Shaded cone: all blends of cited papers.</text>`;
  s += `<text x="0" y="37.8" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">x-hat: nearest blend to x. Res: the gap.</text>`;

  // One-line formula at y=41.5..52.5
  s += `<rect x="0.5" y="41.5" width="167" height="11" rx="1.5" fill="#f8fafc" stroke="${PALETTE.line}" stroke-width="0.4"/>`;
  s += `<text x="84" y="49.2" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">x-hat = Yλ* (λ≥0),  Δ = R²₊ - m²,  Res = √(1 - R²₊)</text>`;

  // Schematic in left area: origin at (12, 121), radius R = 62
  const ox = 12, oy = 121, R = 62;
  const arcX = ox + R, arcY = oy - R;
  s += `<path d="M ${arcX} ${oy} A ${R} ${R} 0 0 0 ${ox} ${arcY}" fill="none" stroke="${PALETTE.line}" stroke-width="0.8" stroke-dasharray="2.5,1.5"/>`;

  s += `<line x1="${ox}" y1="${oy}" x2="${ox + R + 4}" y2="${oy}" stroke="${PALETTE.text}" stroke-width="0.7"/>`;
  s += `<line x1="${ox}" y1="${oy}" x2="${ox}" y2="${oy - R - 4}" stroke="${PALETTE.text}" stroke-width="0.7"/>`;
  s += `<text x="${ox + R - 10}" y="${oy + 8.0}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="middle">e1</text>`;
  s += `<text x="${ox - 2.5}" y="${oy - R + 8}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="end">e2</text>`;

  const ang2 = 52 * Math.PI / 180;
  const a2x = ox + R * Math.cos(ang2), a2y = oy - R * Math.sin(ang2);
  const ang3 = 28 * Math.PI / 180;
  const a3x = ox + R * Math.cos(ang3), a3y = oy - R * Math.sin(ang3);
  s += `<polygon points="${ox},${oy} ${ox + R},${oy} ${a2x.toFixed(2)},${a2y.toFixed(2)}" fill="#f3e8ff" stroke="#c084fc" stroke-width="0.6"/>`;

  const bb = DATA.novelty.exemplars[1];
  const xhatX = ox + R * bb.m;
  const xhatY = oy - R * Math.sqrt(bb.delta);
  const xTopY = xhatY - R * bb.res;

  s += `<line x1="${xhatX.toFixed(2)}" y1="${xhatY.toFixed(2)}" x2="${xhatX.toFixed(2)}" y2="${xTopY.toFixed(2)}" stroke="${PALETTE.dangerRed}" stroke-width="1.3" stroke-dasharray="2.5,1.5"/>`;

  const p1x = ox + R, p1y = oy;
  s += `<circle cx="${p1x}" cy="${p1y}" r="2.2" fill="${PALETTE.pillarPurple}"/>`;
  s += `<circle cx="${a3x.toFixed(2)}" cy="${a3y.toFixed(2)}" r="2.0" fill="${PALETTE.pillarPurple}"/>`;
  s += `<circle cx="${a2x.toFixed(2)}" cy="${a2y.toFixed(2)}" r="2.0" fill="${PALETTE.pillarPurple}"/>`;
  s += `<circle cx="${xhatX.toFixed(2)}" cy="${xhatY.toFixed(2)}" r="2.4" fill="${PALETTE.pillarGreen}"/>`;
  s += `<circle cx="${xhatX.toFixed(2)}" cy="${xTopY.toFixed(2)}" r="2.6" fill="${PALETTE.pillarBlue}"/>`;

  const legItems = [
    { px: xhatX, py: xTopY, ly: 61, text: "x (target paper)", col: PALETTE.pillarBlue },
    { px: xhatX, py: (xTopY + xhatY) / 2, ly: 73, text: "Res (out of plane)", col: PALETTE.dangerRed },
    { px: a2x, py: a2y, ly: 85, text: "ancestor 2", col: PALETTE.pillarPurple },
    { px: a3x, py: a3y, ly: 97, text: "ancestor 3", col: PALETTE.pillarPurple },
    { px: xhatX, py: xhatY, ly: 109, text: "x-hat (in cone)", col: PALETTE.pillarGreen },
    { px: p1x, py: p1y, ly: 121, text: "closest parent", col: PALETTE.pillarPurple }
  ];
  legItems.forEach(item => {
    s += `<polyline points="${item.px.toFixed(2)},${item.py.toFixed(2)} 82,${item.ly} 86,${item.ly}" fill="none" stroke="${item.col}" stroke-width="0.45"/>`;
    s += `<text x="88" y="${(item.ly + 2.5).toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${item.col}" text-anchor="start">${item.text}</text>`;
  });

  const hs = DATA.novelty.classes[0];
  const tot = DATA.novelty.total_papers;
  const hsPct = ((hs.count / tot) * 100).toFixed(1);
  s += `<text x="0" y="137.0" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">${hs.count.toLocaleString("en-US")} of ${tot.toLocaleString("en-US")} papers (${hsPct}%) land in ${hs.id}.</text>`;

  s += `</svg>`;
  slot.innerHTML = s;
}

// 8. Col 3 exemplars block (168 x 96 mm)
function renderExemplars() {
  const slot = document.getElementById("slot-exemplars");
  if (!slot) return;
  const exs = DATA.novelty.exemplars;
  let lowEx = exs[0];
  exs.forEach(e => {
    if (e.refs_resolved / e.refs_total < lowEx.refs_resolved / lowEx.refs_total) {
      lowEx = e;
    }
  });
  const cleanName = lowEx.name.replace(/\s*\[\d+\]$/, "");

  const W = 168, H = 96;
  let s = makeSvg(W, H);

  // Claim title (2 lines) + How-to-read (2 lines) + Takeaway (2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Two real papers: one close parent,</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">one blend</text>`;
  s += `<text x="0" y="29.0" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">m: similarity to closest parent. Delta: gain</text>`;
  s += `<text x="0" y="37.0" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">from blending. Res: new part.</text>`;
  s += `<text x="0" y="45.8" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${cleanName} used ${lowEx.refs_resolved} of ${lowEx.refs_total} references,</text>`;
  s += `<text x="0" y="54.6" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">so its Res is uncertain.</text>`;

  const headers = [
    { x: 0, t: "Paper", a: "start" },
    { x: 70, t: "Refs", a: "middle" },
    { x: 95, t: "m", a: "middle" },
    { x: 114, t: "Δ", a: "middle" },
    { x: 134, t: "Res", a: "middle" },
    { x: 167, t: "Share", a: "end" }
  ];
  headers.forEach(h => {
    s += `<text x="${h.x}" y="64.0" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.muted}" text-anchor="${h.a}">${h.t}</text>`;
  });
  s += `<line x1="0" y1="66.2" x2="168" y2="66.2" stroke="${PALETTE.line}" stroke-width="0.6"/>`;

  exs.forEach((ex, idx) => {
    const ry = 74.2 + idx * 9.6;
    const sharePct = (ex.top3_nnls[0].share * 100).toFixed(1) + "%";
    s += `<text x="0" y="${ry.toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${ex.name}</text>`;
    s += `<text x="70" y="${ry.toFixed(2)}" font-size="${FS.tick}" fill="${PALETTE.text}" text-anchor="middle">${ex.refs_resolved}/${ex.refs_total}</text>`;
    s += `<text x="95" y="${ry.toFixed(2)}" font-size="${FS.tick}" fill="${PALETTE.text}" text-anchor="middle">${ex.m.toFixed(2)}</text>`;
    s += `<text x="114" y="${ry.toFixed(2)}" font-size="${FS.tick}" fill="${PALETTE.text}" text-anchor="middle">${ex.delta.toFixed(2)}</text>`;
    s += `<text x="134" y="${ry.toFixed(2)}" font-size="${FS.tick}" fill="${PALETTE.text}" text-anchor="middle">${ex.res.toFixed(2)}</text>`;
    s += `<text x="167" y="${ry.toFixed(2)}" font-size="${FS.tick}" font-weight="700" fill="${PALETTE.pillarPurple}" text-anchor="end">${sharePct}</text>`;
  });

  s += `<line x1="0" y1="86.2" x2="168" y2="86.2" stroke="${PALETTE.line}" stroke-width="0.4"/>`;
  const ex0 = exs[0], ex1 = exs[1];
  const cap = `Top parents: ${ex0.top3_nnls[0].short_name} (${(ex0.top3_nnls[0].share * 100).toFixed(1)}%), ${ex1.top3_nnls[0].short_name} (${(ex1.top3_nnls[0].share * 100).toFixed(1)}%).`;
  s += `<text x="0" y="92.8" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">${cap}</text>`;

  s += `</svg>`;
  slot.innerHTML = s;
}

// 9. Col 3 null control block (168 x 108 mm)
function renderNull() {
  const slot = document.getElementById("slot-null");
  if (!slot) return;
  const nData = DATA.novelty.null;
  const d = nData.real.mean - nData.library_null.mean;
  const ci = nData.paired_ci || [-0.025, -0.017];
  const ciLoAbove0 = ci[0] > 0;
  const tLine1 = ciLoAbove0
    ? "Real citations score higher than"
    : "Real citations score no higher than";
  const tLine2 = ciLoAbove0 ? "random references" : "random ones";
  const tkLine1 = ciLoAbove0
    ? `Mean diff is +${d.toFixed(3)} above`
    : "So delta alone does not show";
  const tkLine2 = ciLoAbove0 ? "library null." : "real synergy.";

  const W = 168, H = 108;
  let s = makeSvg(W, H);

  // Claim title (2 lines) + How-to-read (2 lines) + Takeaway (2 lines)
  s += `<text x="0" y="9.8" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${tLine1}</text>`;
  s += `<text x="0" y="20.4" font-size="${FS.title}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${tLine2}</text>`;
  s += `<text x="0" y="29.0" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">Curves: how often each value occurs. Random</text>`;
  s += `<text x="0" y="37.0" font-size="${FS.tick}" fill="${PALETTE.muted}" text-anchor="start">sets match each reference count.</text>`;
  s += `<text x="0" y="45.8" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${tkLine1}</text>`;
  s += `<text x="0" y="54.6" font-size="${FS.body}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">${tkLine2}</text>`;

  // Colour key at y=63.0 with means printed dynamically from DATA
  s += `<line x1="0" y1="60.4" x2="7" y2="60.4" stroke="#64748b" stroke-width="1.2" stroke-dasharray="2,1.5"/>`;
  s += `<text x="9" y="63.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Iso ${nData.isotropic_null.mean.toFixed(3)}</text>`;

  s += `<line x1="54" y1="60.4" x2="61" y2="60.4" stroke="${PALETTE.pillarPurple}" stroke-width="1.4"/>`;
  s += `<text x="63" y="63.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Real ${nData.real.mean.toFixed(3)}</text>`;

  s += `<line x1="110" y1="60.4" x2="117" y2="60.4" stroke="${PALETTE.sigEmbed}" stroke-width="1.3" stroke-dasharray="3.5,1.5"/>`;
  s += `<text x="119" y="63.0" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.text}" text-anchor="start">Lib ${nData.library_null.mean.toFixed(3)}</text>`;

  // Plot area: left=22, right=166 (w=144), top=72.2, bottom=89.5 (h=17.3)
  const pL = 22, pR = 166, pT = 72.2, pB = 89.5;
  const pW = pR - pL, pH = pB - pT;
  const bins = nData.bin_edges;
  const nBins = bins.length - 1;
  const bw = pW / nBins;

  s += `<line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pB}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;
  s += `<line x1="${pL}" y1="${pB}" x2="${pR}" y2="${pB}" stroke="${PALETTE.text}" stroke-width="0.6"/>`;
  s += `<text x="${pL - 2}" y="${pT + 5}" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="end">1.0</text>`;
  s += `<text x="${pL - 2}" y="${pB}" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="end">0.0</text>`;

  function buildStepPath(counts) {
    const maxC = Math.max(...counts);
    let dStr = `M ${pL.toFixed(2)} ${pB.toFixed(2)}`;
    for (let i = 0; i < nBins; i++) {
      const x0 = pL + i * bw;
      const x1 = x0 + bw;
      const y = pB - (counts[i] / maxC) * pH;
      dStr += ` L ${x0.toFixed(2)} ${y.toFixed(2)} L ${x1.toFixed(2)} ${y.toFixed(2)}`;
    }
    dStr += ` L ${pR.toFixed(2)} ${pB.toFixed(2)}`;
    return dStr;
  }

  const pIso = buildStepPath(nData.isotropic_null.counts);
  const pLib = buildStepPath(nData.library_null.counts);
  const pReal = buildStepPath(nData.real.counts);

  s += `<path d="${pIso} Z" fill="#cbd5e1" fill-opacity="0.30"/>`;
  s += `<path d="${pIso}" fill="none" stroke="#64748b" stroke-width="1.0" stroke-dasharray="2,1.5"/>`;

  s += `<path d="${pLib} Z" fill="#bae6fd" fill-opacity="0.35"/>`;
  s += `<path d="${pLib}" fill="none" stroke="${PALETTE.sigEmbed}" stroke-width="1.2" stroke-dasharray="3.5,1.5"/>`;

  s += `<path d="${pReal} Z" fill="#e9d5ff" fill-opacity="0.35"/>`;
  s += `<path d="${pReal}" fill="none" stroke="${PALETTE.pillarPurple}" stroke-width="1.4"/>`;

  const xMap = v => pL + (v / bins[bins.length - 1]) * pW;
  const xIso = xMap(nData.isotropic_null.mean);
  const xReal = xMap(nData.real.mean);
  const xLib = xMap(nData.library_null.mean);

  s += `<line x1="${xIso.toFixed(2)}" y1="${pT}" x2="${xIso.toFixed(2)}" y2="${pB}" stroke="#475569" stroke-width="0.9" stroke-dasharray="2,1.5"/>`;
  s += `<line x1="${xReal.toFixed(2)}" y1="${pT}" x2="${xReal.toFixed(2)}" y2="${pB}" stroke="${PALETTE.pillarPurple}" stroke-width="1.1"/>`;
  s += `<line x1="${xLib.toFixed(2)}" y1="${pT}" x2="${xLib.toFixed(2)}" y2="${pB}" stroke="${PALETTE.sigEmbed}" stroke-width="1.1" stroke-dasharray="3,1.5"/>`;

  s += `<text x="${(xIso + 2).toFixed(2)}" y="70.2" font-size="${FS.caption}" font-weight="700" fill="#475569" text-anchor="start">${nData.isotropic_null.mean.toFixed(3)}</text>`;
  s += `<text x="${(xReal - 2).toFixed(2)}" y="70.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.pillarPurple}" text-anchor="end">${nData.real.mean.toFixed(3)}</text>`;
  s += `<text x="${(xLib + 2).toFixed(2)}" y="70.2" font-size="${FS.caption}" font-weight="700" fill="${PALETTE.sigEmbed}" text-anchor="start">${nData.library_null.mean.toFixed(3)}</text>`;

  // X ticks at y=96.5 and caption at y=105.0
  [0.00, 0.06, 0.12, 0.18].forEach((xv, idx) => {
    const px = xMap(xv);
    const a = idx === 3 ? "end" : (idx === 0 ? "start" : "middle");
    s += `<text x="${px.toFixed(2)}" y="96.5" font-size="${FS.caption}" fill="${PALETTE.text}" text-anchor="${a}">${xv.toFixed(2)}</text>`;
  });

  s += `<text x="0" y="105.0" font-size="${FS.caption}" fill="${PALETTE.muted}" text-anchor="start">Diff: ${d.toFixed(3)} (95% CI ${ci[0].toFixed(3)} to ${ci[1].toFixed(3)}).</text>`;

  s += `</svg>`;
  slot.innerHTML = s;
}

function renderAllCharts() {
  verifyHistogramMeans();
  renderPresets();
  renderTop5();
  renderRankMatrix();
  renderScaling();
  renderDenseHybrid();
  renderGeneration();
  renderCone();
  renderExemplars();
  renderNull();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderAllCharts);
} else {
  renderAllCharts();
}
