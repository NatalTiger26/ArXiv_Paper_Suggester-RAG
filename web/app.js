// ArXiv Scientific Discovery Platform — Modern Web Studio Client

document.addEventListener("DOMContentLoaded", () => {
  // ── State ───────────────────────────────────────────────────────────
  let currentRankedPapers = [];
  let currentSources = [];
  let currentAnswer = "";

  // ── Elements ────────────────────────────────────────────────────────
  const catalogCountEl = document.getElementById("catalogCount");
  const chunksCountEl = document.getElementById("chunksCount");
  const telemetryCallsEl = document.getElementById("telemetryCalls");

  const rankQueryInput = document.getElementById("rankQuery");
  const btnRank = document.getElementById("btnRank");
  const rankResultsList = document.getElementById("rankResultsList");
  const rankCountBadge = document.getElementById("rankCountBadge");
  const ftsKeywordInput = document.getElementById("ftsKeyword");

  const wEmb = document.getElementById("wEmb");
  const wRec = document.getElementById("wRec");
  const wCit = document.getElementById("wCit");
  const wAuth = document.getElementById("wAuth");
  const wVel = document.getElementById("wVel");
  const wTemp = document.getElementById("wTemp");

  const qaQueryInput = document.getElementById("qaQuery");
  const btnAsk = document.getElementById("btnAsk");
  const chkStream = document.getElementById("chkStream");
  const chkCache = document.getElementById("chkCache");
  const answerContent = document.getElementById("answerContent");
  const qaMetaBar = document.getElementById("qaMetaBar");
  const metaCoverage = document.getElementById("metaCoverage");
  const metaCache = document.getElementById("metaCache");
  const metaLatency = document.getElementById("metaLatency");
  const btnExportBib = document.getElementById("btnExportBib");
  const btnExportMd = document.getElementById("btnExportMd");

  const tabBtnSubspace = document.getElementById("tabBtnSubspace");
  const tabBtnPassages = document.getElementById("tabBtnPassages");
  const tabBtnHubs = document.getElementById("tabBtnHubs");
  const tabSubspace = document.getElementById("tabSubspace");
  const tabPassages = document.getElementById("tabPassages");
  const tabHubs = document.getElementById("tabHubs");

  const noveltyArxivIdInput = document.getElementById("noveltyArxivId");
  const btnComputeNovelty = document.getElementById("btnComputeNovelty");
  const subspaceCanvas = document.getElementById("subspaceCanvas");
  const mVal = document.getElementById("mVal");
  const deltaVal = document.getElementById("deltaVal");
  const r2Val = document.getElementById("r2Val");
  const resVal = document.getElementById("resVal");
  const attributionTable = document.getElementById("attributionTable");

  const passagesList = document.getElementById("passagesList");
  const passagesCount = document.getElementById("passagesCount");
  const hubsList = document.getElementById("hubsList");
  const btnRefreshHubs = document.getElementById("btnRefreshHubs");

  // ── Sync Weight Sliders ──────────────────────────────────────────────
  const syncSliders = () => {
    document.getElementById("wEmbVal").textContent = parseFloat(wEmb.value).toFixed(2);
    document.getElementById("wRecVal").textContent = parseFloat(wRec.value).toFixed(2);
    document.getElementById("wCitVal").textContent = parseFloat(wCit.value).toFixed(2);
    document.getElementById("wAuthVal").textContent = parseFloat(wAuth.value).toFixed(2);
    document.getElementById("wVelVal").textContent = parseFloat(wVel.value).toFixed(2);
    document.getElementById("wTempVal").textContent = parseFloat(wTemp.value).toFixed(2);
  };

  [wEmb, wRec, wCit, wAuth, wVel, wTemp].forEach(slider => {
    slider.addEventListener("input", syncSliders);
  });
  syncSliders();

  // ── Health & Diagnostics ─────────────────────────────────────────────
  const loadHealth = async () => {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) return;
      const data = await res.json();
      if (data.catalog && data.catalog.total_papers) {
        catalogCountEl.textContent = data.catalog.total_papers.toLocaleString();
      }
      if (data.chroma_rag && data.chroma_rag.total_chunks) {
        chunksCountEl.textContent = data.chroma_rag.total_chunks.toLocaleString();
      }
      if (data.telemetry && data.telemetry.total_calls !== undefined) {
        telemetryCallsEl.textContent = data.telemetry.total_calls;
      }
    } catch (e) {
      console.warn("Health check error:", e);
    }
  };
  loadHealth();

  // ── Tabs Switching ───────────────────────────────────────────────────
  const selectTab = (tab) => {
    [tabBtnSubspace, tabBtnPassages, tabBtnHubs].forEach(b => b.classList.remove("active"));
    [tabSubspace, tabPassages, tabHubs].forEach(c => c.style.display = "none");

    if (tab === "subspace") {
      tabBtnSubspace.classList.add("active");
      tabSubspace.style.display = "flex";
    } else if (tab === "passages") {
      tabBtnPassages.classList.add("active");
      tabPassages.style.display = "flex";
    } else if (tab === "hubs") {
      tabBtnHubs.classList.add("active");
      tabHubs.style.display = "flex";
      loadHubs();
    }
  };

  tabBtnSubspace.addEventListener("click", () => selectTab("subspace"));
  tabBtnPassages.addEventListener("click", () => selectTab("passages"));
  tabBtnHubs.addEventListener("click", () => selectTab("hubs"));

  // ── Preset Pills ─────────────────────────────────────────────────────
  document.querySelectorAll(".preset-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      qaQueryInput.value = pill.getAttribute("data-q");
      btnAsk.click();
    });
  });

  // ── Multi-Signal Ranking ─────────────────────────────────────────────
  const renderRankedPapers = (papers) => {
    const filter = (ftsKeywordInput.value || "").toLowerCase().trim();
    const filtered = filter
      ? papers.filter(p => (p.title || "").toLowerCase().includes(filter) || (p.arxiv_id || "").includes(filter))
      : papers;

    rankCountBadge.textContent = `${filtered.length} papers`;
    if (!filtered.length) {
      rankResultsList.innerHTML = `<div class="empty-state">No matching papers found.</div>`;
      return;
    }

    rankResultsList.innerHTML = filtered.map(p => {
      const aid = p.arxiv_id || p.paper_id;
      const sig = p.signals || {};
      const score = (p.final_score !== undefined) ? p.final_score.toFixed(3) : "—";
      const year = p.year || "—";
      const cites = (p.citation_count !== "" && p.citation_count !== null && p.citation_count !== undefined)
        ? `${p.citation_count} cites`
        : "—";

      return `
        <div class="paper-card" data-aid="${aid}">
          <div class="card-top">
            <span class="score-badge">Score: ${score}</span>
            <span class="aid-badge">${aid}</span>
          </div>
          <div class="paper-title">${p.title}</div>
          <div class="paper-meta">
            <span>📅 ${year}</span>
            <span>⭐ ${cites}</span>
          </div>
          <div class="signal-pills">
            <span class="signal-pill">emb: ${(sig.embedding || 0).toFixed(2)}</span>
            <span class="signal-pill">rec: ${(sig.recency || 0).toFixed(2)}</span>
            <span class="signal-pill">cit: ${(sig.citation || 0).toFixed(2)}</span>
            <span class="signal-pill">vel: ${(sig.velocity || 0).toFixed(2)}</span>
          </div>
          <div class="card-actions">
            <button class="btn-card-action btn-ask-this" data-title="${p.title}" data-aid="${aid}">Ask This</button>
            <button class="btn-card-action btn-nov-this" data-aid="${aid}">Novelty</button>
            <button class="btn-card-action btn-ingest-this" data-aid="${aid}">Ingest</button>
          </div>
        </div>
      `;
    }).join("");

    // Bind action buttons
    rankResultsList.querySelectorAll(".btn-ask-this").forEach(b => {
      b.addEventListener("click", () => {
        const aid = b.getAttribute("data-aid");
        const title = b.getAttribute("data-title");
        qaQueryInput.value = `What are the key findings, methodologies, and limitations of paper ${aid} (${title})?`;
        btnAsk.click();
      });
    });

    rankResultsList.querySelectorAll(".btn-nov-this").forEach(b => {
      b.addEventListener("click", () => {
        noveltyArxivIdInput.value = b.getAttribute("data-aid");
        selectTab("subspace");
        btnComputeNovelty.click();
      });
    });

    rankResultsList.querySelectorAll(".btn-ingest-this").forEach(b => {
      b.addEventListener("click", async () => {
        const aid = b.getAttribute("data-aid");
        b.textContent = "Ingesting...";
        b.disabled = true;
        try {
          const res = await fetch("/api/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ arxiv_id: aid }),
          });
          const d = await res.json();
          if (d.skipped) {
            b.textContent = "✓ In Index";
          } else {
            b.textContent = "✓ Ingested";
          }
          loadHealth();
        } catch (e) {
          b.textContent = "Error";
        }
      });
    });
  };

  ftsKeywordInput.addEventListener("input", () => renderRankedPapers(currentRankedPapers));

  btnRank.addEventListener("click", async () => {
    const q = rankQueryInput.value.trim();
    if (!q) return;

    btnRank.textContent = "Ranking...";
    btnRank.disabled = true;

    try {
      const payload = {
        query: q,
        k: 15,
        weights: {
          embedding: parseFloat(wEmb.value),
          recency: parseFloat(wRec.value),
          citation: parseFloat(wCit.value),
          author: parseFloat(wAuth.value),
          velocity: parseFloat(wVel.value),
        },
        temperature: parseFloat(wTemp.value),
        half_life: 4.0,
      };

      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      currentRankedPapers = data.results || [];
      renderRankedPapers(currentRankedPapers);
    } catch (e) {
      rankResultsList.innerHTML = `<div class="empty-state">Error during ranking: ${e.message}</div>`;
    } finally {
      btnRank.textContent = "Rank Papers";
      btnRank.disabled = false;
    }
  });

  // ── Source Passages Rendering & Citation Tag Linking ────────────────
  const renderSourcePassages = (chunks) => {
    currentSources = chunks || [];
    passagesCount.textContent = `${currentSources.length} retrieved passages`;

    if (!currentSources.length) {
      passagesList.innerHTML = `<div class="empty-state">No source passages retrieved.</div>`;
      return;
    }

    passagesList.innerHTML = currentSources.map((c, i) => {
      const idx = i + 1;
      const pid = c.paper_id || c.title || "Unknown";
      const sec = c.section || "body";
      const score = (c.rrf_score !== undefined) ? `RRF: ${c.rrf_score.toFixed(3)}` : (c.score ? `Score: ${parseFloat(c.score).toFixed(3)}` : "");
      const text = c.text || c.chunk_text || "";

      return `
        <div class="passage-card" id="passage-${idx}" data-idx="${idx}">
          <div class="passage-header">
            <strong>[${idx}] ${pid} § ${sec}</strong>
            <span>${score}</span>
          </div>
          <div class="passage-text">${escapeHtml(text)}</div>
        </div>
      `;
    }).join("");
  };

  const highlightPassage = (idx) => {
    selectTab("passages");
    document.querySelectorAll(".passage-card").forEach(c => c.classList.remove("highlighted"));
    const target = document.getElementById(`passage-${idx}`);
    if (target) {
      target.classList.add("highlighted");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const formatAnswerWithCitations = (text) => {
    // Replace [1], [2] with clickable pills
    const linked = text.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<span class="citation-tag" data-ref="${p1}">[${p1}]</span>`;
    });
    return linked.replace(/\n/g, "<br>");
  };

  const bindCitationClicks = () => {
    document.querySelectorAll(".citation-tag").forEach(tag => {
      tag.addEventListener("click", () => {
        const ref = tag.getAttribute("data-ref");
        highlightPassage(ref);
      });
    });
  };

  // ── Citation-Grounded Q&A (SSE Streaming + POST) ─────────────────────
  btnAsk.addEventListener("click", async () => {
    const q = qaQueryInput.value.trim();
    if (!q) return;

    btnAsk.textContent = "Asking...";
    btnAsk.disabled = true;
    answerContent.innerHTML = `<div class="loading-state">Retrieving relevant literature and generating grounded response...</div>`;
    qaMetaBar.style.display = "flex";
    metaCoverage.textContent = "Coverage: ...";
    metaCache.textContent = "Cache: ...";
    metaLatency.textContent = "Latency: ...";

    const useStream = chkStream.checked;
    const useCache = chkCache.checked;
    const t0 = performance.now();

    if (useStream) {
      try {
        const url = `/api/ask/stream?q=${encodeURIComponent(q)}&top_k=5&use_cache=${useCache}`;
        const eventSource = new EventSource(url);
        let accumulated = "";

        eventSource.addEventListener("sources", (e) => {
          const payload = JSON.parse(e.data);
          renderSourcePassages(payload.chunks || payload.references || []);
          metaCoverage.textContent = `Coverage: ${payload.coverage || "—"}`;
          metaCache.textContent = payload.cache_hit ? "Cache: Hit (0ms)" : "Cache: Miss";
          metaLatency.textContent = `Latency: ${payload.latency_ms} ms`;
          answerContent.innerHTML = "";
        });

        eventSource.addEventListener("token", (e) => {
          const payload = JSON.parse(e.data);
          accumulated += payload.token;
          currentAnswer = accumulated;
          answerContent.innerHTML = formatAnswerWithCitations(accumulated);
          bindCitationClicks();
        });

        eventSource.addEventListener("done", (e) => {
          eventSource.close();
          btnAsk.textContent = "Ask Grounded RAG";
          btnAsk.disabled = false;
          bindCitationClicks();
          loadHealth();
        });

        eventSource.onerror = (err) => {
          eventSource.close();
          btnAsk.textContent = "Ask Grounded RAG";
          btnAsk.disabled = false;
          if (!accumulated) {
            answerContent.innerHTML = `<div class="empty-state">Error during streaming. Switched to standard request...</div>`;
          }
        };
      } catch (err) {
        answerContent.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
        btnAsk.textContent = "Ask Grounded RAG";
        btnAsk.disabled = false;
      }
    } else {
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, top_k: 5, use_cache: useCache }),
        });
        const d = await res.json();
        currentAnswer = d.answer || "";
        answerContent.innerHTML = formatAnswerWithCitations(currentAnswer);
        bindCitationClicks();

        renderSourcePassages(d.chunks || d.references || []);
        metaCoverage.textContent = `Coverage: ${d.coverage || "—"}`;
        metaCache.textContent = d.cache_hit ? "Cache: Hit" : "Cache: Miss";
        metaLatency.textContent = `Latency: ${d.latency_ms || Math.round(performance.now() - t0)} ms`;
        loadHealth();
      } catch (err) {
        answerContent.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
      } finally {
        btnAsk.textContent = "Ask Grounded RAG";
        btnAsk.disabled = false;
      }
    }
  });

  // ── Export Actions ───────────────────────────────────────────────────
  btnExportMd.addEventListener("click", () => {
    if (!currentAnswer) return;
    navigator.clipboard.writeText(currentAnswer);
    btnExportMd.textContent = "✓ Copied!";
    setTimeout(() => btnExportMd.textContent = "Copy Markdown", 2000);
  });

  btnExportBib.addEventListener("click", () => {
    if (!currentSources.length) return;
    const bib = currentSources.map(c => {
      const pid = c.paper_id || "paper";
      const title = c.title || "Untitled";
      return `@article{arxiv_${pid.replace(/[^a-zA-Z0-9]/g, "_")},\n  title = {${title}},\n  eprint = {${pid}},\n  archivePrefix = {arXiv}\n}`;
    }).join("\n\n");
    navigator.clipboard.writeText(bib);
    btnExportBib.textContent = "✓ Copied BibTeX!";
    setTimeout(() => btnExportBib.textContent = "Export BibTeX", 2000);
  });

  // ── 2D Novelty Cone Canvas Visualizer ────────────────────────────────
  let currentVizData = null;
  const canvasTooltip = document.getElementById("canvasTooltip");

  const drawSubspace = (viz) => {
    currentVizData = viz;
    const ctx = subspaceCanvas.getContext("2d");
    const w = subspaceCanvas.width;
    const h = subspaceCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Padding & dimensions (generous right/top margin to prevent clipping)
    const padLeft = 60;
    const padRight = 50;
    const padTop = 35;
    const padBottom = 45;
    const cx = padLeft;
    const cy = h - padBottom;
    const scale = Math.min(w - padLeft - padRight, h - padTop - padBottom);

    // Draw Subtle Grid & Quadrant Background
    ctx.fillStyle = "rgba(15, 23, 42, 0.4)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, scale, -Math.PI / 2, 0, false);
    ctx.closePath();
    ctx.fill();

    // Concentric grid arcs at r = 0.5 and r = 1.0
    [0.5, 1.0].forEach(r => {
      ctx.strokeStyle = r === 1.0 ? "rgba(100, 116, 139, 0.45)" : "rgba(255, 255, 255, 0.08)";
      ctx.setLineDash(r === 1.0 ? [4, 4] : [2, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, scale * r, -Math.PI / 2, 0, false);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Grid ticks on X-axis (0.5, 1.0)
    [0.5, 1.0].forEach(v => {
      const tx = cx + v * scale;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(tx, cy - 3);
      ctx.lineTo(tx, cy + 3);
      ctx.stroke();

      ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(v.toFixed(1), tx, cy + 14);
    });

    // Grid ticks on Y-axis (0.5, 1.0)
    [0.5, 1.0].forEach(v => {
      const ty = cy - v * scale;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 3, ty);
      ctx.lineTo(cx + 3, ty);
      ctx.stroke();

      ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(v.toFixed(1), cx - 6, ty + 3);
    });

    // Draw Main Axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(w - 20, cy); // X-axis
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, 15);     // Y-axis
    ctx.stroke();

    // Axis Arrows
    ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
    ctx.beginPath();
    ctx.moveTo(w - 18, cy - 3);
    ctx.lineTo(w - 12, cy);
    ctx.lineTo(w - 18, cy + 3);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(cx - 3, 18);
    ctx.lineTo(cx, 12);
    ctx.lineTo(cx + 3, 18);
    ctx.fill();

    // Axis Titles (Clean, Uncluttered, Semantic)
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
    ctx.font = "600 11px Inter, sans-serif";
    ctx.fillText("e₂: Blend Direction (√Δ)", cx + 10, 24);

    ctx.textAlign = "right";
    ctx.fillText("e₁: Dominant Parent (m)", w - 24, cy - 8);

    // Unit circle label
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText("‖x‖ = 1", cx + scale * 0.73, cy - scale * 0.73);

    if (!viz || !viz.ok || !viz.points) return;

    const points = {};
    const refPoints = [];
    viz.points.forEach(p => {
      points[p.id] = p;
      if (p.role === "Cited Reference") refPoints.push(p);
    });

    // Helper: Draw pill tag with background
    const drawPillTag = (text, x, y, bgColor, textColor, borderColor) => {
      ctx.font = "500 11px Inter, sans-serif";
      const metrics = ctx.measureText(text);
      const pw = metrics.width + 12;
      const ph = 20;
      const rx = x;
      const ry = y - ph / 2;

      ctx.fillStyle = bgColor;
      ctx.beginPath();
      ctx.roundRect(rx, ry, pw, ph, 4);
      ctx.fill();

      if (borderColor) {
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      ctx.fillStyle = textColor;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, rx + 6, y);
      ctx.textBaseline = "alphabetic"; // reset
    };

    // Draw Cited Reference Dots (Scattered prior art in bibliography)
    refPoints.forEach(rp => {
      const rpx = cx + rp.x * scale;
      const rpy = cy - rp.y * scale;
      if (rpx >= cx - 10 && rpy <= cy + 10) {
        if (rp.is_contributor || (rp.weight && rp.weight > 1e-4)) {
          // Foundational contributor that actively shapes the blue cone projection
          ctx.fillStyle = "#818cf8";
          ctx.beginPath();
          ctx.arc(rpx, rpy, 4.5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = "rgba(129, 140, 248, 0.5)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else {
          // Background cited prior art (scattered grey dots)
          ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
          ctx.beginPath();
          ctx.arc(rpx, rpy, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    });

    // 1. Dominant Parent at (1, 0)
    const px = cx + scale;
    const py = cy;

    // Vector to dominant parent
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.stroke();

    // Dominant Parent Dot
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, 2 * Math.PI);
    ctx.fill();

    // Outer glow ring
    ctx.strokeStyle = "rgba(244, 63, 94, 0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, 2 * Math.PI);
    ctx.stroke();

    drawPillTag("Parent (1, 0)", px - 35, py + 22, "rgba(15, 23, 42, 0.9)", "#f43f5e", "rgba(244, 63, 94, 0.4)");

    // 2. Cone Projection x̂ at (m, sqrt(Delta))
    const recon = points["reconstruction"];
    if (recon) {
      const rx = cx + recon.x * scale;
      const ry = cy - recon.y * scale;

      // Synergy Cone Blend triangle fill
      ctx.fillStyle = "rgba(6, 182, 212, 0.08)";
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(px, py);
      ctx.lineTo(rx, ry);
      ctx.closePath();
      ctx.fill();

      // Synergy blend projection guide line (vertical from (m, 0) to (m, sqrt(Delta)))
      ctx.strokeStyle = "rgba(6, 182, 212, 0.5)";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx + recon.x * scale, cy);
      ctx.lineTo(rx, ry);
      ctx.moveTo(cx, ry);
      ctx.lineTo(rx, ry);
      ctx.stroke();
      ctx.setLineDash([]);

      // Vector from origin to x_hat
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(rx, ry);
      ctx.stroke();

      // Cone Projection Dot
      ctx.fillStyle = "#06b6d4";
      ctx.beginPath();
      ctx.arc(rx, ry, 6.5, 0, 2 * Math.PI);
      ctx.fill();

      // Outer ring
      ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(rx, ry, 10, 0, 2 * Math.PI);
      ctx.stroke();

      // Smart positioning of x̂ label to prevent boundary clipping
      const tagText = `x̂ (${recon.x.toFixed(2)}, ${recon.y.toFixed(2)})`;
      const tagX = (rx > w - 110) ? rx - 105 : rx + 12;
      const tagY = (ry < 50) ? ry + 18 : ry - 14;
      drawPillTag(tagText, tagX, tagY, "rgba(15, 23, 42, 0.9)", "#38bdf8", "rgba(6, 182, 212, 0.5)");

      // 3. Novelty Residual Line from x_hat radially to unit circle
      const r_norm = Math.sqrt(recon.x * recon.x + recon.y * recon.y);
      if (r_norm > 1e-4) {
        const circle_x = cx + (recon.x / r_norm) * scale;
        const circle_y = cy - (recon.y / r_norm) * scale;

        ctx.strokeStyle = "#f59e0b";
        ctx.setLineDash([3, 2]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(rx, ry);
        ctx.lineTo(circle_x, circle_y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Unit circle departure dot
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(circle_x, circle_y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  };

  // Canvas Mouse Hover Tooltip Interaction
  subspaceCanvas.addEventListener("mousemove", (e) => {
    if (!currentVizData || !currentVizData.points || !canvasTooltip) return;

    const rect = subspaceCanvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (subspaceCanvas.width / rect.width);
    const my = (e.clientY - rect.top) * (subspaceCanvas.height / rect.height);

    const padLeft = 60;
    const padRight = 50;
    const padTop = 35;
    const padBottom = 45;
    const cx = padLeft;
    const cy = subspaceCanvas.height - padBottom;
    const scale = Math.min(subspaceCanvas.width - padLeft - padRight, subspaceCanvas.height - padTop - padBottom);

    let nearest = null;
    let minDist = 18; // pixel radius threshold

    currentVizData.points.forEach(p => {
      const px = (p.id === "Dominant Parent" || p.role === "Dominant Parent") ? (cx + scale) : (cx + p.x * scale);
      const py = (p.id === "Dominant Parent" || p.role === "Dominant Parent") ? cy : (cy - p.y * scale);
      const d = Math.hypot(mx - px, my - py);
      if (d < minDist) {
        minDist = d;
        nearest = p;
      }
    });

    if (nearest) {
      const title = nearest.title || nearest.label || "Reference Paper";
      const role = nearest.role || "Subspace Element";
      const wStr = (nearest.weight !== undefined) ? `Weight: ${nearest.weight.toFixed(4)}` : "";
      const shStr = (nearest.share_pct !== undefined) ? `${nearest.share_pct.toFixed(1)}% share` : "";
      const coordStr = `(${nearest.x.toFixed(2)}, ${nearest.y.toFixed(2)})`;

      canvasTooltip.innerHTML = `
        <div style="font-weight: 600; color: #38bdf8; margin-bottom: 2px;">${escapeHtml(role)}</div>
        <div style="font-size: 11px; margin-bottom: 4px; color: #fff; line-height: 1.3;">${escapeHtml(title)}</div>
        <div style="font-size: 10px; color: var(--text-dim); display: flex; justify-content: space-between; gap: 8px;">
          <span>Coords: ${coordStr}</span>
          <span>${shStr || wStr}</span>
        </div>
      `;
      canvasTooltip.style.left = `${Math.min(rect.width - 240, Math.max(10, e.clientX - rect.left + 12))}px`;
      canvasTooltip.style.top = `${Math.max(10, e.clientY - rect.top - 20)}px`;
      canvasTooltip.style.display = "block";
    } else {
      canvasTooltip.style.display = "none";
    }
  });

  subspaceCanvas.addEventListener("mouseleave", () => {
    if (canvasTooltip) canvasTooltip.style.display = "none";
  });

  btnComputeNovelty.addEventListener("click", async () => {
    const aid = noveltyArxivIdInput.value.trim();
    if (!aid) return;

    btnComputeNovelty.textContent = "Computing...";
    btnComputeNovelty.disabled = true;

    try {
      const res = await fetch(`/api/novelty?arxiv_id=${encodeURIComponent(aid)}`);
      const profile = await res.json();

      if (!res.ok || profile.error || profile.ok === false) {
        attributionTable.innerHTML = `<div class="empty-state-sm">Error: ${profile.error || profile.message || "Computation failed."}</div>`;
        return;
      }

      // Populate typology and coverage badges
      const typologyHeader = document.getElementById("typologyHeader");
      const typologyPill = document.getElementById("typologyPill");
      const coveragePill = document.getElementById("coveragePill");
      if (typologyHeader && typologyPill && coveragePill) {
        typologyHeader.style.display = "flex";
        const thint = profile.typology_hint || "Novel Extension";
        typologyPill.textContent = thint;
        if (thint.toLowerCase().includes("blend") || thint.toLowerCase().includes("synerg")) {
          typologyPill.style.background = "rgba(147, 51, 234, 0.2)";
          typologyPill.style.color = "#c084fc";
          typologyPill.style.borderColor = "rgba(147, 51, 234, 0.4)";
        } else if (thint.toLowerCase().includes("linear") || thint.toLowerCase().includes("extension")) {
          typologyPill.style.background = "rgba(6, 182, 212, 0.2)";
          typologyPill.style.color = "#38bdf8";
          typologyPill.style.borderColor = "rgba(6, 182, 212, 0.4)";
        } else {
          typologyPill.style.background = "rgba(245, 158, 11, 0.2)";
          typologyPill.style.color = "#fbbf24";
          typologyPill.style.borderColor = "rgba(245, 158, 11, 0.4)";
        }
        coveragePill.textContent = profile.coverage_warning ? "⚠ Partial Abstract Coverage" : "✓ Full Abstract Coverage";
        coveragePill.title = profile.coverage_warning || "All reference abstracts loaded into geometry";
      }

      // Populate metrics cards
      mVal.textContent = (profile.m !== undefined && profile.m !== null) ? profile.m.toFixed(3) : "—";
      deltaVal.textContent = (profile.Delta !== undefined && profile.Delta !== null) ? profile.Delta.toFixed(3) : "—";
      r2Val.textContent = (profile.R2_plus !== undefined && profile.R2_plus !== null) ? profile.R2_plus.toFixed(3) : "—";
      resVal.textContent = (profile.viz_2d && profile.viz_2d.metrics && profile.viz_2d.metrics.residual_distance !== undefined)
        ? profile.viz_2d.metrics.residual_distance.toFixed(3)
        : ((profile.R2_plus !== undefined && profile.R2_plus !== null) ? Math.sqrt(Math.max(0, 1 - profile.R2_plus)).toFixed(3) : "—");

      // Draw 2D canvas
      if (profile.viz_2d) {
        drawSubspace(profile.viz_2d);
      }

      // Populate NNLS attribution table
      const attr = profile.attribution || [];
      if (!attr.length) {
        attributionTable.innerHTML = `<div class="empty-state-sm">Zero reference weights or no references extracted.</div>`;
      } else {
        attributionTable.innerHTML = attr.map(r => {
          const pid = r.paper_id || r.ref_id || "Unknown";
          const cleanPid = pid.startsWith("arxiv:") ? pid.slice(6) : pid;
          const title = r.title && r.title !== pid ? r.title : "Foundational Reference";
          const yearStr = r.year ? `${r.year}` : "";
          const shareVal = (r.share !== undefined && r.share !== null) ? (r.share * 100).toFixed(1) : (r.weight ? (r.weight * 100).toFixed(1) : "—");
          const weightVal = (r.weight !== undefined && r.weight !== null) ? r.weight.toFixed(4) : "—";
          const provider = r.provider || (cleanPid.match(/^\d{4}\.\d{4,5}/) ? "arXiv" : "Semantic Scholar");
          const providerClass = provider.toLowerCase().replace(/[^a-z]/g, "");
          const url = r.url || (cleanPid.match(/^\d{4}\.\d{4,5}/) ? `https://arxiv.org/abs/${cleanPid}` : `https://www.semanticscholar.org/paper/${cleanPid.replace("s2:", "")}`);
          const displayId = (cleanPid.startsWith("s2:") || cleanPid.length > 25) ? cleanPid.slice(0, 16) + "…" : cleanPid;

          return `
            <div class="attr-row">
              <div class="attr-info">
                <span class="attr-title" title="${escapeHtml(title)}">${escapeHtml(title)}</span>
                <div class="attr-meta-line">
                  <span class="provider-badge ${providerClass}">${escapeHtml(provider)}</span>
                  <a href="${escapeHtml(url)}" target="_blank" rel="noopener" class="attr-link" title="Open official paper page (${escapeHtml(provider)})">
                    ${escapeHtml(displayId)} ↗
                  </a>
                  ${yearStr ? `<span class="attr-year">• ${escapeHtml(yearStr)}</span>` : ""}
                </div>
              </div>
              <div class="attr-values">
                <span class="attr-weight">λ = ${weightVal}</span>
                <span class="attr-share">${shareVal}% share</span>
              </div>
            </div>
          `;
        }).join("");
      }
    } catch (e) {
      attributionTable.innerHTML = `<div class="empty-state-sm">Error: ${e.message}</div>`;
    } finally {
      btnComputeNovelty.textContent = "Project Subspace";
      btnComputeNovelty.disabled = false;
    }
  });

  // Initial subspace canvas render
  drawSubspace(null);

  // ── Landmark Hubs Tab ────────────────────────────────────────────────
  const loadHubs = async () => {
    try {
      const res = await fetch("/api/hubs");
      const d = await res.json();
      const hubs = d.hubs || [];
      if (!hubs.length) {
        hubsList.innerHTML = `<div class="empty-state">No landmark hubs found in database.</div>`;
        return;
      }

      hubsList.innerHTML = hubs.map(h => `
        <div class="hub-card">
          <div class="hub-header">
            <strong>${h.title}</strong>
            <span class="score-badge">${h.citation_count.toLocaleString()} cites</span>
          </div>
          <div class="paper-meta">
            <span>ID: ${h.arxiv_id || h.canonical_id}</span>
            <span>Year: ${h.year}</span>
          </div>
          <div class="card-actions">
            <button class="btn-card-action" onclick="document.getElementById('noveltyArxivId').value='${h.arxiv_id}'; document.getElementById('tabBtnSubspace').click(); document.getElementById('btnComputeNovelty').click();">Inspect Subspace</button>
          </div>
        </div>
      `).join("");
    } catch (e) {
      hubsList.innerHTML = `<div class="empty-state">Error loading hubs: ${e.message}</div>`;
    }
  };

  btnRefreshHubs.addEventListener("click", loadHubs);

  // ── User Guide Modal ────────────────────────────────────────────────
  const btnOpenGuide = document.getElementById("btnOpenGuide");
  const btnCloseGuide = document.getElementById("btnCloseGuide");
  const guideModal = document.getElementById("guideModal");
  if (btnOpenGuide && guideModal) {
    btnOpenGuide.addEventListener("click", () => {
      guideModal.style.display = "flex";
    });
    if (btnCloseGuide) {
      btnCloseGuide.addEventListener("click", () => {
        guideModal.style.display = "none";
      });
    }
    guideModal.addEventListener("click", (e) => {
      if (e.target === guideModal) {
        guideModal.style.display = "none";
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && guideModal.style.display === "flex") {
        guideModal.style.display = "none";
      }
    });
  }

  // ── Helper ───────────────────────────────────────────────────────────
  function escapeHtml(str) {
    return (str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
});
