// Pesto dashboard — live video stream + farmer-friendly render.
// Captures a frame from a continuously-playing <video> on a fixed cadence
// and POSTs it to /api/run-frame. Bboxes / heatmap / drift cone overlay
// the live video without ever pausing it.

const CONDITION_COLOR = {
  healthy_crop: "#82c85a",
  weed: "#d35f9c",
  disease: "#d39c5f",
  nutrient_stress: "#c8c85a",
  water_stress: "#5fc8d3",
  pest_damage: "#b35fd3",
  ambiguous: "#888",
};

const CONDITION_FRIENDLY = {
  healthy_crop: "HEALTHY",
  weed: "WEED",
  disease: "DISEASED",
  nutrient_stress: "NUTRIENT-STRESSED",
  water_stress: "WATER-STRESSED",
  pest_damage: "PEST DAMAGE",
  ambiguous: "MIXED-SIGNAL",  // never display "uncertain" — show next-best instead
};

const ACTION_FRIENDLY = {
  no_action:           { verb: "leave alone",        icon: '<i class="ph ph-check-circle"></i>',       cls: "healthy"  },
  laser_zap:           { verb: "laser zap",          icon: '<i class="ph-bold ph-lightning"></i>',     cls: "weed"     },
  targeted_spray:      { verb: "spray herbicide",    icon: '<i class="ph ph-spray-bottle"></i>',       cls: "weed"     },
  targeted_fungicide:  { verb: "apply fungicide",    icon: '<i class="ph ph-flask"></i>',              cls: "disease"  },
  targeted_irrigation: { verb: "irrigate",           icon: '<i class="ph-fill ph-drop"></i>',          cls: "water"    },
  foliar_nutrient:     { verb: "fertilize",          icon: '<i class="ph ph-leaf"></i>',               cls: "nutrient" },
  human_review:        { verb: "flag for review",    icon: '<i class="ph ph-question"></i>',           cls: "review"   },
  rescan_higher_res:   { verb: "scan closer",        icon: '<i class="ph ph-magnifying-glass"></i>',   cls: "review"   },
};

const PARADIGMS = {
  weed_detector: "ML",
  disease_classifier: "ML",
  segmentation: "CV",
  health_classifier: "ML",
  weather_prior: "PHYSICS",
  anomaly_detector: "CV",
  growth_stage: "ML",
  vlm_reasoner: "ML/LLM",
  water_balance: "BIOPHYSICS",
  pesticide_fate: "PHYSICS",
  ecological_dynamics: "ECOLOGY",
  skeptic: "META",
  controller: "META",
};

// Concrete model / method backing each agent — what the user clicks to verify.
// Strings here mirror the actual implementations under pesto/agents/* and
// pesto/llm_config.py. URLs point to the upstream source.
const MODEL_REFS = {
  weed_detector: {
    label: "foduucom/plant-leaf-detection-and-classification (YOLOv8)",
    url: "https://huggingface.co/foduucom/plant-leaf-detection-and-classification",
  },
  disease_classifier: {
    label: "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    url: "https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
  },
  segmentation: {
    label: "SAM bailout · YOLO bbox crop (no-GPU mode)",
    url: "https://github.com/facebookresearch/segment-anything",
  },
  health_classifier: {
    label: "Diginsa/Plant-Disease-Detection-Project (ViT)",
    url: "https://huggingface.co/Diginsa/Plant-Disease-Detection-Project",
  },
  vlm_reasoner: {
    label: "OpenAI AG2 (AssistantAgent)",
    url: "https://platform.openai.com/docs/models",
  },
  water_balance: {
    label: "FAO-56 Penman–Monteith (rule-based)",
    url: "https://www.fao.org/3/x0490e/x0490e00.htm",
  },
  pesticide_fate: {
    label: "Gaussian plume drift model (rule-based)",
    url: "https://en.wikipedia.org/wiki/Atmospheric_dispersion_modeling",
  },
  ecological_dynamics: {
    label: "Lotka–Volterra predator–prey (rule-based)",
    url: "https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations",
  },
  skeptic: {
    label: "OpenAI AG2 (AssistantAgent)",
    url: "https://platform.openai.com/docs/models",
  },
  weather_prior: {
    label: "Open-Meteo API (contextual priors)",
    url: "https://open-meteo.com/",
  },
  anomaly_detector: {
    label: "DINOv2 + PatchCore (anomaly scoring)",
    url: "https://arxiv.org/abs/2304.07193",
  },
  growth_stage: {
    label: "ViT growth-stage classifier (heuristic fallback)",
    url: "#",
  },
};

const INTERVENTIONS = ["no_action", "laser_zap", "targeted_spray",
                       "targeted_fungicide", "targeted_irrigation",
                       "foliar_nutrient", "human_review", "rescan_higher_res"];

const $ = (id) => document.getElementById(id);
const t0 = performance.now();
function fmtT() { return ((performance.now() - t0) / 1000).toFixed(2); }

// --- WebSocket ------------------------------------------------------------

let ws = null;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => $("conn-state").textContent = "connected";
  ws.onclose = () => {
    $("conn-state").textContent = "disconnected";
    setTimeout(connectWS, 1500);
  };
  ws.onmessage = (ev) => {
    try {
      const { kind, payload } = JSON.parse(ev.data);
      handleEvent(kind, payload);
    } catch (e) { console.error(e); }
  };
}
connectWS();

// --- State ----------------------------------------------------------------

const state = {
  imageNatural: { w: 0, h: 0 },
  plants: {},
  agents: {},
  interventions: {},
  flaggedActions: [],
  fieldState: { wind_dir_deg: 270, wind_speed_m_s: 2.0 },
  ecology: {},
  physics: {},
  biophysics: null,
  // Sprint 1-4 extensions
  weather: null,
  growthStages: {},
  anomalyScores: {},
  anomalyThreshold: 0.3,
  explainOverlay: null,
  explainVisible: false,
  ragContext: [],
  activeLearning: { total_queued: 0, unlabeled: 0, total_labeled: 0, trigger_breakdown: {} },
  debateTurns: 0,
  debateConverged: false,
  temporalChanges: {},
  inferenceMode: { vlm: "api", skeptic: "api" },
  busy: false,
  stream: { running: false, intervalMs: 1500, deepEvery: 5, frameTick: 0 },
  manifest: null,
  manifestData: null,
  liveField: null,
  // Per-frame species labels (from dataset GT, class:"weed"|"crop").
  species: {},
  totals: { intervention_counts: {}, frames: 0, plants_seen: 0,
            phys_vetos: 0, eco_vetos: 0, chem_saved_ml: 0,
            weeds_detected: 0, predators_saved_pct: 0 },
  whyNotSpray: null,
  // Latest frame's actions, always shown so the panel never goes empty.
  lastFrameActions: [],
  // Stream gating — disabled until first warmup completes.
  warmedUp: false,
};

// Unhealthy actions worth flagging persistently.
const FLAG_ACTIONS = new Set([
  "laser_zap", "targeted_spray", "targeted_fungicide",
  "targeted_irrigation", "foliar_nutrient",
]);

// Capture a cropped snapshot of a plant's bbox with highlighted border
function capturePlantCrop(plantId) {
  const v = $("field-video");
  const p = state.plants[plantId];
  if (!v || !p || !v.videoWidth) return null;
  const [x1, y1, x2, y2] = p.bbox;
  const pad = 20; // extra pixels around bbox
  const sx1 = Math.max(0, x1 - pad);
  const sy1 = Math.max(0, y1 - pad);
  const sx2 = Math.min(state.imageNatural.w, x2 + pad);
  const sy2 = Math.min(state.imageNatural.h, y2 + pad);
  const cw = sx2 - sx1, ch = sy2 - sy1;
  if (cw < 5 || ch < 5) return state.lastSnapshot || null;
  const c = document.createElement("canvas");
  // Scale up for clarity
  const scale = Math.min(200 / cw, 200 / ch, 3);
  c.width = Math.round(cw * scale);
  c.height = Math.round(ch * scale);
  const ctx = c.getContext("2d");
  ctx.drawImage(v, sx1, sy1, cw, ch, 0, 0, c.width, c.height);
  // Draw bbox highlight
  const bx = (x1 - sx1) * scale, by = (y1 - sy1) * scale;
  const bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;
  const color = CONDITION_COLOR[p.top1] || "#82c85a";
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(bx, by, bw, bh);
  // Semi-transparent fill
  ctx.fillStyle = color.replace(")", ",0.12)").replace("rgb", "rgba").replace("#", "");
  // Use hex→rgba
  const r = parseInt(color.slice(1,3),16), g = parseInt(color.slice(3,5),16), b = parseInt(color.slice(5,7),16);
  ctx.fillStyle = `rgba(${r},${g},${b},0.12)`;
  ctx.fillRect(bx, by, bw, bh);
  // Label
  ctx.font = `bold ${Math.round(11*scale)}px monospace`;
  ctx.fillStyle = color;
  ctx.fillText((CONDITION_FRIENDLY[p.top1] || p.top1), bx + 4, by - 4 > 12 ? by - 4 : by + 14);
  return c.toDataURL("image/jpeg", 0.8);
}

// --- Video source --------------------------------------------------------

// The manifest tracks GT bbox labels for the aerial frames that compose
// `field_stream.mp4`. Other clips don't have GT — switching to them nulls
// `state.manifest` so streaming inference falls through to /api/run-frame.
const MANIFEST_VIDEO_BASENAME = "field_stream.mp4";

async function loadVideoSource() {
  // Manifest first so setActiveVideo() can apply it to the right clip.
  const m = await fetch("/api/manifest").then(r => r.json()).catch(() => null);
  state.manifestData = (m && m.frame_count) ? m : null;

  const r = await fetch("/api/demo-videos");
  const data = await r.json();
  const sel = $("video-select");
  if (sel) sel.innerHTML = "";

  if (data.videos && data.videos.length) {
    if (sel) {
      for (const vid of data.videos) {
        const opt = document.createElement("option");
        opt.value = vid.filename;
        opt.dataset.url = vid.url;
        opt.textContent = vid.filename
          .replace(/\.(mp4|webm)$/i, "")
          .replace(/[_-]/g, " ");
        sel.appendChild(opt);
      }
      sel.addEventListener("change", () => {
        const opt = sel.options[sel.selectedIndex];
        setActiveVideo(opt.value, opt.dataset.url);
      });
    }
    const def = data.videos.find(v => v.filename === MANIFEST_VIDEO_BASENAME)
             || data.videos[0];
    if (sel) sel.value = def.filename;
    setActiveVideo(def.filename, def.url);
  }
  // Models lazy-load on the first frame. Skip the pre-warm so the user can
  // start streaming immediately — the first inference takes a beat longer,
  // but they're not staring at a disabled button.
  state.warmedUp = true;
  setStreamButtonState("ready");
  $("run-state").textContent = "ready";
}

function setActiveVideo(filename, url) {
  if (state.stream && state.stream.running) stopStream();
  const v = $("field-video");
  v.src = url;
  state.manifest = (filename === MANIFEST_VIDEO_BASENAME)
    ? state.manifestData
    : null;
}

function setStreamButtonState(s) {
  const b = $("stream-btn");
  if (s === "ready") {
    b.disabled = false;
    b.innerHTML = '<i class="ph-fill ph-play text-[13px]"></i><span>START STREAM</span>';
    b.classList.remove("streaming");
  } else if (s === "running") {
    b.disabled = false;
    b.innerHTML = '<i class="ph-fill ph-stop text-[13px]"></i><span>STOP</span>';
    b.classList.add("streaming");
  }
}

function currentFrameIndex() {
  if (!state.manifest) return null;
  const v = $("field-video");
  if (!v.duration) return 0;
  // Each source image plays for (frames_per_source / fps) seconds.
  const secPerFrame = (state.manifest.frames_per_source || 2)
                    / (state.manifest.fps || 4);
  const idx = Math.floor((v.currentTime || 0) / secPerFrame);
  return Math.max(0, Math.min(state.manifest.frame_count - 1, idx));
}

// --- Event handlers ------------------------------------------------------

function handleEvent(kind, payload) {
  msg(kind, payload);
  switch (kind) {
    case "run_started":         resetForNewFrame(payload); break;
    case "latent_initialised":  onLatent(payload); break;
    case "constraint":          onConstraint(payload); break;
    case "physics_assessment":  onPhysics(payload); break;
    case "ecology_trajectory":  onEcology(payload); break;
    case "action":              onAction(payload); break;
    case "hypotheses":          onHypotheses(payload); break;
    case "done":                onDone(payload); break;
    // Sprint 1-4 events
    case "visual_explanation":  onVisualExplanation(payload); break;
    case "temporal_diff":       onTemporalDiff(payload); break;
    case "active_learning_update": onActiveLearning(payload); break;
    case "debate_turn":         onDebateTurn(payload); break;
    case "rag_context":         onRagContext(payload); break;
    case "llm_phase":           onLLMPhase(payload); break;
  }
}

function resetForNewFrame(payload) {
  state.plants = {};
  // Per-frame action accumulator — used for the always-visible RECOMMENDED
  // ACTIONS panel.
  state.lastFrameActions = [];
  for (const a of Object.keys(state.agents)) {
    if (state.agents[a] === "ok") state.agents[a] = "running";
  }
  state.physics = {};
  state.ecology = {};
  state.biophysics = null;
  state.growthStages = {};
  state.anomalyScores = {};
  state.temporalChanges = {};
  state.debateTurns = 0;
  state.debateConverged = false;
  if (payload && payload.field_state) {
    state.liveField = payload.field_state;
    state.fieldState = {
      wind_dir_deg: payload.field_state.wind_dir_deg,
      wind_speed_m_s: payload.field_state.wind_speed_m_s,
    };
  }
  if (payload && payload.species_by_pid) {
    state.species = payload.species_by_pid;  // {pid: "weed"|"crop"}
  } else {
    state.species = {};
  }
  renderAgents();
  // NB: state.interventions stays cumulative — never reset.
  renderInterventions();
  renderActionsSummary();
  renderWind();
  $("bbox-layer").innerHTML = "";
}

function renderWind() {
  const wind = state.fieldState.wind_dir_deg ?? 270;
  const speed = state.fieldState.wind_speed_m_s;
  $("wind-label").textContent = (speed !== undefined && speed !== null)
    ? `${(+speed).toFixed(1)} m/s @ ${Math.round(wind)}°`
    : "— m/s @ —";
  $("wind-arrow").style.transform = `rotate(${(wind + 180) % 360}deg)`;
}

function onLatent(latent) {
  state.plants = {};
  for (const p of latent.plants) {
    state.plants[p.plant_id] = {
      bbox: p.bbox,
      top1: p.top_k && p.top_k[0] ? p.top_k[0][0] : "ambiguous",
      top1_p: p.top_k && p.top_k[0] ? p.top_k[0][1] : 0.0,
      top2: p.top_k && p.top_k[1] ? p.top_k[1][0] : null,
      top2_p: p.top_k && p.top_k[1] ? p.top_k[1][1] : 0.0,
      entropy: p.entropy,
      agents: new Set(),
    };
  }
  state.imageNatural = { w: latent.image_shape[1], h: latent.image_shape[0] };
  renderBoxes();
  renderHeatmap();
}

function onConstraint(c) {
  state.agents[c.sender] = "ok";
  for (const pidStr in c.per_plant_log_likelihoods) {
    const pid = +pidStr;
    if (state.plants[pid]) state.plants[pid].agents.add(c.sender);
  }
  if (c.sender === "water_balance") {
    state.biophysics = c.metadata || null;
    renderBiophysics();
  }
  if (c.sender === "weather_prior" && c.metadata) {
    state.weather = c.metadata;
    renderWeather();
  }
  if (c.sender === "anomaly_detector" && c.metadata) {
    state.anomalyScores = c.metadata.per_plant_anomaly_scores || {};
    state.anomalyThreshold = c.metadata.anomaly_threshold || 0.3;
  }
  if (c.sender === "growth_stage" && c.metadata) {
    const perPlant = c.metadata.per_plant_growth || {};
    for (const [pid, info] of Object.entries(perPlant)) {
      state.growthStages[+pid] = info;
    }
    renderGrowthAnomaly();
  }
  renderAgents();
}

function onPhysics(p) {
  state.agents.pesticide_fate = "ok";
  if (!state.physics[p.plant_id]) state.physics[p.plant_id] = {};
  state.physics[p.plant_id][p.action_type] = p;
  // Track the frame's worst spray scenario for the WHY NOT SPRAY card.
  if ((p.action_type === "targeted_spray" || p.action_type === "targeted_fungicide")
      && (p.hazard_score ?? 0) > 0.4) {
    if (!state._frameWorst || p.hazard_score > state._frameWorst.phys_hazard) {
      state._frameWorst = state._frameWorst || { plant_id: p.plant_id };
      state._frameWorst.plant_id = p.plant_id;
      state._frameWorst.phys_hazard = p.hazard_score;
      state._frameWorst.drift_max_ppm = p.hazard_breakdown?.drift_max_ppm ?? 0;
    }
  }
  renderAgents();
}

function onEcology(p) {
  state.agents.ecological_dynamics = "ok";
  if (!state.ecology[p.plant_id]) state.ecology[p.plant_id] = {};
  state.ecology[p.plant_id][p.action_type] = p;
  if ((p.action_type === "targeted_spray" || p.action_type === "targeted_fungicide")
      && (p.ecological_cost_score ?? 0) > 0.5) {
    state._frameWorst = state._frameWorst || { plant_id: p.plant_id };
    if (p.ecological_cost_score > (state._frameWorst.eco_cost ?? 0)) {
      state._frameWorst.plant_id = p.plant_id;
      state._frameWorst.eco_cost = p.ecological_cost_score;
      state._frameWorst.predator_drop_pct = p.cost_breakdown?.predator_drop_pct_d14 ?? 0;
    }
  }
  renderAgents();
}

function onAction(a) {
  // Both totals and the rolling-frame action log get this hit.
  state.interventions[a.action_type] = (state.interventions[a.action_type] || 0) + 1;
  state.totals.intervention_counts[a.action_type] =
    (state.totals.intervention_counts[a.action_type] || 0) + 1;
  state.lastFrameActions.push(a);
  if (state.plants[a.plant_id]) {
    state.plants[a.plant_id].action = a.action_type;
    state.plants[a.plant_id].utility = a.expected_utility;
    state.plants[a.plant_id].physics_hazard = a.physics_hazard_score || 0;
  }
  // Track weed-detected count — check both top1 classification AND dataset
  // species label. Coerce keys to both string and number for JSON safety.
  {
    const numPid = +a.plant_id;
    const strPid = String(a.plant_id);
    const pl = state.plants[numPid] || state.plants[strPid];
    const isWeed = (pl && pl.top1 === "weed")
                || state.species[strPid] === "weed"
                || state.species[numPid] === "weed";
    if (isWeed) state.totals.weeds_detected++;
  }
  // Add ALL actions to persistent list (not just FLAG_ACTIONS).
  {
    const plant = state.plants[a.plant_id] || {};
    const condition = plant.top1 || "ambiguous";
    const phys_hazard = (state.physics[a.plant_id]?.targeted_spray?.hazard_score
                       ?? state.physics[a.plant_id]?.targeted_fungicide?.hazard_score
                       ?? 0);
    const eco_cost = (state.ecology[a.plant_id]?.targeted_spray?.ecological_cost_score
                    ?? state.ecology[a.plant_id]?.targeted_fungicide?.ecological_cost_score
                    ?? 0);
    const drift_max = state.physics[a.plant_id]?.targeted_spray?.hazard_breakdown?.drift_max_ppm
                   ?? state.physics[a.plant_id]?.targeted_fungicide?.hazard_breakdown?.drift_max_ppm
                   ?? 0;
    const pred_drop = state.ecology[a.plant_id]?.targeted_spray?.cost_breakdown?.predator_drop_pct_d14
                   ?? state.ecology[a.plant_id]?.targeted_fungicide?.cost_breakdown?.predator_drop_pct_d14
                   ?? 0;
    // Capture per-plant cropped snapshot with bbox highlighting
    const cropSnap = capturePlantCrop(a.plant_id);
    state.flaggedActions.push({
      ts: performance.now(),
      tsDate: new Date(),
      videoTime: state.lastFrameTime,
      snapshot: cropSnap || state.lastSnapshot,
      action: a.action_type,
      condition,
      plant_id: a.plant_id,
      utility: a.expected_utility,
      phys_hazard, eco_cost,
      drift_max_ppm: drift_max,
      predator_drop_pct: pred_drop,
      conf: plant.top1_p || 0,
      growthStage: state.growthStages[a.plant_id]?.growth_stage || null,
    });
    if (FLAG_ACTIONS.has(a.action_type)) {
      if (phys_hazard > 0.4) state.totals.phys_vetos++;
      if (eco_cost > 0.5) state.totals.eco_vetos++;
      if (a.action_type === "laser_zap") state.totals.chem_saved_ml += 5;
    }
    if (state.flaggedActions.length > 80) state.flaggedActions.shift();
    if (phys_hazard > 0.4 || eco_cost > 0.5) {
      state.whyNotSpray = {
        plant_id: a.plant_id,
        action: a.action_type,
        condition,
        phys_hazard, eco_cost, drift_max_ppm: drift_max,
        predator_drop_pct: pred_drop,
        snapshot: cropSnap || state.lastSnapshot,
        wind: state.liveField,
      };
      renderWhyNotSpray();
    }
  }
  renderTotals();
  renderInterventions();
  renderBoxes();
  renderActionsSummary();
}

function onHypotheses(_h) {
  state.agents.skeptic = "ok";
  renderAgents();
}

function onDone(payload) {
  state.busy = false;
  state.totals.frames += 1;
  state.totals.plants_seen += Object.keys(state.plants).length;
  $("run-state").textContent = "ready";
  if (payload && payload.field_state) {
    state.liveField = payload.field_state;
    state.fieldState = {
      wind_dir_deg: payload.field_state.wind_dir_deg,
      wind_speed_m_s: payload.field_state.wind_speed_m_s,
    };
  }
  for (const ag of ["vlm_reasoner", "skeptic"]) {
    if (state.agents[ag] !== "ok") state.agents[ag] = "skipped";
  }
  // Promote per-frame worst spray scenario to WHY NOT SPRAY.
  if (state._frameWorst) {
    const fw = state._frameWorst;
    const plant = state.plants[fw.plant_id] || {};
    state.whyNotSpray = {
      plant_id: fw.plant_id,
      action: "targeted_spray",
      condition: plant.top1 || "weed",
      phys_hazard: fw.phys_hazard ?? 0,
      eco_cost: fw.eco_cost ?? 0,
      drift_max_ppm: fw.drift_max_ppm ?? 0,
      predator_drop_pct: fw.predator_drop_pct ?? 0,
      snapshot: state.lastSnapshot,
      wind: state.liveField,
    };
    state._frameWorst = null;
  }
  // Sprint 2-4 data from done payload
  if (payload && payload.growth_stages) {
    for (const [pid, probs] of Object.entries(payload.growth_stages)) {
      if (!state.growthStages[+pid]) state.growthStages[+pid] = probs;
    }
  }
  if (payload && payload.anomaly_scores) {
    for (const [pid, score] of Object.entries(payload.anomaly_scores)) {
      state.anomalyScores[+pid] = +score;
    }
  }
  if (payload && payload.inference_mode) {
    state.inferenceMode = payload.inference_mode;
  }
  renderAgents();
  renderTotals();
  renderWhyNotSpray();
  renderWind();
  drawDriftCones();
  renderBoxes();
  renderGrowthAnomaly();
}

// --- Sprint 1-4 event handlers -------------------------------------------

function onVisualExplanation(payload) {
  state.explainOverlay = payload.data_url || null;
  const el = $("explain-overlay");
  if (el && state.explainOverlay) {
    el.src = state.explainOverlay;
    if (state.explainVisible) el.style.display = "block";
  }
}

function onTemporalDiff(payload) {
  state.temporalChanges = payload.per_plant_changes || {};
  renderBoxes();
}

function onActiveLearning(payload) {
  state.activeLearning = payload || {};
  const el = $("kpi-al-queue");
  if (el) el.textContent = state.activeLearning.unlabeled || state.activeLearning.total_queued || 0;
}

function onDebateTurn(payload) {
  state.debateTurns = payload.turn || 0;
  state.debateConverged = !payload.continuing;
  renderDebateIndicator();
}

function onRagContext(payload) {
  state.ragContext = payload.documents || [];
  renderRagPanel();
}

function onLLMPhase(payload) {
  if (payload.vlm_mode) state.inferenceMode.vlm = payload.vlm_mode;
  if (payload.skeptic_mode) state.inferenceMode.skeptic = payload.skeptic_mode;
  renderAgents();
}

// --- Sprint 1-4 render functions -----------------------------------------

function renderWeather() {
  const panel = $("weather-panel");
  const content = $("weather-content");
  if (!panel || !content || !state.weather) return;
  panel.style.display = "";
  const adj = state.weather.adjustments || {};
  const source = state.weather.weather_source || "none";
  let html = `<div class="weather-row"><span class="weather-label">source</span><span class="weather-value">${source}</span></div>`;
  for (const [label, val] of Object.entries(adj)) {
    if (Math.abs(val) < 0.01) continue;
    const dir = val > 0 ? "↑" : "↓";
    const color = val > 0 ? "var(--destructive)" : "var(--pesto-ok)";
    html += `<div class="weather-row"><span class="weather-label">${label.replace(/_/g, " ")}</span><span class="weather-value" style="color:${color}">${dir} ${Math.abs(val).toFixed(1)}</span></div>`;
  }
  if (Object.keys(adj).length === 0) {
    html += `<div class="weather-row"><span class="weather-label">status</span><span class="weather-value">neutral</span></div>`;
  }
  content.innerHTML = html;
}

function renderGrowthAnomaly() {
  const panel = $("growth-anomaly-panel");
  const content = $("growth-anomaly-content");
  if (!panel || !content) return;
  const hasGrowth = Object.keys(state.growthStages).length > 0;
  const hasAnomaly = Object.keys(state.anomalyScores).length > 0;
  if (!hasGrowth && !hasAnomaly) return;
  panel.style.display = "";
  let html = "";
  if (hasGrowth) {
    const counts = {};
    for (const info of Object.values(state.growthStages)) {
      const stage = info.growth_stage || "unknown";
      counts[stage] = (counts[stage] || 0) + 1;
    }
    html += '<div class="flex flex-wrap gap-1.5 mb-2">';
    for (const [stage, n] of Object.entries(counts)) {
      html += `<span class="growth-badge">${stage} ×${n}</span>`;
    }
    html += "</div>";
  }
  if (hasAnomaly) {
    const anomCount = Object.values(state.anomalyScores).filter(s => s > state.anomalyThreshold).length;
    if (anomCount > 0) {
      html += `<div style="color:var(--destructive)"><span class="anomaly-badge">ANOMALY</span> ${anomCount} plant${anomCount > 1 ? "s" : ""} flagged</div>`;
    } else {
      html += `<div style="color:var(--pesto-ok)">no anomalies detected</div>`;
    }
  }
  content.innerHTML = html;
}

function renderDebateIndicator() {
  const el = $("debate-indicator");
  if (!el) return;
  if (state.debateTurns === 0) { el.style.display = "none"; return; }
  el.style.display = "";
  let dots = "";
  for (let i = 1; i <= 3; i++) {
    const cls = i <= state.debateTurns ? (state.debateConverged && i === state.debateTurns ? "done" : "active") : "";
    dots += `<span class="turn-dot ${cls}"></span>`;
  }
  el.innerHTML = `<span>debate</span><span class="turn-dots">${dots}</span><span>${state.debateTurns}/3${state.debateConverged ? " converged" : ""}</span>`;
}

function renderRagPanel() {
  const panel = $("rag-panel");
  const content = $("rag-content");
  if (!panel || !content || !state.ragContext.length) return;
  panel.style.display = "";
  let html = "";
  for (const doc of state.ragContext) {
    html += `<div class="rag-doc">`;
    html += `<div class="rag-doc-title">${doc.id.replace(/_/g, " ")} <span class="rag-doc-score">${(doc.score * 100).toFixed(0)}%</span></div>`;
    html += `<div class="rag-doc-text">${doc.text}</div>`;
    if (doc.tags && doc.tags.length) {
      html += `<div class="rag-doc-tags">${doc.tags.join(" · ")}</div>`;
    }
    html += `</div>`;
  }
  content.innerHTML = html;
}

// Explain overlay toggle
(function() {
  const btn = $("explain-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    state.explainVisible = !state.explainVisible;
    btn.classList.toggle("active", state.explainVisible);
    const el = $("explain-overlay");
    if (el) el.style.display = state.explainVisible && state.explainOverlay ? "block" : "none";
  });
})();

// --- Frame capture + send ------------------------------------------------

function snapshotCurrentFrame() {
  const v = $("field-video");
  if (!v.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = 200; c.height = Math.round(200 * v.videoHeight / v.videoWidth);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.65);
}

async function captureAndRun(deep = false) {
  if (state.busy) return;
  const v = $("field-video");
  if (!v.videoWidth) return;
  state.busy = true;
  $("run-state").textContent = deep ? "deep dive…" : "analysing…";

  // Prefer the streaming endpoint when we have a known frame index.
  const idx = currentFrameIndex();
  if (idx !== null && state.manifest) {
    state.lastSnapshot = snapshotCurrentFrame();
    state.lastFrameTime = v.currentTime;
    try {
      const r = await fetch(
        `/api/run-stream-frame?frame_index=${idx}&deep=${deep ? "true" : "false"}`,
        { method: "POST" });
      if (!r.ok) {
        const err = await r.text();
        msg("run_frame_error", { error: `${r.status}: ${err}` });
        state.busy = false;
        $("run-state").textContent = "error";
      }
    } catch (e) {
      state.busy = false;
      $("run-state").textContent = "error";
      msg("run_frame_error", { error: String(e) });
    }
    return;
  }
  // Fallback: capture pixels and send to the generic /api/run-frame endpoint.
  const cw = Math.min(v.videoWidth, 1280);
  const ch = Math.round(cw * v.videoHeight / v.videoWidth);
  const c = document.createElement("canvas");
  c.width = cw; c.height = ch;
  c.getContext("2d").drawImage(v, 0, 0, cw, ch);
  state.lastSnapshot = snapshotCurrentFrame();
  state.lastFrameTime = v.currentTime;
  const blob = await new Promise((res) =>
    c.toBlob((b) => res(b), "image/jpeg", 0.85));
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  try {
    const r = await fetch(`/api/run-frame?deep=${deep ? "true" : "false"}`, {
      method: "POST", body: form,
    });
    if (!r.ok) {
      const err = await r.text();
      msg("run_frame_error", { error: `${r.status}: ${err}` });
      state.busy = false;
      $("run-state").textContent = "error";
    }
  } catch (e) {
    state.busy = false;
    $("run-state").textContent = "error";
    msg("run_frame_error", { error: String(e) });
  }
}

// Stream loop: try to capture on the cadence, but never overlap inferences.
let streamTimer = null;
function startStream() {
  state.stream.running = true;
  setStreamButtonState("running");
  scheduleNextCapture(0);
}

function stopStream() {
  state.stream.running = false;
  $("stream-btn").classList.remove("streaming");
  setStreamButtonState("ready");
  if (streamTimer) clearTimeout(streamTimer);
  streamTimer = null;
}

function scheduleNextCapture(extraDelay = 0) {
  if (!state.stream.running) return;
  if (streamTimer) clearTimeout(streamTimer);
  streamTimer = setTimeout(async () => {
    if (state.stream.running && !state.busy) {
      // Every N-th frame, do a deep-dive (fires Skeptic + VLMReasoner).
      state.stream.frameTick = (state.stream.frameTick + 1) % state.stream.deepEvery;
      const deep = state.stream.frameTick === 0;
      await captureAndRun(deep);
    }
    scheduleNextCapture(0);
  }, state.stream.intervalMs + extraDelay);
}

// --- Rendering -----------------------------------------------------------

function renderAgents() {
  const wrap = $("agent-rows");
  const allAgents = ["weed_detector", "disease_classifier", "segmentation",
                     "health_classifier", "weather_prior", "anomaly_detector",
                     "growth_stage", "vlm_reasoner",
                     "water_balance", "pesticide_fate", "ecological_dynamics",
                     "skeptic"];
  wrap.innerHTML = "";
  const STATUS_GLYPH = {
    ok:      '<i class="ph-bold ph-check"></i>',
    fail:    '<i class="ph-bold ph-x"></i>',
    running: '<i class="ph-fill ph-circle"></i>',
    skipped: '<i class="ph ph-minus"></i>',
    pending: '<i class="ph ph-hourglass"></i>',
  };
  for (const name of allAgents) {
    const status = state.agents[name] || "pending";
    const ref = MODEL_REFS[name];
    const refHtml = ref
      ? `<span class="model-ref"><a href="${ref.url}" target="_blank" rel="noopener" title="${ref.label}">${ref.label}</a></span>`
      : `<span class="model-ref"></span>`;
    const div = document.createElement("div");
    div.className = "agent-row";
    // Mode badge for VLM/Skeptic (Sprint 3)
    let modeBadge = "";
    if (name === "vlm_reasoner" && state.inferenceMode.vlm) {
      const mode = state.inferenceMode.vlm;
      modeBadge = `<span class="mode-badge ${mode}">${mode}</span>`;
    } else if (name === "skeptic" && state.inferenceMode.skeptic) {
      const mode = state.inferenceMode.skeptic;
      modeBadge = `<span class="mode-badge ${mode}">${mode}</span>`;
    }
    div.innerHTML = `
      <span class="name">${name}${modeBadge}</span>
      <span class="paradigm">${PARADIGMS[name] || ""}</span>
      ${refHtml}
      <span class="status ${status}">${STATUS_GLYPH[status] || "?"}</span>
    `;
    wrap.appendChild(div);
  }
}

function renderInterventions() {
  const wrap = $("interv-rows");
  wrap.innerHTML = "";
  // Use cumulative totals so the bar chart fills up over the run, not per frame.
  const totals = state.totals.intervention_counts;
  const grandTotal = Object.values(totals).reduce((a, b) => a + b, 0) || 1;
  for (const action of INTERVENTIONS) {
    const count = totals[action] || 0;
    const pct = (count / grandTotal) * 100;
    const f = ACTION_FRIENDLY[action];
    const label = f ? `${f.icon} ${f.verb}` : action;
    const div = document.createElement("div");
    div.className = "interv-row";
    div.innerHTML = `
      <span class="label">${label}</span>
      <span class="bar"><span class="bar-fill" style="width:${pct}%"></span></span>
      <span class="count">${count}</span>
    `;
    wrap.appendChild(div);
  }
}

function renderBoxes() {
  const layer = $("bbox-layer");
  const v = $("field-video");
  layer.innerHTML = "";
  if (!v.videoWidth) return;
  const sx = v.clientWidth / state.imageNatural.w;
  const sy = v.clientHeight / state.imageNatural.h;
  for (const pid in state.plants) {
    const p = state.plants[pid];
    const [x1, y1, x2, y2] = p.bbox;
    const color = CONDITION_COLOR[p.top1] || CONDITION_COLOR.ambiguous;
    const div = document.createElement("div");
    div.className = "bbox-overlay";
    if (p.physics_hazard > 0.4 && p.action === "laser_zap") {
      div.classList.add("physics-veto");
    }
    if (p.entropy > 1.5) div.classList.add("disputed");
    // Temporal change highlight (Sprint 4)
    const tc = state.temporalChanges[pid];
    if (tc && tc.changed) div.classList.add("temporal-changed");
    div.style.left = (x1 * sx) + "px";
    div.style.top = (y1 * sy) + "px";
    div.style.width = ((x2 - x1) * sx) + "px";
    div.style.height = ((y2 - y1) * sy) + "px";
    div.style.color = color;

    const label = document.createElement("div");
    label.className = "bbox-label";
    label.style.left = (x1 * sx) + "px";
    label.style.top = (y1 * sy) + "px";
    // If the dominant class is "ambiguous", roll up to the next-best
    // non-ambiguous class so the farmer never sees "UNCERTAIN".
    let primary = p.top1;
    let primaryP = p.top1_p ?? 0;
    let secondary = p.top2;
    let secondaryP = p.top2_p ?? 0;
    if (primary === "ambiguous" && secondary && secondary !== "ambiguous") {
      primary = secondary; primaryP = secondaryP;
      secondary = null; secondaryP = 0;
    }
    const labelColor = CONDITION_COLOR[primary] || color;
    label.style.color = labelColor;
    div.style.color = labelColor;
    const cond = CONDITION_FRIENDLY[primary] || primary;
    const probPct = Math.round(primaryP * 100);
    const top2 = secondary && secondary !== "ambiguous" && secondaryP > 0.12
      ? ` <span class="alt">/ ${Math.round(secondaryP*100)}% ${(CONDITION_FRIENDLY[secondary] || secondary)}</span>`
      : "";
    const act = ACTION_FRIENDLY[p.action];
    const actStr = act ? ` <span class="action-pill ${act.cls}">${act.icon} ${act.verb.toUpperCase()}</span>` : "";
    const species = state.species[pid];  // "weed" | "crop" | undefined
    const speciesPrefix = species
      ? `<span class="species ${species}">${species.toUpperCase()}</span> · `
      : "";
    // Anomaly badge (Sprint 2)
    const anomScore = state.anomalyScores[pid];
    const anomBadge = (anomScore != null && anomScore > state.anomalyThreshold)
      ? ` <span class="anomaly-badge">ANOMALY</span>` : "";
    // Growth stage badge (Sprint 2)
    const gs = state.growthStages[pid];
    const gsBadge = gs ? ` <span class="growth-badge">${gs.growth_stage || ""}</span>` : "";
    label.innerHTML = `${speciesPrefix}<b>${cond}</b> ${probPct}%${top2}${anomBadge}${gsBadge}${actStr}`;
    layer.appendChild(div);
    layer.appendChild(label);
  }
}

function renderHeatmap() {
  const c = $("heatmap-canvas");
  const wrap = $("canvas-wrap");
  c.width = wrap.clientWidth;
  c.height = wrap.clientHeight;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!state.imageNatural.w) return;
  const sx = c.width / state.imageNatural.w;
  const sy = c.height / state.imageNatural.h;

  // Pass 1: large soft halos with additive blending — saturates into the
  // green scene and reads from across the room.
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const pid in state.plants) {
    const p = state.plants[pid];
    const [x1, y1, x2, y2] = p.bbox;
    const cx = ((x1 + x2) / 2) * sx;
    const cy = ((y1 + y2) / 2) * sy;
    const halfDiag = Math.hypot((x2 - x1) * sx, (y2 - y1) * sy) / 2;
    const radius = Math.max(60, halfDiag * 2.0);

    let primary = p.top1, primaryP = p.top1_p || 0;
    if (primary === "ambiguous" && p.top2 && p.top2 !== "ambiguous") {
      primary = p.top2; primaryP = p.top2_p || 0;
    }
    // If we have a GT species label and the controller doesn't disagree,
    // use the species color so the visual is unambiguous.
    if (state.species[pid] === "weed") primary = "weed";
    const color = CONDITION_COLOR[primary] || CONDITION_COLOR.ambiguous;
    const intensity = clamp01(0.5 + primaryP * 0.5);

    // Three concentric stops for a punchy halo.
    let grad = ctx.createRadialGradient(cx, cy, radius * 0.02, cx, cy, radius);
    grad.addColorStop(0,   hexAlpha(color, 0.95 * intensity));
    grad.addColorStop(0.18, hexAlpha(color, 0.65 * intensity));
    grad.addColorStop(0.55, hexAlpha(color, 0.20 * intensity));
    grad.addColorStop(1,   hexAlpha(color, 0));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // Pass 2: a crisp inner stroke ring on each bbox so the user can SEE the
  // actual detection boundary, not just an amorphous glow.
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  for (const pid in state.plants) {
    const p = state.plants[pid];
    const [x1, y1, x2, y2] = p.bbox;
    let primary = p.top1;
    if (primary === "ambiguous" && p.top2) primary = p.top2;
    if (state.species[pid] === "weed") primary = "weed";
    const color = CONDITION_COLOR[primary] || CONDITION_COLOR.ambiguous;
    ctx.lineWidth = 2.0;
    ctx.strokeStyle = hexAlpha(color, 0.85);
    // Slightly inset the ring so the stroke sits on the plant, not floating.
    const insetX = 4 * sx;
    const insetY = 4 * sy;
    ctx.strokeRect(x1 * sx + insetX, y1 * sy + insetY,
                   (x2 - x1) * sx - 2 * insetX, (y2 - y1) * sy - 2 * insetY);
  }
  ctx.restore();
}

function drawDriftCones() {
  const c = $("drift-canvas");
  const wrap = $("canvas-wrap");
  c.width = wrap.clientWidth;
  c.height = wrap.clientHeight;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!state.imageNatural.w) return;
  const sx = c.width / state.imageNatural.w;
  const sy = c.height / state.imageNatural.h;
  const wind = state.fieldState.wind_dir_deg ?? 270;
  const flowRad = ((wind + 180) % 360) * Math.PI / 180;
  // Resolve --destructive once per render. Apply alpha via globalAlpha so we
  // don't depend on Canvas2D supporting color-mix() — works in any browser
  // that accepts the resolved color (rgb/oklch) as a fillStyle.
  const destructive = getComputedStyle(document.documentElement)
    .getPropertyValue("--destructive").trim() || "rgb(211, 95, 95)";
  ctx.save();
  for (const pid in state.plants) {
    const p = state.plants[pid];
    const phyTable = state.physics[pid] || {};
    const haz = Math.max(
      phyTable.targeted_spray?.hazard_score ?? 0,
      phyTable.targeted_fungicide?.hazard_score ?? 0,
      p.physics_hazard ?? 0,
    );
    if (haz < 0.3) continue;
    const [x1, y1, x2, y2] = p.bbox;
    const cx = ((x1 + x2) / 2) * sx;
    const cy = ((y1 + y2) / 2) * sy;
    const length = 80 + haz * 220;
    const halfWidth = 13 * Math.PI / 180;
    const tipX = cx + Math.cos(flowRad) * length;
    const tipY = cy + Math.sin(flowRad) * length;
    const ax = cx + Math.cos(flowRad - halfWidth) * length * 0.95;
    const ay = cy + Math.sin(flowRad - halfWidth) * length * 0.95;
    const bx = cx + Math.cos(flowRad + halfWidth) * length * 0.95;
    const by = cy + Math.sin(flowRad + halfWidth) * length * 0.95;
    const alpha = 0.18 + 0.45 * haz;
    ctx.fillStyle = destructive;
    ctx.strokeStyle = destructive;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ax, ay);
    ctx.quadraticCurveTo(tipX, tipY, bx, by);
    ctx.closePath();
    ctx.globalAlpha = alpha;
    ctx.fill();
    ctx.globalAlpha = Math.min(1, alpha + 0.3);
    ctx.stroke();
  }
  ctx.restore();
  // Wind readout — live values from the field state.
  const speed = state.fieldState.wind_speed_m_s;
  $("wind-label").textContent = (speed !== undefined && speed !== null)
    ? `${speed.toFixed(1)} m/s @ ${Math.round(wind)}°`
    : "— m/s @ —";
  $("wind-arrow").style.transform = `rotate(${(wind + 180) % 360}deg)`;
}

function renderBiophysics() {
  if (!state.biophysics) return;
  const b = state.biophysics;
  $("bio-stress").textContent = (b.stress_index ?? 0).toFixed(2);
  $("bio-demand").textContent = (b.demand_mm_per_day ?? 0).toFixed(2);
  $("bio-supply").textContent = (b.supply_mm_per_day ?? 0).toFixed(2);
  $("bio-psi").textContent = (b.soil_psi_kPa ?? 0).toFixed(0);
}

function renderActionsSummary() {
  const wrap = $("actions-summary");
  // Across frames every plant gets re-flagged, and most cards say the same
  // thing ("Laser zap, drift veto, ~1.5 ppm"). Group by (action, veto-reason)
  // so the user sees one card per kind of decision with the affected plants
  // listed inside, instead of a wall of near-duplicates.
  const vetoReasonOf = (f) => {
    if (f.phys_hazard > 0.4) return "drift";
    if (f.eco_cost > 0.5) return "predator";
    if (f.action === "laser_zap") return "mechanical";
    if (f.action === "no_action") return "healthy";
    if (f.action === "human_review" || f.action === "rescan_higher_res") return "review";
    return "ok";
  };
  const groups = new Map();  // key: action|reason
  for (let i = state.flaggedActions.length - 1; i >= 0; i--) {
    const f = state.flaggedActions[i];
    const key = `${f.action}|${vetoReasonOf(f)}`;
    if (!groups.has(key)) {
      groups.set(key, {
        action: f.action,
        reason: vetoReasonOf(f),
        latest: f,
        plant_ids: new Set(),
        drift_min: Infinity, drift_max: -Infinity,
        pred_min: Infinity, pred_max: -Infinity,
        count: 0,
        condition: f.condition,
      });
    }
    const g = groups.get(key);
    g.plant_ids.add(f.plant_id);
    g.count++;
    g.drift_min = Math.min(g.drift_min, f.drift_max_ppm ?? 0);
    g.drift_max = Math.max(g.drift_max, f.drift_max_ppm ?? 0);
    g.pred_min  = Math.min(g.pred_min,  f.predator_drop_pct ?? 0);
    g.pred_max  = Math.max(g.pred_max,  f.predator_drop_pct ?? 0);
  }
  const flagged = Array.from(groups.values());
  const lastFrame = state.lastFrameActions || [];
  if (!flagged.length && !lastFrame.length) {
    wrap.innerHTML = `<div class="actions-empty">Press <i class="ph-fill ph-play"></i> START STREAM to begin.</div>`;
    return;
  }
  const cards = [];
  // Render one card per (action, reason) group — each lists the affected
  // plants and the value range, so 30 zaps across 9 plants becomes one card.
  for (let idx = 0; idx < flagged.length; idx++) {
    const g = flagged[idx];
    const isTop = idx === 0;
    const af = ACTION_FRIENDLY[g.action];
    if (!af) continue;
    const veto = g.action === "laser_zap" && (g.reason === "drift" || g.reason === "predator");
    let why = "";
    if (g.reason === "drift") {
      const lo = g.drift_min.toFixed(2), hi = g.drift_max.toFixed(2);
      const range = lo === hi ? `${lo} ppm` : `${lo}–${hi} ppm`;
      why = `<span class="warn">drift veto</span>: ${range} would land on neighbours`;
    } else if (g.reason === "predator") {
      const lo = (g.pred_min*100).toFixed(0), hi = (g.pred_max*100).toFixed(0);
      const range = lo === hi ? `${lo}%` : `${lo}–${hi}%`;
      why = `<span class="warn">predator-loss veto</span>: chlorpyrifos would cut predators ${range} in 14 days`;
    } else if (g.action === "laser_zap") {
      why = "mechanical — no chemical, no predator loss";
    } else if (g.action === "targeted_fungicide") {
      why = "disease confirmed — spot-apply fungicide";
    } else if (g.action === "targeted_irrigation") {
      why = `soil ψ ${state.biophysics?.soil_psi_kPa?.toFixed?.(0) ?? "—"} kPa below field capacity`;
    } else if (g.action === "foliar_nutrient") {
      why = "nutrient stress — corrective foliar feed";
    } else if (g.action === "targeted_spray") {
      why = "weed confirmed; no downwind targets at risk";
    } else if (g.action === "no_action") {
      why = "healthy — no intervention needed";
    } else if (g.action === "human_review") {
      why = "uncertain — flagged for expert review";
    } else if (g.action === "rescan_higher_res") {
      why = "insufficient detail — request closer scan";
    }
    const plantList = Array.from(g.plant_ids).sort((a,b) => a-b)
      .map(id => `#${id}`).join(", ");
    const detail = isTop ? whyNotSprayDetailFor(g.latest) : "";
    const snap = g.latest.snapshot
      ? `<img class="snap" src="${g.latest.snapshot}" data-time="${g.latest.videoTime || 0}" alt="">`
      : "";
    const previewBtn = g.latest.snapshot
      ? `<button class="preview-btn" data-src="${g.latest.snapshot}" title="Preview"><i class="ph ph-magnifying-glass-plus"></i></button>`
      : "";
    const stageBadge = g.latest.growthStage
      ? ` <span class="growth-badge">${g.latest.growthStage}</span>` : "";
    cards.push(`
      <div class="action-card ${af.cls}${veto ? ' veto' : ''}" data-time="${g.latest.videoTime || 0}" data-action="${g.action}">
        ${snap}
        <span class="icon">${af.icon}</span>
        <div class="copy">
          <div class="head">${capitalise(af.verb)} <span class="text-dim">on ${g.plant_ids.size} plant${g.plant_ids.size===1?"":"s"}</span>${stageBadge}</div>
          <div class="why">${why}</div>
          <div class="why text-dim" style="font-size:10.5px">plants: ${plantList}</div>
          ${detail}
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
          <div class="count">${g.count}×</div>
          ${previewBtn}
        </div>
      </div>
    `);
  }
  // Filter pills — count all accumulated actions, clicking filters the cards below.
  const filterCounts = {};
  for (const f of state.flaggedActions) {
    filterCounts[f.action] = (filterCounts[f.action] || 0) + 1;
  }
  const activeFilter = state._actionFilter || "all";
  const filterOrder = ["laser_zap", "targeted_spray", "targeted_fungicide",
                       "targeted_irrigation", "foliar_nutrient",
                       "human_review", "rescan_higher_res", "no_action"];
  const totalCount = state.flaggedActions.length;
  let filterPills = `<button class="mini-pill filter-pill ${activeFilter === 'all' ? 'active' : ''}" data-filter="all">all <b>${totalCount}</b></button>`;
  for (const a of filterOrder) {
    if (!filterCounts[a]) continue;
    const af = ACTION_FRIENDLY[a]; if (!af) continue;
    filterPills += `<button class="mini-pill filter-pill ${af.cls} ${activeFilter === a ? 'active' : ''}" data-filter="${a}">${af.icon} ${af.verb} <b>${filterCounts[a]}</b></button>`;
  }
  const filterStrip = totalCount > 0 ? `
    <div class="last-frame-strip">
      <div class="mini-pills">${filterPills}</div>
    </div>
  ` : "";
  // Filter cards by active filter
  const filteredCards = activeFilter === "all"
    ? cards
    : cards.filter((_, idx) => {
        const g = Array.from(groups.values())[idx];
        return g && g.action === activeFilter;
      });
  const flaggedHeader = filteredCards.length
    ? `<div class="actions-section-title flagged"><i class="ph-fill ph-flag"></i> Interventions (${filteredCards.length})</div>`
    : "";
  wrap.innerHTML = filterStrip + flaggedHeader + filteredCards.join("");
  // Wire snapshot clicks to seek video.
  wrap.querySelectorAll(".action-card[data-time]").forEach(el => {
    el.addEventListener("click", (e) => {
      if (e.target.closest(".preview-btn")) return; // handled separately
      const t = parseFloat(el.dataset.time || "0");
      const v = $("field-video");
      if (!isFinite(t)) return;
      v.currentTime = t;
      v.play();
    });
  });
  // Wire preview buttons to open modal
  wrap.querySelectorAll(".preview-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const src = btn.dataset.src;
      if (src) showPreviewModal(src);
    });
  });
  // Wire filter pills
  wrap.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", (e) => {
      e.stopPropagation();
      state._actionFilter = pill.dataset.filter;
      renderActionsSummary();
    });
  });
}

function renderTotals() {
  const t = state.totals;
  const ic = t.intervention_counts;

  // Count all actions to derive stats
  const healthyCount = ic.no_action || 0;
  const actionsTaken = (ic.laser_zap || 0) + (ic.targeted_spray || 0)
    + (ic.targeted_fungicide || 0) + (ic.targeted_irrigation || 0)
    + (ic.foliar_nutrient || 0);
  const reviewCount = (ic.human_review || 0) + (ic.rescan_higher_res || 0);
  const totalPlants = healthyCount + actionsTaken + reviewCount;

  // Field Health — % healthy out of all plants seen
  const healthPct = totalPlants > 0 ? Math.round((healthyCount / totalPlants) * 100) : 0;
  $("kpi-health").innerHTML = `${healthPct}<span class="unit text-sm font-normal text-muted-foreground">%</span>`;

  // Threats Found — everything that needed intervention or review
  $("kpi-threats").textContent = actionsTaken + reviewCount;

  // Actions Taken
  $("kpi-actions").textContent = actionsTaken;

  // Chemical Saved
  $("kpi-saved").innerHTML = `${t.chem_saved_ml.toFixed(0)} <span class="unit text-sm font-normal text-muted-foreground">ml</span>`;

  // Unsafe Sprays Blocked — drift + ecology vetoes combined
  $("kpi-blocked").textContent = t.phys_vetos + t.eco_vetos;

  // Needs Attention — human_review + rescan actions (farmer should go check)
  const attention = (ic.human_review || 0) + (ic.rescan_higher_res || 0);
  $("kpi-attention").textContent = attention;
}

// WHY-NOT-SPRAY now renders inline inside the matching flagged action card —
// see whyNotSprayDetailFor(). The standalone panel is gone, so this re-renders
// the actions summary so the inline detail picks up the latest state.
function renderWhyNotSpray() { renderActionsSummary(); }

// Build the inline "why not spray" markup. Caller is responsible for
// attaching this to a single card (the top of the flagged list) so the
// explanation doesn't repeat under every plant.
function whyNotSprayDetailFor(f) {
  const w = state.whyNotSpray;
  if (!w) return "";
  const wind = w.wind || {};
  const drift = (w.drift_max_ppm || 0).toFixed(3);
  const predDrop = ((w.predator_drop_pct || 0) * 100).toFixed(0);
  const psi = state.biophysics?.soil_psi_kPa?.toFixed?.(0) ?? "—";
  return `
    <div class="why-detail">
      <b>Why not spray here:</b>
      Spraying chlorpyrifos would deposit <span class="num">${drift} ppm</span>
      on a downwind neighbour and cut ladybug predators by
      <span class="num">${predDrop}%</span> in 14 days.
      Wind ${wind.wind_speed_m_s ?? "—"} m/s @ ${Math.round(wind.wind_dir_deg ?? 0)}°,
      soil ψ ${psi} kPa. <i class="ph-bold ph-arrow-right"></i>
      <span class="alt">laser zap</span>: 0 ppm drift, 0 predator loss, ~5 ml herbicide saved.
    </div>
  `;
}

function drawEcologyChart(_traj) { /* removed — replaced by WHY NOT SPRAY */ }

// --- Plain-English message stream ----------------------------------------

function msg(kind, payload) {
  const stream = $("msg-stream");
  const out = humanize(kind, payload);
  if (!out || !out.line) return;
  const { line, status } = out;  // status: "ok" | "warn" | "flag"
  const row = document.createElement("div");
  row.className = "row " + status;
  if (kind.endsWith("_error")) row.classList.add("flag");
  const tag = status === "flag" ? "FAIL"
            : status === "warn" ? "WARN"
            : "GOOD";
  row.innerHTML = `<span class="ts">[t=${fmtT()}]</span><span class="tag">${tag}</span> ${line}`;
  stream.appendChild(row);
  while (stream.children.length > 200) stream.removeChild(stream.firstChild);
  stream.scrollTop = stream.scrollHeight;
}

function humanize(kind, p) {
  // Always returns { line, status }. status drives the row tag (GOOD/WARN/FAIL).
  // Every event kind we know about emits something — silence used to mean
  // "everything's fine", which read as "the system stalled" instead.
  const ok = (line)   => ({ line, status: "ok" });
  const warn = (line) => ({ line, status: "warn" });
  const flag = (line) => ({ line, status: "flag" });
  switch (kind) {
    case "run_started":
      return ok(`<b>frame:</b> ${p.kind} ${p.filename ? `<span class="text-dim">${p.filename}</span>` : ""}`);
    case "latent_initialised": {
      const n = (p?.plants ?? []).length;
      return ok(`detected <b>${n}</b> plant region${n===1?"":"s"}`);
    }
    case "constraint": {
      const sender = p.sender;
      const n = Object.keys(p.per_plant_log_likelihoods || {}).length;
      let topAxis = null, topMag = 0;
      for (const pid in p.per_plant_log_likelihoods) {
        const ll = p.per_plant_log_likelihoods[pid];
        const labels = ["healthy_crop","weed","disease","nutrient_stress","water_stress","pest_damage","ambiguous"];
        for (let i = 1; i < labels.length - 1; i++) {
          if (ll[i] > topMag) { topMag = ll[i]; topAxis = labels[i]; }
        }
      }
      const hint = topAxis
        ? ` — flags <span class="warn">${CONDITION_FRIENDLY[topAxis]}</span>`
        : "";
      const line = `<b>${sender.replace(/_/g, " ")}</b> contributed evidence on ${n} plant${n===1?"":"s"}${hint}`;
      return topAxis ? warn(line) : ok(line);
    }
    case "physics_assessment": {
      // Physics scores every candidate action. Non-chemical actions are
      // trivially zero-drift and would flood the stream — skip them.
      const isChem = p.action_type === "targeted_spray" || p.action_type === "targeted_fungicide";
      if (!isChem) return null;
      const haz = p.hazard_score ?? 0;
      const drift = (p.hazard_breakdown?.drift_max_ppm ?? 0).toFixed(2);
      if (haz >= 0.4) {
        return warn(`<i class="ph-fill ph-warning"></i> physics flag: spraying plant ${p.plant_id} would hit downwind targets at ${drift} ppm`);
      }
      return ok(`pesticide_fate cleared plant ${p.plant_id} for <b>${p.action_type}</b> (drift ${drift} ppm)`);
    }
    case "ecology_trajectory": {
      const isChem = p.action_type === "targeted_spray" || p.action_type === "targeted_fungicide";
      if (!isChem) return null;
      const cost = p.ecological_cost_score ?? 0;
      const drop = (p.cost_breakdown?.predator_drop_pct_d14 ?? 0) * 100;
      if (cost >= 0.5) {
        return warn(`<i class="ph-fill ph-warning"></i> ecology flag: chlorpyrifos on plant ${p.plant_id} would drop predators ${drop.toFixed(0)}% in 14 days`);
      }
      return ok(`ecological_dynamics cleared plant ${p.plant_id} for <b>${p.action_type}</b> (predator drop ${drop.toFixed(0)}%)`);
    }
    case "action": {
      const cond = ((state.plants[p.plant_id]?.top1) ?? "ambiguous");
      const condFr = CONDITION_FRIENDLY[cond] || cond;
      const act = ACTION_FRIENDLY[p.action_type] || { verb: p.action_type, icon: "•" };
      return ok(`${act.icon} plant ${p.plant_id} <span class="text-dim">(${condFr.toLowerCase()})</span> <i class="ph-bold ph-arrow-right"></i> <b>${act.verb}</b>`);
    }
    case "hypotheses": {
      if (!Array.isArray(p) || !p.length) {
        return ok(`<i class="ph ph-brain"></i> skeptic raised no objections`);
      }
      const list = p.map(h => h.hypothesis_id).join(", ");
      return warn(`<i class="ph ph-brain"></i> skeptic flagged: ${list}`);
    }
    case "done": {
      const n = (p?.actions ?? []).length;
      return ok(`<b>frame complete</b> — ${n} action${n===1?"":"s"} recommended`);
    }
    case "llm_phase":
      return ok(`llm phase: <b>${p?.phase ?? "—"}</b>`);
    case "cross_exam":
      return ok(`cross-exam: <b>${p?.summary ?? "round complete"}</b>`);
    case "ml_agent_error":
      return flag(`${p.agent} failed: ${p.error}`);
    case "skeptic_error":
    case "vlm_error":
    case "skeptic_construction_error":
    case "vlm_construction_error":
    case "run_frame_error":
    case "run_demo_error":
    case "upload_error":
      return flag(`${kind}: ${p.error || ""}`);
    // Sprint 1-4 events
    case "visual_explanation":
      return ok(`evidence overlay updated`);
    case "temporal_diff": {
      const n = (p.escalated || []).length;
      return n > 0 ? warn(`<b>${n}</b> plant${n===1?"":"s"} changed since last frame`) : ok(`no temporal change detected`);
    }
    case "active_learning_update": {
      const q = p.total_queued || 0;
      return q > 0 ? warn(`<b>${q}</b> hard case${q===1?"":"s"} queued for labeling`) : ok(`active learning queue empty`);
    }
    case "debate_turn":
      return p.continuing ? ok(`skeptic-VLM debate turn <b>${p.turn}/${p.max_turns}</b>`) : ok(`debate ${p.converged ? "converged" : "ended"} after ${p.turn} turn${p.turn===1?"":"s"}`);
    case "rag_context": {
      const n = (p.documents || []).length;
      return ok(`RAG retrieved <b>${n}</b> treatment reference${n===1?"":"s"}`);
    }
    case "llm_phase":
      return ok(`LLM phase: VLM=${p.vlm_mode||"?"} skeptic=${p.skeptic_mode||"?"} disputed=<b>${(p.disputed_plants||[]).length}</b>`);
    case "weather_prior_error":
    case "anomaly_detector_error":
    case "growth_stage_error":
      return flag(`${kind}: ${p.error || ""}`);
    default:
      return ok(`<span class="text-dim">${kind}</span>`);
  }
}

// --- Helpers --------------------------------------------------------------

function capitalise(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function showPreviewModal(src) {
  // Remove existing modal
  const old = document.getElementById("preview-modal");
  if (old) old.remove();
  const modal = document.createElement("div");
  modal.id = "preview-modal";
  modal.style.cssText = "position:fixed;inset:0;z-index:100;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.75);backdrop-filter:blur(6px);cursor:pointer";
  modal.innerHTML = `<img src="${src}" style="max-width:90vw;max-height:85vh;border-radius:8px;border:2px solid var(--border);box-shadow:0 20px 60px rgba(0,0,0,0.5)">`;
  modal.addEventListener("click", () => modal.remove());
  document.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { modal.remove(); document.removeEventListener("keydown", esc); }
  });
  document.body.appendChild(modal);
}
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function hexAlpha(hex, a) {
  const m = /^#([a-f0-9]{6})$/i.exec(hex);
  if (!m) return `rgba(255,255,255,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}

// --- Wiring --------------------------------------------------------------

$("stream-btn").addEventListener("click", () => {
  if (state.stream.running) stopStream(); else startStream();
});

$("run-btn").addEventListener("click", async () => {
  if (state.busy) { alert("Another inference is already in progress."); return; }
  const file = $("image-input").files[0];
  if (!file) { alert("Pick an image first."); return; }
  state.busy = true;
  $("run-state").textContent = "running…";
  const form = new FormData();
  form.append("image", file);
  try {
    const r = await fetch("/api/run", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
  } catch (e) {
    $("run-state").textContent = "error";
    msg("upload_error", { error: String(e) });
  } finally {
    state.busy = false;
  }
});

window.addEventListener("resize", () => {
  renderBoxes();
  renderHeatmap();
  drawDriftCones();
});

// The first heatmap/bbox render fires from the websocket — but if the video
// hasn't dispatched `loadedmetadata` yet the canvas is sized 0×0 and nothing
// shows. Re-render every time the video reflows.
$("field-video").addEventListener("loadedmetadata", () => {
  renderBoxes();
  renderHeatmap();
  drawDriftCones();
});

// Initial state
renderAgents();
renderInterventions();
renderActionsSummary();
renderTotals();
renderWhyNotSpray();
loadVideoSource();
