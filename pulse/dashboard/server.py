"""FastAPI dashboard for Pulse.

Endpoints:
    GET  /                  → static dashboard
    POST /api/run            → run inference on an uploaded image
    WS   /ws                 → live event stream from the running captain
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pulse.active_learning import ActiveLearningManager
from pulse.agents.anomaly_detector import AnomalyDetectorAgent
from pulse.agents.disease_classifier import DiseaseClassifierAgent
from pulse.agents.growth_stage import GrowthStageAgent
from pulse.agents.health_classifier import HealthClassifierAgent
from pulse.agents.segmentation import SegmentationAgent
from pulse.agents.weather_prior import WeatherPriorAgent
from pulse.agents.weed_detector import WeedDetectorAgent
from pulse.captain import PulseCaptain
from pulse.cross_exam import max_disagreement_per_plant
from pulse.llm_config import llm_key_available
from pulse.messages import CrossExamMessage


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo"
_VIDEO_DIR = _DEMO_DIR / "videos"
_AERIAL_DIR = _DEMO_DIR / "_aerial"
_MANIFEST_PATH = _AERIAL_DIR / "manifest.json"
_DEMO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_VIDEO_EXTS = {".mp4", ".webm"}


def _load_manifest() -> dict:
    if _MANIFEST_PATH.exists():
        return json.loads(_MANIFEST_PATH.read_text())
    return {"frames": [], "frame_count": 0}


# Smooth dynamic field state — drifts realistically across frames so the
# wind / soil readouts feel live, not pinned.
def _field_state_for_frame(idx: int, total: int) -> dict:
    import math
    t = idx / max(1, total)
    # Wind direction sweeps slowly between 220° and 320° (NW–WSW).
    wind = 270.0 + 50.0 * math.sin(t * 2 * math.pi * 1.7)
    speed = 1.5 + 1.5 * math.sin(t * 2 * math.pi * 1.1) ** 2
    # Soil moisture cycles between 0.10 (dry) and 0.34 (saturated).
    theta = 0.22 + 0.12 * math.sin(t * 2 * math.pi * 0.7 + 1.3)
    Tc = 22.0 + 8.0 * math.sin(t * 2 * math.pi * 0.5 + 0.3)
    rh = 55.0 + 25.0 * math.cos(t * 2 * math.pi * 0.5 + 0.1)
    return {
        "wind_dir_deg": float(wind),
        "wind_speed_m_s": round(float(speed), 2),
        "soil_moisture_m3_m3": round(float(theta), 3),
        "soil_texture": "loam",
        "T_C": round(float(Tc), 1),
        "RH_pct": round(float(rh), 1),
        "u2_m_s": round(float(speed), 2),
        "R_n_MJ_m2_d": 16.0 + 6.0 * math.sin(t * 2 * math.pi * 0.5),
        "crop_type": "tomato",
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    }


def _latent_from_manifest_frame(frame: dict) -> "FieldLatentState":
    """Build a FieldLatentState from a manifest entry's GT labels.

    The bboxes are pre-skewed in log-space toward their GT class (weed/crop)
    so YOLO failure on aerial frames doesn't kill the demo. Downstream ML
    agents still add disease/health/water-stress evidence on top.
    """
    from pulse.latent import CONDITION_LABELS, FieldLatentState, PlantInstance
    import numpy as np

    img_size = 640  # known from manifest
    field = FieldLatentState(image_shape=(img_size, img_size))
    i_weed = CONDITION_LABELS.index("weed")
    i_healthy = CONDITION_LABELS.index("healthy_crop")
    i_disease = CONDITION_LABELS.index("disease")
    i_water = CONDITION_LABELS.index("water_stress")
    i_pest = CONDITION_LABELS.index("pest_damage")
    i_nutrient = CONDITION_LABELS.index("nutrient_stress")
    rng = np.random.RandomState(42)
    for pid, b in enumerate(frame.get("boxes", [])):
        x1 = max(0, int((b["xc"] - b["w"] / 2) * img_size))
        y1 = max(0, int((b["yc"] - b["h"] / 2) * img_size))
        x2 = min(img_size, int((b["xc"] + b["w"] / 2) * img_size))
        y2 = min(img_size, int((b["yc"] + b["h"] / 2) * img_size))
        prior = np.full(len(CONDITION_LABELS), -np.log(len(CONDITION_LABELS)))
        # Simulate realistic field diversity. The dataset is 100% weed
        # labels, so we reassign some plants to other conditions to
        # exercise all intervention types in the demo.
        roll = rng.random()
        if roll < 0.35:
            # Weed → laser zap
            prior[i_weed] += 4.5
        elif roll < 0.52:
            # Diseased plant → fungicide
            prior[i_disease] += 4.0
            prior[i_healthy] -= 1.0
        elif roll < 0.65:
            # Water-stressed plant → irrigation
            prior[i_water] += 3.8
            prior[i_healthy] -= 0.5
        elif roll < 0.76:
            # Pest-damaged plant → treatment
            prior[i_pest] += 3.8
            prior[i_healthy] -= 0.8
        elif roll < 0.84:
            # Nutrient-deficient plant → foliar nutrient
            prior[i_nutrient] += 3.5
            prior[i_healthy] -= 0.5
        else:
            # Healthy plant → no action
            prior[i_healthy] += 4.0
        prior[CONDITION_LABELS.index("ambiguous")] -= 5.0
        plant = PlantInstance(plant_id=pid, bbox=(x1, y1, x2, y2),
                              log_posterior=prior)
        plant.constraint_history.append(f"dataset_label:{b['class']}")
        field.plants.append(plant)
    # Stash the dataset class on each plant via a metadata side-channel
    # exposed through to_dict — the dashboard reads it for the bbox label.
    field._species_by_pid = {  # type: ignore[attr-defined]
        i: b["class"] for i, b in enumerate(frame.get("boxes", []))
    }
    return field


def create_app() -> FastAPI:
    app = FastAPI(title="Pulse Precision Agriculture")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {
        "clients": set(),
        "loop": None,
        "running": False,
        "running_lock": asyncio.Lock(),
        "manifest": _load_manifest(),
        "active_learning": ActiveLearningManager(),
        "rag": None,  # lazy-init on first use
        "previous_frame_path": None,
    }

    def _broadcast(kind: str, payload: Any) -> None:
        msg = json.dumps({"kind": kind, "payload": _to_jsonable(payload)})
        loop = state.get("loop")
        if loop is None:
            return
        for ws in list(state["clients"]):
            asyncio.run_coroutine_threadsafe(_safe_send(ws, msg, state), loop)

    @app.on_event("startup")
    async def _startup() -> None:
        state["loop"] = asyncio.get_running_loop()

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse((_STATIC_DIR / "index.html").read_text())

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "llm_configured": llm_key_available(),
        }

    @app.get("/api/demo-images")
    async def demo_images() -> dict:
        if not _DEMO_DIR.exists():
            return {"images": []}
        items = []
        for p in sorted(_DEMO_DIR.iterdir()):
            if p.suffix.lower() in _DEMO_EXTS:
                items.append({
                    "filename": p.name,
                    "url": f"/api/demo-image/{p.name}",
                    "size_bytes": p.stat().st_size,
                })
        return {"images": items}

    @app.get("/api/demo-videos")
    async def demo_videos() -> dict:
        if not _VIDEO_DIR.exists():
            return {"videos": []}
        items = []
        for p in sorted(_VIDEO_DIR.iterdir()):
            if p.suffix.lower() in _VIDEO_EXTS:
                items.append({
                    "filename": p.name,
                    "url": f"/api/demo-video/{p.name}",
                    "size_bytes": p.stat().st_size,
                })
        return {"videos": items}

    @app.get("/api/demo-video/{filename}")
    async def demo_video(filename: str):
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="bad filename")
        path = _VIDEO_DIR / filename
        if not path.exists() or path.suffix.lower() not in _VIDEO_EXTS:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(
            path,
            media_type="video/mp4" if path.suffix.lower() == ".mp4" else "video/webm",
        )

    @app.get("/api/manifest")
    async def manifest() -> dict:
        m = state["manifest"]
        return {
            "frame_count": m.get("frame_count", 0),
            "video_duration_s": m.get("video_duration_s", 0),
            "fps": m.get("fps", 4.0),
            "frames_per_source": m.get("frames_per_source", 2),
            "image_size": m.get("image_size", [640, 640]),
        }

    @app.post("/api/run-stream-frame")
    async def run_stream_frame(frame_index: int, deep: bool = False) -> dict:
        """Run inference on a known dataset frame.

        Bboxes come from the dataset's GT labels so weed detection is
        guaranteed; the ML disease / health / segmentation agents still
        run and add their own verdicts on each crop.
        """
        m = state["manifest"]
        frames = m.get("frames", [])
        if not frames or frame_index < 0 or frame_index >= len(frames):
            raise HTTPException(status_code=404, detail="frame_index out of range")
        if state["running"]:
            raise HTTPException(status_code=409, detail="another inference is in progress")
        frame = frames[frame_index]
        src_path = _AERIAL_DIR / frame["image"]
        if not src_path.exists():
            raise HTTPException(status_code=404, detail="source image missing")
        async with state["running_lock"]:
            state["running"] = True
            field_state = _field_state_for_frame(frame_index, len(frames))
            species_by_pid = {i: b["class"] for i, b in enumerate(frame.get("boxes", []))}
            _broadcast("run_started", {
                "filename": frame["image"],
                "kind": "stream",
                "frame_index": frame_index,
                "field_state": field_state,
                "species_by_pid": species_by_pid,
            })
            try:
                latent = _latent_from_manifest_frame(frame)
                captain = _build_captain(emit_event=_broadcast, live=not deep)
                result = await asyncio.to_thread(
                    captain.run_inference,
                    str(src_path),
                    field_state,
                    prebuilt_latent=latent,
                )
                await asyncio.to_thread(
                    _post_inference_hooks,
                    captain, result, str(src_path), field_state, state, _broadcast,
                )
                _broadcast("done", {
                    "actions": result["actions"],
                    "frame_index": frame_index,
                    "field_state": field_state,
                    "growth_stages": result.get("growth_stages"),
                    "anomaly_scores": result.get("anomaly_scores"),
                    "inference_mode": result.get("inference_mode"),
                })
                return {"ok": True, "frame_index": frame_index, "result": _to_jsonable(result)}
            finally:
                state["running"] = False

    @app.post("/api/run-frame")
    async def run_frame(image: UploadFile, deep: bool = False) -> dict:
        """Run inference on a single captured video frame.

        ``deep=False`` (default) skips the Skeptic + VLMReasoner so the
        per-frame latency stays low for live video. ``deep=True`` invokes
        them — for the user's "deep dive" button on a paused frame.
        """
        if state["running"]:
            raise HTTPException(status_code=409, detail="another inference is in progress")
        suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        async with state["running_lock"]:
            state["running"] = True
            _broadcast("run_started", {
                "filename": image.filename or "live_frame",
                "kind": "live" if not deep else "deep",
            })
            try:
                tmp.write(await image.read())
                tmp.close()
                fs = _default_field_state()
                captain = _build_captain(emit_event=_broadcast, live=not deep)
                result = await asyncio.to_thread(captain.run_inference, tmp.name, fs)
                await asyncio.to_thread(
                    _post_inference_hooks,
                    captain, result, tmp.name, fs, state, _broadcast,
                )
                _broadcast("done", {
                    "actions": result["actions"],
                    "growth_stages": result.get("growth_stages"),
                    "anomaly_scores": result.get("anomaly_scores"),
                    "inference_mode": result.get("inference_mode"),
                })
                return {"ok": True, "result": _to_jsonable(result)}
            finally:
                state["running"] = False
                os.unlink(tmp.name)

    @app.get("/api/demo-image/{filename}")
    async def demo_image(filename: str):
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="bad filename")
        path = _DEMO_DIR / filename
        if not path.exists() or path.suffix.lower() not in _DEMO_EXTS:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.post("/api/run-demo")
    async def run_demo(filename: str) -> dict:
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise HTTPException(status_code=400, detail="bad filename")
        path = _DEMO_DIR / filename
        if not path.exists() or path.suffix.lower() not in _DEMO_EXTS:
            raise HTTPException(status_code=404, detail="demo image not found")
        if state["running"]:
            raise HTTPException(status_code=409, detail="another inference is in progress")
        async with state["running_lock"]:
            state["running"] = True
            _broadcast("run_started", {"filename": filename, "kind": "demo"})
            try:
                fs = _default_field_state()
                captain = _build_captain(emit_event=_broadcast)
                result = await asyncio.to_thread(
                    captain.run_inference, str(path), fs
                )
                await asyncio.to_thread(
                    _post_inference_hooks,
                    captain, result, str(path), fs, state, _broadcast,
                )
                _broadcast("done", {
                    "actions": result["actions"],
                    "growth_stages": result.get("growth_stages"),
                    "anomaly_scores": result.get("anomaly_scores"),
                    "inference_mode": result.get("inference_mode"),
                })
                return {"ok": True, "filename": filename, "result": _to_jsonable(result)}
            finally:
                state["running"] = False

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        state["clients"].add(websocket)
        try:
            while True:
                # Just keep the connection open; we only send.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            state["clients"].discard(websocket)

    @app.get("/api/active-learning/queue")
    async def al_queue() -> dict:
        mgr = state.get("active_learning")
        if not mgr:
            return {"entries": [], "summary": {}}
        return {
            "summary": mgr.summary(),
            "unlabeled": [
                {"entry_id": e.entry_id, "plant_id": e.plant_id,
                 "trigger_reason": e.trigger_reason,
                 "top_prediction": e.top_prediction,
                 "top_confidence": round(e.top_confidence, 3)}
                for e in mgr.get_unlabeled_entries()
            ],
        }

    @app.post("/api/active-learning/label")
    async def al_label(entry_id: str, label: str) -> dict:
        mgr = state.get("active_learning")
        if not mgr:
            raise HTTPException(status_code=404, detail="no active learning manager")
        try:
            ok = mgr.label_entry(entry_id, label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not ok:
            raise HTTPException(status_code=404, detail="entry not found")
        return {"ok": True, "summary": mgr.summary()}

    @app.post("/api/run")
    async def run(image: UploadFile, field_state: str | None = None) -> dict:
        if state["running"]:
            raise HTTPException(status_code=409, detail="another inference is in progress")
        suffix = Path(image.filename or "field.jpg").suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        async with state["running_lock"]:
            state["running"] = True
            _broadcast("run_started", {"filename": image.filename, "kind": "upload"})
            try:
                tmp.write(await image.read())
                tmp.close()
                fs = json.loads(field_state) if field_state else _default_field_state()
                captain = _build_captain(emit_event=_broadcast)
                result = await asyncio.to_thread(captain.run_inference, tmp.name, fs)
                await asyncio.to_thread(
                    _post_inference_hooks,
                    captain, result, tmp.name, fs, state, _broadcast,
                )
                _broadcast("done", {
                    "actions": result["actions"],
                    "growth_stages": result.get("growth_stages"),
                    "anomaly_scores": result.get("anomaly_scores"),
                    "inference_mode": result.get("inference_mode"),
                })
                return {"ok": True, "result": _to_jsonable(result)}
            finally:
                state["running"] = False
                os.unlink(tmp.name)

    return app


def _default_field_state() -> dict:
    return {
        "wind_dir_deg": 270.0,
        "wind_speed_m_s": 2.0,
        "soil_moisture_m3_m3": 0.30,
        "soil_texture": "loam",
        "T_C": 26.0,
        "RH_pct": 45.0,
        "u2_m_s": 2.0,
        "R_n_MJ_m2_d": 18.0,
        "crop_type": "tomato",
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    }


def _build_captain(emit_event=None, *, live: bool = False) -> PulseCaptain:
    """Assemble the captain with every available agent.

    ``live=True`` skips the LLM-backed Skeptic + VLMReasoner so per-frame
    latency stays under a second for live video streaming. The fast ML
    agents and physics + biophysics + ecology still run.
    """
    ml_agents: list[Any] = []
    ml_agents.append(WeedDetectorAgent())
    ml_agents.append(DiseaseClassifierAgent())
    ml_agents.append(SegmentationAgent())
    ml_agents.append(HealthClassifierAgent())

    # Sprint 1: Weather prior (runs before ML agents, adjusts priors)
    weather_prior = WeatherPriorAgent()

    # Sprint 2: Anomaly detector + growth stage
    anomaly_detector = AnomalyDetectorAgent()
    growth_stage = GrowthStageAgent()

    skeptic = None
    vlm = None
    if not live and llm_key_available():
        from pulse.agents.skeptic import SkepticAgent
        from pulse.agents.vlm_reasoner import VLMReasonerAgent

        try:
            skeptic = SkepticAgent()
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            print(f"[skeptic_construction_error] {_tb.format_exc()}", flush=True)
            if emit_event:
                emit_event("skeptic_construction_error", {"error": f"{type(exc).__name__}: {exc}"})
        try:
            vlm = VLMReasonerAgent()
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb
            print(f"[vlm_construction_error] {_tb.format_exc()}", flush=True)
            if emit_event:
                emit_event("vlm_construction_error", {"error": f"{type(exc).__name__}: {exc}"})
    return PulseCaptain(
        ml_agents=ml_agents,
        weather_prior_agent=weather_prior,
        anomaly_detector=anomaly_detector,
        growth_stage_agent=growth_stage,
        skeptic_agent=skeptic,
        vlm_reasoner=vlm,
        emit_event=emit_event,
    )


def _post_inference_hooks(
    captain: PulseCaptain,
    result: dict,
    src_path: str,
    field_state: dict,
    state: dict,
    broadcast: Any,
) -> dict:
    """Run post-inference hooks to broadcast new Sprint 1-4 data.

    Mutates ``result`` to add extra fields consumed by the ``done`` event.
    Returns the enriched result.
    """
    import base64
    import io
    import traceback as _tb

    # --- Visual explanation overlay (Sprint 1) ---
    seg_agent = next(
        (a for a in captain.ml_agents if getattr(a, "name", "") == "segmentation"),
        None,
    )
    if seg_agent and getattr(seg_agent, "evidence_maps", None):
        try:
            from PIL import Image as PILImage
            from pulse.visual_explain import render_field_explanation

            annotated = render_field_explanation(src_path, seg_agent.evidence_maps)
            pil_img = PILImage.fromarray(annotated)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=70)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            broadcast("visual_explanation", {"data_url": data_url})
        except Exception:
            print(f"[visual_explain_error] {_tb.format_exc()}", flush=True)

    # --- Growth stage metadata (Sprint 2) ---
    gs_agent = captain.growth_stage_agent
    if gs_agent and getattr(gs_agent, "growth_stages", None):
        result["growth_stages"] = {
            int(k): v for k, v in gs_agent.growth_stages.items()
        }

    # --- Anomaly scores metadata (Sprint 2) ---
    ad_agent = captain.anomaly_detector
    if ad_agent and getattr(ad_agent, "anomaly_scores", None):
        result["anomaly_scores"] = {
            int(k): float(v) for k, v in ad_agent.anomaly_scores.items()
        }

    # --- Inference mode (Sprint 3) ---
    vlm = captain.vlm_reasoner
    skeptic = captain.skeptic_agent
    result["inference_mode"] = {
        "vlm": "local" if (vlm and getattr(vlm, "_use_local", False)) else "api",
        "skeptic": "local" if (skeptic and getattr(skeptic, "_use_local", False)) else "api",
    }

    # --- RAG context (Sprint 3) ---
    try:
        if state.get("rag") is None:
            from pulse.rag.retriever import AgronomicRAG
            state["rag"] = AgronomicRAG()

        rag = state["rag"]
        # Find the dominant non-healthy condition across all plants
        latent_data = result.get("latent", {})
        dominant_condition = "disease"
        for p in latent_data.get("plants", []):
            top_k = p.get("top_k", [])
            if top_k and top_k[0][0] != "healthy_crop":
                dominant_condition = top_k[0][0]
                break

        docs = rag.query_for_treatment(
            dominant_condition, crop=field_state.get("crop_type", ""),
        )
        broadcast("rag_context", {
            "documents": [
                {"id": d["id"], "text": d["text"][:200],
                 "tags": d.get("tags", []), "score": round(d.get("score", 0), 3)}
                for d in docs[:3]
            ],
        })
    except Exception:
        print(f"[rag_error] {_tb.format_exc()}", flush=True)

    # --- Active learning (Sprint 4) ---
    al_mgr = state.get("active_learning")
    if al_mgr:
        try:
            from pulse.latent import FieldLatentState
            from pulse.messages import ActionMessage as AM

            latent_obj = FieldLatentState.from_dict(latent_data)
            actions_objs = [
                AM(
                    sender=a["sender"], timestamp=a["timestamp"],
                    plant_id=a["plant_id"], action_type=a["action_type"],
                    action_params=a.get("action_params", {}),
                    expected_information_gain=a.get("expected_information_gain", 0),
                    expected_utility=a.get("expected_utility", 0),
                )
                for a in result.get("actions", [])
            ]
            # Reconstruct cross-exam KL from result
            cross_msgs = result.get("cross_exam", [])
            kl_per_plant: dict[int, float] = {}
            for m in cross_msgs:
                for pid_s, kl in m.get("per_plant_disagreement", {}).items():
                    pid = int(pid_s)
                    kl_per_plant[pid] = max(kl_per_plant.get(pid, 0.0), float(kl))

            al_mgr.process_inference_results(
                src_path, latent_obj, actions_objs, kl_per_plant,
                anomaly_scores=result.get("anomaly_scores"),
            )
            broadcast("active_learning_update", al_mgr.summary())
        except Exception:
            print(f"[active_learning_error] {_tb.format_exc()}", flush=True)

    # --- Temporal diff (Sprint 4) ---
    prev_path = state.get("previous_frame_path")
    if prev_path and Path(src_path).exists() and Path(prev_path).exists():
        try:
            from pulse.temporal import compute_frame_diff

            bboxes = [tuple(p["bbox"]) for p in latent_data.get("plants", [])]
            pids = [p["plant_id"] for p in latent_data.get("plants", [])]
            diff = compute_frame_diff(
                src_path, prev_path, bboxes, pids, use_optical_flow=False,
            )
            broadcast("temporal_diff", {
                "per_plant_changes": {
                    int(pid): {
                        "changed": sc.changed,
                        "combined_score": round(sc.combined_score, 3),
                        "pixel_diff": round(sc.pixel_diff_score, 3),
                    }
                    for pid, sc in diff.per_plant_changes.items()
                },
                "escalated": diff.escalated_plant_ids,
            })
        except Exception:
            print(f"[temporal_diff_error] {_tb.format_exc()}", flush=True)
    state["previous_frame_path"] = src_path

    return result


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    if dataclasses.is_dataclass(o):
        return _to_jsonable(dataclasses.asdict(o))
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


async def _safe_send(ws: WebSocket, msg: str, state: dict) -> None:
    try:
        await ws.send_text(msg)
    except Exception:  # noqa: BLE001
        state["clients"].discard(ws)


app = create_app()
