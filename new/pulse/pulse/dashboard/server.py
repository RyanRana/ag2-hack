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

from pulse.agents.disease_classifier import DiseaseClassifierAgent
from pulse.agents.health_classifier import HealthClassifierAgent
from pulse.agents.segmentation import SegmentationAgent
from pulse.agents.weed_detector import WeedDetectorAgent
from pulse.captain import PulseCaptain
from pulse.llm_config import llm_key_available


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
    for pid, b in enumerate(frame.get("boxes", [])):
        x1 = max(0, int((b["xc"] - b["w"] / 2) * img_size))
        y1 = max(0, int((b["yc"] - b["h"] / 2) * img_size))
        x2 = min(img_size, int((b["xc"] + b["w"] / 2) * img_size))
        y2 = min(img_size, int((b["yc"] + b["h"] / 2) * img_size))
        prior = np.full(len(CONDITION_LABELS), -np.log(len(CONDITION_LABELS)))
        # Strong GT pre-skew so the dataset's annotations anchor the
        # weed-vs-crop call even when the ML disease classifier gets
        # confused on aerial drone imagery.
        if b["class"] == "weed":
            prior[i_weed] += 6.0
        else:
            prior[i_healthy] += 3.0
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
                _broadcast("done", {
                    "actions": result["actions"],
                    "frame_index": frame_index,
                    "field_state": field_state,
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
                _broadcast("done", {"actions": result["actions"]})
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
                captain = _build_captain(emit_event=_broadcast)
                result = await asyncio.to_thread(
                    captain.run_inference, str(path), _default_field_state()
                )
                _broadcast("done", {"actions": result["actions"]})
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
                _broadcast("done", {"actions": result["actions"]})
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
    # Each ML agent loads its model lazily on first use, so cheap to construct.
    ml_agents.append(WeedDetectorAgent())
    ml_agents.append(DiseaseClassifierAgent())
    ml_agents.append(SegmentationAgent())
    ml_agents.append(HealthClassifierAgent())
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
        skeptic_agent=skeptic,
        vlm_reasoner=vlm,
        emit_event=emit_event,
    )


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
