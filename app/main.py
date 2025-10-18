# app/main.py
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime

import os
from dotenv import load_dotenv

# Load environment variables from .env file (local dev only)
load_dotenv()

# Optional: Add secret key for production (if needed later)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-not-for-production")



from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .analytics import analytics

# Best-effort optional deps (engine runs in stub mode if missing)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:  # pragma: no cover
    FaceAnalysis = None  # type: ignore

API_VERSION = "0.4.1-analytics"

BUILD_INFO: Dict[str, Any] = {
    "api": "famililook",
    "version": API_VERSION,
    "capabilities": ["detect", "embed", "sanity", "version", "status", "explain", "analytics"],
    "device": "CPU",
    "provider": "onnxruntime",
    "models": None,
}

MODULE_STARTED_AT = time.time()


# ----------------------------- Engine Holder -----------------------------
class Engine:
    def __init__(self) -> None:
        self.ready: bool = False
        self.fa: Optional[FaceAnalysis] = None
        self.provider_names: List[str] = []
        self.models_pack: str = "buffalo_l"
        self.det_size = (640, 640)
        self.device_desc = "CPU"

    def load(self) -> None:
        if FaceAnalysis is None or cv2 is None or np is None:
            self.ready = False
            return
        
        download_models_if_needed()  # Add this line

        fa = FaceAnalysis(
            name=self.models_pack,
            allowed_modules=["detection", "recognition"],
        )
        try:
            fa.prepare(ctx_id=-1, det_size=self.det_size)
            self.fa = fa
            try:
                import onnxruntime as ort  # type: ignore
                self.provider_names = list(ort.get_available_providers())
                if any("CUDA" in p for p in self.provider_names):
                    self.device_desc = "CUDA"
            except Exception:
                self.provider_names = []
            self.ready = True
        except Exception:
            self.fa = None
            self.ready = False

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "models_pack": self.models_pack if self.ready else None,
            "providers": self.provider_names,
            "device": self.device_desc if self.ready else "CPU",
        }


engine = Engine()

# Add this helper function after the Engine class definition
def download_models_if_needed():
    """Pre-download models during startup"""
    import os
    from pathlib import Path
    
    models_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    
    if not models_dir.exists():
        print("📥 Downloading InsightFace models (this may take 2-5 minutes)...")
    else:
        print("✅ InsightFace models already cached")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("🚀 FamiliLook Engine API with Analytics")
    print("="*60)
    
    app.state.started_at = time.time()
    engine.load()
    app.state.ready = engine.ready
    BUILD_INFO["device"] = engine.device_desc if engine.ready else "CPU"
    BUILD_INFO["models"] = engine.models_pack if engine.ready else None
    
    if engine.ready:
        print(f"✅ Engine ready - Device: {engine.device_desc}")
    else:
        print("⚠️  Engine in stub mode")
    
    print(f"📊 Analytics enabled - Data: analytics_data/")
    print("="*60 + "\n")
    
    yield


app = FastAPI(title="Famililook Engine", lifespan=lifespan)

# CORS - Allow all for testing
# CORS - Production configuration
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Local development
    "http://localhost:4173",  # Local preview
    "https://famililook-beta.vercel.app",  # Production
    "https://famililook-beta-*.vercel.app",  # Preview deployments
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ----------------------------- Utilities -----------------------------
def _uptime_sec() -> float:
    started = getattr(app.state, "started_at", MODULE_STARTED_AT)
    return round(max(0.0, time.time() - started), 3)


def _is_ready() -> bool:
    return bool(getattr(app.state, "ready", True))


def _load_bgr(upload: UploadFile) -> Optional["np.ndarray"]:
    if cv2 is None or np is None:
        return None
    data = upload.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _boxes_from_faces(faces) -> List[List[float]]:
    out: List[List[float]] = []
    for f in faces:
        b = getattr(f, "bbox", None)
        if b is not None and len(b) >= 4:
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return out


def _landmarks_from_faces(faces) -> List[List[List[float]]]:
    out: List[List[List[float]]] = []
    for f in faces:
        kps = getattr(f, "kps", None)
        if kps is not None and np is not None:
            k = np.asarray(kps).tolist()
            out.append([[float(x), float(y)] for (x, y) in k])
        else:
            out.append([])
    return out


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(1e-6, area_a + area_b - inter)
    return inter / union


# ----------------------------- Meta -----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "message": "Famililook Engine", "uptime_sec": _uptime_sec()}


@app.get("/sanity")
def sanity() -> Dict[str, Any]:
    return {"ok": True, "uptime_sec": _uptime_sec()}


@app.get("/version")
def version() -> Dict[str, Any]:
    return BUILD_INFO


@app.get("/status")
def status() -> Dict[str, Any]:
    s = engine.status()
    s.update({"uptime_sec": _uptime_sec()})
    return s


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "ready": _is_ready(), "uptime_sec": _uptime_sec()}


# ----------------------------- DETECT (with analytics) ---------------------------------
@app.post("/detect")
async def detect(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    cfg: Optional[str] = Form(default=None),
):
    if file is None:
        raise HTTPException(status_code=422, detail="No file uploaded")
    
    # Log upload
    file_size = 0
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    analytics.log_upload(
        ip=request.client.host,
        filename=file.filename or "unknown",
        file_size=file_size,
        content_type=file.content_type or "unknown"
    )
    
    start_time = time.time()
    
    if not _is_ready() or not engine.ready or engine.fa is None:
        analytics.log_detection(
            ip=request.client.host,
            faces_detected=0,
            processing_time=time.time() - start_time,
            success=False
        )
        return JSONResponse(
            status_code=200,
            content={
                "faces": [],
                "summary": {"count": 0, "note": "stub detector (engine not ready)"},
                "cfg_echo": cfg,
                "filename": file.filename,
            },
        )

    img = _load_bgr(file)
    if img is None:
        analytics.log_error(
            ip=request.client.host,
            error_type="ImageLoadError",
            error_message="Could not load image",
            endpoint="/detect"
        )
        raise HTTPException(status_code=415, detail="Unsupported image or OpenCV unavailable")

    faces = engine.fa.get(img)
    boxes = _boxes_from_faces(faces)
    kps = _landmarks_from_faces(faces)
    scores = [float(getattr(f, "det_score", 0.0)) for f in faces]

    faces_out: List[Dict[str, Any]] = []
    for i, b in enumerate(boxes):
        faces_out.append({
            "box": b,
            "landmarks": kps[i],
            "score": scores[i],
        })
    
    processing_time = time.time() - start_time
    
    # Log detection result
    analytics.log_detection(
        ip=request.client.host,
        faces_detected=len(faces_out),
        processing_time=processing_time,
        success=True
    )

    return JSONResponse(
        status_code=200,
        content={
            "faces": faces_out,
            "summary": {"count": len(faces_out), "note": "insightface detect"},
            "cfg_echo": cfg,
            "filename": file.filename,
        },
    )


# ----------------------------- EMBED (with analytics) ---------------------------------
@app.post("/embed")
async def embed(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    boxes: Optional[str] = Form(default="[]"),
):
    if file is None:
        raise HTTPException(status_code=422, detail="No file uploaded")
    
    # Log upload
    file_size = 0
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    analytics.log_upload(
        ip=request.client.host,
        filename=file.filename or "unknown",
        file_size=file_size,
        content_type=file.content_type or "unknown"
    )
    
    start_time = time.time()

    if not _is_ready() or not engine.ready or engine.fa is None:
        return JSONResponse(
            status_code=200,
            content={
                "embeddings": [],
                "count": 0,
                "boxes_echo": boxes,
                "filename": file.filename,
                "note": "stub embedder (engine not ready)",
            },
        )

    img = _load_bgr(file)
    if img is None:
        analytics.log_error(
            ip=request.client.host,
            error_type="ImageLoadError",
            error_message="Could not load image for embedding",
            endpoint="/embed"
        )
        raise HTTPException(status_code=415, detail="Unsupported image or OpenCV unavailable")

    try:
        parsed = json.loads(boxes) if boxes else []
        if not isinstance(parsed, list):
            parsed = []
    except Exception:
        parsed = []

    faces = []
    if parsed:
        det = engine.fa.get(img)
        det_boxes = _boxes_from_faces(det)
        for user_box in parsed:
            best, best_idx = -1.0, -1
            for i, db in enumerate(det_boxes):
                v = _iou(user_box, db)
                if v > best:
                    best, best_idx = v, i
            if best_idx >= 0:
                faces.append(det[best_idx])
    else:
        faces = engine.fa.get(img)

    embs: List[List[float]] = []
    out_boxes: List[List[float]] = []
    for f in faces:
        emb = getattr(f, "embedding", None)
        if emb is not None and np is not None:
            embs.append([float(x) for x in np.asarray(emb).tolist()])
            b = getattr(f, "bbox", None)
            if b is not None:
                out_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    processing_time = time.time() - start_time
    
    # Log embedding
    analytics.log_feature_use(
        ip=request.client.host,
        feature="embedding",
        details={
            "faces_embedded": len(embs),
            "processing_time_ms": round(processing_time * 1000, 2)
        }
    )

    return JSONResponse(
        status_code=200,
        content={
            "embeddings": embs,
            "count": len(embs),
            "boxes": out_boxes,
            "filename": file.filename,
            "note": "insightface embed",
        },
    )


# ----------------------------- ANALYTICS ENDPOINTS -----------------------------
@app.get("/analytics/summary")
def get_analytics_summary(date: Optional[str] = None):
    """Get analytics summary for a specific date"""
    summary = analytics.generate_daily_summary(date)
    return summary


@app.get("/analytics/dashboard")
def analytics_dashboard():
    """Analytics dashboard - today's summary"""
    today_summary = analytics.generate_daily_summary()
    return JSONResponse(content={
        "today": today_summary,
        "message": "Today's analytics summary"
    })


@app.post("/analytics/track")
async def track_analytics(request: Request, event: Dict[str, Any]):
    """Receive frontend analytics events"""
    try:
        event_type = event.get("eventType", "unknown")
        
        # Log based on event type
        if event_type == "analysis":
            analytics.log_analysis(
                ip=request.client.host,
                child_count=event.get("childCount", 0),
                processing_time=event.get("durationMs", 0) / 1000,
                results=None
            )
        elif event_type == "feature_use":
            analytics.log_feature_use(
                ip=request.client.host,
                feature=event.get("feature", "unknown"),
                details=event.get("details", {})
            )
        elif event_type == "error":
            analytics.log_error(
                ip=request.client.host,
                error_type=event.get("errorType", "UnknownError"),
                error_message=event.get("errorMessage", ""),
                endpoint=event.get("url", "")
            )
        elif event_type == "session_start":
            analytics.log_session_start(
                ip=request.client.host,
                plan=event.get("plan", "free")
            )
        
        return {"ok": True}
    except Exception as e:
        # Silently fail - don't break the app if analytics fails
        print(f"Analytics tracking error: {e}")
        return {"ok": False, "error": str(e)}


# ----------------------------- EXPLAIN (stub) -------------------------
@app.post("/explain")
async def explain_stub(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "note": "Explain API not yet implemented on server.",
        "echo": payload,
    }