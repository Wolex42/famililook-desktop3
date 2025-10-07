# familook/service/search_service.py
# GPU-backed face similarity API with labels + thumbnails + batch CSV
# Requires: insightface, onnxruntime-gpu (or cpu), fastapi, uvicorn, python-multipart, opencv-python

import json
import pathlib

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from insightface.app import FaceAnalysis
from pydantic import BaseModel

# --- paths ---
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root
GALLERY_DIR = ROOT / "artifacts" / "custom_gallery"
THUMBS_DIR = ROOT / "artifacts" / "thumbs"          # optional, for /thumb/<name>
LABELS_CSV = ROOT / "familook" / "service" / "labels.csv"
CURATED_KEEP = ROOT / "curated" / "good_frontal_unique_roll4"  # used by /search-folder

# --- load gallery (embeddings + filenames) ---
EMB = np.load(GALLERY_DIR / "embeddings.npy")  # (N,512) L2-normalized
IDX = json.load(open(GALLERY_DIR / "index.json", "r", encoding="utf-8"))["files"]  # list[str]

# --- load labels (optional) ---
labels: dict[str, dict] = {}
if LABELS_CSV.exists():
    import csv

    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = (row.get("filename") or "").strip()
            if not fname:
                continue
            labels[fname] = {
                "person_id": (row.get("person_id") or "").strip(),
                "display_name": (row.get("display_name") or "").strip(),
            }

# --- helper to decorate a search hit with label/thumb, score set later ---
def decorate(i: int):
    fname = IDX[i]
    item = {"id": fname, "score": None}
    # add label if present
    if fname in labels:
        item.update(labels[fname])
    # add thumb URL if the file exists
    if (THUMBS_DIR / fname).exists():
        item["thumb"] = f"/thumb/{fname}"
    return item


# --- InsightFace init (prefer CUDA if available) ---
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)
fa = FaceAnalysis(name="buffalo_l", providers=providers)
fa.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1, det_size=(512, 512))


# --- FastAPI app + CORS ---
api = FastAPI(title="Familook Similarity API", version="0.2")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production (e.g., ["http://localhost:5173"])
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResponse(BaseModel):
    probe_name: str
    topk: int
    hits: list


def embed_bytes(content: bytes):
    """Decode bytes -> BGR image -> run detector -> return first face embedding (512-D, L2-normalized)."""
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = fa.get(img)
    if not faces:
        return None
    return faces[0].normed_embedding


@api.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": ("CUDAExecutionProvider" in providers),
        "gallery_size": len(IDX),
    }


@api.get("/labels")
def get_labels():
    """Return the filename -> {person_id, display_name} mapping."""
    return labels


@api.get("/thumb/{name}")
def thumb(name: str):
    """Serve precomputed thumbnails if present (generated into artifacts/thumbs/)."""
    p = THUMBS_DIR / name
    if not p.exists():
        raise HTTPException(404, "thumb not found")
    return FileResponse(str(p), media_type="image/jpeg")


@api.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    topk: int = Form(5),
    exclude_self: bool = Form(True),
):
    """Search the gallery for nearest neighbors of the uploaded photo."""
    data = await file.read()
    q = embed_bytes(data)
    if q is None:
        raise HTTPException(status_code=422, detail="No face detected in probe.")

    scores = EMB @ q  # cosine similarity (embeddings are L2-normalized)

    # Optional: suppress exact filename self-match if the probe name exists in gallery
    if exclude_self and file.filename in IDX:
        self_idx = IDX.index(file.filename)
        scores[self_idx] = -1.0

    k = int(topk)
    k = max(1, min(k, len(scores)))

    order = np.argpartition(-scores, range(k))[:k]
    order = order[np.argsort(-scores[order])]

    hits = [decorate(i) for i in order]
    for h, i in zip(hits, order):
        h["score"] = float(scores[i])

    return JSONResponse(SearchResponse(probe_name=file.filename, topk=k, hits=hits).dict())


@api.post("/search-folder")
async def search_folder(topk: int = Form(1)):
    """
    Batch QA: probe each image in the curated keep folder, return CSV with top-k neighbors
    (self-match excluded). Adjust CURATED_KEEP above if your keep path changes.
    """
    import io
    import csv

    if not CURATED_KEEP.exists():
        raise HTTPException(400, f"Keep folder not found: {CURATED_KEEP}")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["image", "rank", "id", "score", "display_name", "person_id", "thumb"])

    for fname in IDX:
        img_path = CURATED_KEEP / fname
        if not img_path.exists():
            # skip if the curated image is missing; you can log this if needed
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = fa.get(img)
        if not faces:
            continue

        q = faces[0].normed_embedding
        scores = EMB @ q

        # exclude self
        try:
            i_self = IDX.index(fname)
            scores[i_self] = -1.0
        except ValueError:
            pass

        k = max(1, min(int(topk), len(scores)))
        order = np.argpartition(-scores, range(k))[:k]
        order = order[np.argsort(-scores[order])]

        for r, i in enumerate(order, 1):
            hit = decorate(i)
            hit["score"] = float(scores[i])
            w.writerow(
                [
                    fname,
                    r,
                    hit["id"],
                    hit["score"],
                    hit.get("display_name", ""),
                    hit.get("person_id", ""),
                    hit.get("thumb", ""),
                ]
            )

    return PlainTextResponse(buf.getvalue(), media_type="text/csv")
