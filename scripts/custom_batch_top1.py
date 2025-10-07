import csv, json, pathlib, numpy as np, cv2, onnxruntime as ort
from insightface.app import FaceAnalysis

KEEP = pathlib.Path(r".\curated\good_frontal_unique_roll4")
GAL  = pathlib.Path(r".\artifacts\custom_gallery")
OUT  = pathlib.Path(r".\artifacts\custom_batch_top1.csv")

# load gallery
emb = np.load(GAL/"embeddings.npy")           # (N,512), L2-normalized
idx = json.load(open(GAL/"index.json","r",encoding="utf-8"))["files"]  # list of filenames

# init face analysis (GPU if available)
providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0 if providers[0]=="CUDAExecutionProvider" else -1, det_size=(512,512))

def embed(path: pathlib.Path):
    img = cv2.imread(str(path))
    if img is None: return None
    faces = app.get(img)
    if not faces: return None
    return faces[0].normed_embedding

rows = []
files = sorted([p for p in KEEP.glob("*.*") if p.is_file()])
for i, p in enumerate(files, 1):
    q = embed(p)
    if q is None:
        rows.append({"image": p.name, "best_id": "", "score": "", "note": "no_face"})
        continue
    scores = emb @ q  # cosine similarity (embeddings are L2-normalized)
    # optional: exclude exact filename self-match if present
    try:
        self_idx = idx.index(p.name)
        scores[self_idx] = -1.0
    except ValueError:
        pass
    top = int(np.argmax(scores))
    rows.append({"image": p.name, "best_id": idx[top], "score": float(scores[top]), "note": ""})
    if i % 200 == 0:
        print(f"processed {i}/{len(files)}")

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image","best_id","score","note"])
    w.writeheader(); w.writerows(rows)

print(f"wrote {OUT} ({len(rows)} rows)")
