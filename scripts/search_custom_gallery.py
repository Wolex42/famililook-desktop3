import json, pathlib, numpy as np, cv2, onnxruntime as ort
from insightface.app import FaceAnalysis

PROBE = (pathlib.Path(r".\curated\good_frontal_unique_roll4").glob("*.*")).__iter__().__next__()  # first image
GAL   = pathlib.Path(r".\artifacts\custom_gallery")
emb   = np.load(GAL/"embeddings.npy")
idx   = json.load(open(GAL/"index.json","r",encoding="utf-8"))["files"]

providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0 if providers[0]=="CUDAExecutionProvider" else -1, det_size=(512,512))

img = cv2.imread(str(PROBE)); faces = app.get(img)
if not faces: raise SystemExit("no face in probe")
q = faces[0].normed_embedding  # L2-normalized (512,)
scores = emb @ q               # cosine similarity
top = np.argsort(-scores)[:10]
out = [{"id": idx[i], "score": float(scores[i])} for i in top]
print(json.dumps({"probe": str(PROBE), "top10": out}, indent=2))
