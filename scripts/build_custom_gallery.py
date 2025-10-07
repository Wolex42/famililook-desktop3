import pathlib, json, numpy as np, cv2, onnxruntime as ort
from insightface.app import FaceAnalysis

KEEP = r".\curated\good_frontal_unique_roll4"
OUTD = pathlib.Path(r".\artifacts\custom_gallery"); OUTD.mkdir(parents=True, exist_ok=True)

providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0 if providers[0]=="CUDAExecutionProvider" else -1, det_size=(512,512))

files, vecs = [], []
for p in pathlib.Path(KEEP).glob("*.*"):
    img = cv2.imread(str(p))
    if img is None: continue
    faces = app.get(img)
    if not faces: continue
    files.append(p.name)
    vecs.append(faces[0].normed_embedding)

if not vecs: raise SystemExit("No faces embedded.")

emb = np.vstack(vecs)  # (N,512)
np.save(OUTD/"embeddings.npy", emb)
json.dump({"files": files}, open(OUTD/"index.json","w",encoding="utf-8"))
print("saved", emb.shape, "to", OUTD)
