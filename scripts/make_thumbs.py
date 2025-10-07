import pathlib, cv2

SRC = pathlib.Path(r".\curated\good_frontal_unique_roll4")
DST = pathlib.Path(r".\artifacts\thumbs")
DST.mkdir(parents=True, exist_ok=True)

count = 0
for p in SRC.glob("*.*"):
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    s = 256.0 / max(h, w) if max(h, w) > 256 else 1.0
    resized = cv2.resize(img, (int(w*s), int(h*s)))
    # keep the same filename (extension) so /thumb/<filename> works
    cv2.imwrite(str(DST / p.name), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    count += 1

print(f"thumbs written: {count} -> {DST}")
