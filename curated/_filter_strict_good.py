# C:\Users\wole\Documents\familook-desktop3\curated\_filter_strict_good.py
from __future__ import annotations

import argparse, csv, shutil
from pathlib import Path

def getf(row, *names, default=None, cast=float):
    """
    Fetch and cast the first present column among names.
    Returns default if none exist or casting fails.
    """
    for n in names:
        if n in row and row[n] not in (None, "", "NaN", "nan"):
            try:
                return cast(row[n])
            except Exception:
                pass
    return default

def getstr(row, *names, default=None):
    """Like getf but returns a string (no casting)."""
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return default

def main():
    ap = argparse.ArgumentParser(
        description="Filter a strict set of high-quality frontals from categorisation CSV."
    )
    ap.add_argument("--csv", required=True, help="Path to categorisation_reasons.csv")
    ap.add_argument("--root", required=True, help="Root that CSV paths are relative to (e.g., ...\\categories)")
    ap.add_argument("--out", required=True, help="Destination folder for strict-keep images")
    ap.add_argument("--yaw-max", type=float, default=8.0)
    ap.add_argument("--roll-max", type=float, default=6.0)
    ap.add_argument("--min-sharp", type=float, default=150.0)
    ap.add_argument("--min-face-ratio", type=float, default=0.14)
    ap.add_argument("--min-det-conf", type=float, default=0.55)
    ap.add_argument("--write-csv", action="store_true", help="Write filtered CSV alongside outputs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    root     = Path(args.root)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = total = missing = 0
    filtered_rows = []

    # === Aliases aligned to your CSV header ===
    # CSV header you showed:
    # source_path,bucket,reason,num_faces,det_score,roll_deg,yaw_deg,asymmetry,face_ratio,dark_glasses,dark_score,touches_border,sharpness,det_used,det_retried,copied_to
    path_aliases  = ("source_path", "relpath", "path", "image", "img", "file", "filename", "copied_to")
    yaw_aliases   = ("yaw_deg", "yaw", "pose_yaw", "yaw_abs", "yaw_deg_abs")
    roll_aliases  = ("roll_deg", "roll", "pose_roll", "roll_abs", "roll_deg_abs")
    sharp_aliases = ("sharpness", "sharp", "laplace", "variance_of_laplacian")
    ratio_aliases = ("face_ratio", "face_area_ratio", "min_face_ratio", "ratio")
    conf_aliases  = ("det_score", "det_conf", "confidence", "det_confidence", "min_det_conf")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        total += 1

        yaw   = abs(getf(row, *yaw_aliases,   default=999.0))
        roll  = abs(getf(row, *roll_aliases,  default=999.0))
        sharp =     getf(row, *sharp_aliases, default=-1.0)
        ratio =     getf(row, *ratio_aliases, default=-1.0)
        conf  =     getf(row, *conf_aliases,  default=-1.0)

        # strict filters
        if yaw > args.yaw_max: continue
        if roll > args.roll_max: continue
        if sharp < args.min_sharp: continue
        if ratio < args.min_face_ratio: continue
        if conf  < args.min_det_conf: continue

        rel = getstr(row, *path_aliases)
        if not rel:
            continue

        src = Path(rel)
        if not src.is_absolute():
            # most runs store relative to categories root
            src = root / src

        if not src.exists():
            # last-ditch: search by filename under categories
            matches = list(root.rglob(Path(rel).name))
            if matches:
                src = matches[0]
            else:
                missing += 1
                continue

        # preserve relative structure when possible
        try:
            rel_to_root = src.relative_to(root)
        except ValueError:
            rel_to_root = Path(src.name)

        dest = out_dir / rel_to_root
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        kept += 1

        row_out = dict(row)
        row_out["_strict_keep"] = "1"
        row_out["_src"] = str(src)
        row_out["_dst"] = str(dest)
        filtered_rows.append(row_out)

    if args.write_csv and filtered_rows:
        out_csv = out_dir / "strict_keep.csv"
        fieldnames = sorted(set().union(*[set(r.keys()) for r in filtered_rows]))
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(filtered_rows)

    print(f"[strict] scanned: {total} | kept: {kept} | missing files: {missing}")
    print(f"[strict] out: {out_dir}")

if __name__ == "__main__":
    main()
