import json, sys, pathlib

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_matches(obj):
    # Support multiple schemas
    return (obj.get("matches")
            or obj.get("neighbors")
            or obj.get("results")
            or [])

def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r".\artifacts\probe.json")
    j = load_json(path)
    matches = pick_matches(j)
    if not matches:
        print("No matches found in JSON. Keys:", list(j.keys()))
        return
    print(f"Top {min(10,len(matches))} matches from {path}:")
    for i, m in enumerate(matches[:10], 1):
        mid = m.get("id") or m.get("path") or m.get("name") or m.get("file") or "?"
        score = m.get("score") or m.get("sim") or m.get("distance")
        print(f"{i:2d}. {mid}  score={score}")
    print()
    print("Probe image:", j.get("image") or j.get("query") or "?")
    print("Gallery:", j.get("gallery") or "auto-loaded")

if __name__ == "__main__":
    main()
