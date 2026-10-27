"""Safely convert BSLDict bsldict_v1.pkl to a small JSON file.

Only numpy array/scalar/dtype constructors are allowed; anything else aborts.
"""
import json, pickle, sys
import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        mod = module.replace("numpy.core", "numpy._core")
        return getattr(__import__(mod, fromlist=[name]), name)

def plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v

with open(sys.argv[1], "rb") as f:
    data = SafeUnpickler(f).load()

print("top keys:", list(data.keys()))
videos = data["videos"]
print("video keys:", list(videos.keys()))
keep = ["name", "word", "word_id", "alternative_words", "download_method_db", "video_link_db",
        "youtube_identifier_db", "source_site_db", "sign_link_db", "temp_interval", "bbox"]
n = len(videos["name"])
rows = []
for i in range(n):
    row = {k: plain(videos[k][i]) for k in keep if k in videos}
    for src, dst in (("videos_original", "orig"), ("videos_360h_25fps", "v360")):
        vo = videos.get(src)
        if isinstance(vo, dict):
            row[dst] = {k: plain(vo[k][i]) for k in ("T", "H", "W", "fps", "duration_sec") if k in vo}
    rows.append(row)
out = {"words": [plain(w) for w in data["words"]], "videos": rows}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist())
print("videos:", n, "words:", len(out["words"]))
print("sample:", json.dumps(rows[0], default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist())[:900])
