"""Hand-landmark features and sequence matching for sign recognition.

This file must stay in step with demo/static/demo/sign_matcher.js, which runs the
same maths live in the browser. No training is involved: a sign is recognised by
comparing the live hand movement with recorded example clips (templates) using
dynamic time warping (DTW).

Per hand and frame the features are:
  - handshape: how bent each finger joint is, and how spread the fingers are (angles, so
    they do not change with hand size, rotation or distance from the camera)
  - orientation: which way the hand points and which way the palm faces
  - location: where the wrist is relative to the shoulders

Frame input (from MediaPipe):
    hands: list of {"label": "Left"|"Right", "lm": [[x, y, z] * 21]}  (normalised image coords)
    body:  {"nose": [x, y], "ls": [x, y], "rs": [x, y]} or None        (pose: nose, shoulders)
    aspect: image width / height
"""
import math

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
SEQ_LEN = 12             # every sign is resampled to this many frames
SMOOTH = 5               # hand points are averaged over this many neighbouring frames (reduces jitter)
N_SHAPE = 18             # 15 joint angles + 3 finger spreads
N_ORIENT = 5             # pointing direction (x, y) + palm normal (x, y, z)
SLOT_DIM = 1 + N_SHAPE + N_ORIENT + 2   # present, shape, orientation, location
FRAME_DIM = 2 * SLOT_DIM + 1            # dominant hand, other hand, has-body flag

# Matching weights (tuned with tune_params.py; written into sign_templates.json for the browser)
BAND = 3                 # DTW warping window
SHAPE_WEIGHT = 1.5
ORIENT_WEIGHT = 0.5
LOC_WEIGHT = 0.5
MISSING_HAND_COST = 1.0  # one sequence has the hand, the other doesn't
NONDOMINANT_WEIGHT = 0.6
# Sign embedding (train_embedding.py). A sign's distance to an example combines DTW and the embedding:
#   dtw / DTW_SCALE + EMB_WEIGHT * (1 - cosine similarity) / EMB_SCALE    (EMB_WEIGHT 0 = DTW only)
DTW_SCALE = 1.0
EMB_SCALE = 1.0
EMB_WEIGHT = 0.0


def _pt(p, aspect):
    return p[0] * aspect, p[1]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _angle(a, b, c):
    """Angle at b between b->a and b->c, as a fraction of pi."""
    v1, v2 = _sub(a, b), _sub(c, b)
    n = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if n < 1e-9:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    return math.acos(max(-1.0, min(1.0, dot / n))) / math.pi


def dominant_label(frames):
    """The hand that moves most is treated as the dominant hand."""
    travel = {"Left": 0.0, "Right": 0.0}
    seen = {"Left": 0, "Right": 0}
    last = {}
    for fr in frames:
        aspect = fr["aspect"]
        for h in fr["hands"]:
            lab = h["label"]
            w = _pt(h["lm"][0], aspect)
            seen[lab] += 1
            if lab in last:
                travel[lab] += math.dist(w, last[lab])
            last[lab] = w
    if not seen["Left"] and not seen["Right"]:
        return "Right"
    if seen["Left"] and seen["Right"]:
        return "Left" if travel["Left"] > travel["Right"] else "Right"
    return "Left" if seen["Left"] else "Right"


def _hand_slot(lm, aspect, mirror, origin, scale):
    P = [(mirror * p[0] * aspect, p[1], p[2] * aspect) for p in lm]
    out = [1.0]
    for a, b, c, d in FINGERS:
        out += [_angle(P[0], P[a], P[b]), _angle(P[a], P[b], P[c]), _angle(P[b], P[c], P[d])]
    for i in range(1, 4):
        base = P[FINGERS[i][0]]
        u = _sub(P[FINGERS[i][1]], base)
        w = _sub(P[FINGERS[i + 1][1]], P[FINGERS[i + 1][0]])
        out.append(_angle((base[0] + u[0], base[1] + u[1], base[2] + u[2]), base,
                          (base[0] + w[0], base[1] + w[1], base[2] + w[2])))
    d = _sub(P[9], P[0])
    dn = math.hypot(d[0], d[1]) + 1e-9
    e1, e2 = _sub(P[5], P[0]), _sub(P[17], P[0])
    nrm = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
    nn = math.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2) + 1e-9
    out += [d[0] / dn, d[1] / dn, nrm[0] / nn, nrm[1] / nn, nrm[2] / nn]
    if origin:
        out += [(P[0][0] - mirror * origin[0]) / scale, (P[0][1] - origin[1]) / scale]
    else:
        out += [0.0, 0.0]
    return out


def frame_vector(fr, dom):
    aspect = fr["aspect"]
    mirror = -1.0 if dom == "Left" else 1.0
    body = fr.get("body")
    origin = scale = None
    if body:
        ls, rs = _pt(body["ls"], aspect), _pt(body["rs"], aspect)
        scale = math.dist(ls, rs)
        if scale > 1e-6:
            origin = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    vec = []
    by_label = {h["label"]: h for h in fr["hands"]}
    for lab in (dom, "Right" if dom == "Left" else "Left"):
        h = by_label.get(lab)
        vec += _hand_slot(h["lm"], aspect, mirror, origin, scale) if h else [0.0] * SLOT_DIM
    vec.append(1.0 if origin else 0.0)
    return vec


def smooth_hands(frames):
    """Average each hand's points over SMOOTH neighbouring frames (same hand label only)."""
    half = SMOOTH // 2
    out = []
    for i, fr in enumerate(frames):
        hands = []
        for h in fr["hands"]:
            acc, n = [[0.0, 0.0, 0.0] for _ in range(21)], 0
            for j in range(max(0, i - half), min(len(frames), i + half + 1)):
                m = next((g for g in frames[j]["hands"] if g["label"] == h["label"]), None)
                if m:
                    n += 1
                    for k in range(21):
                        acc[k][0] += m["lm"][k][0]
                        acc[k][1] += m["lm"][k][1]
                        acc[k][2] += m["lm"][k][2]
            hands.append({"label": h["label"], "lm": [[c[0] / n, c[1] / n, c[2] / n] for c in acc]})
        out.append({"aspect": fr["aspect"], "hands": hands, "body": fr.get("body")})
    return out


def sequence_vectors(frames, dom=None):
    """Frames -> list of SEQ_LEN feature vectors (smoothing, then nearest-frame resampling)."""
    frames = [f for f in smooth_hands(frames) if f["hands"]]
    if len(frames) < 3:
        return None
    dom = dom or dominant_label(frames)
    vecs = [frame_vector(f, dom) for f in frames]
    n = len(vecs)
    # floor(x + 0.5) matches JavaScript Math.round (Python round() is banker's rounding)
    return [vecs[math.floor(i * (n - 1) / (SEQ_LEN - 1) + 0.5)] for i in range(SEQ_LEN)]


def query_variants(frames):
    """The live sequence read with each hand as the dominant one.

    For two-handed signs both hands move, so which one counts as dominant can flip between
    recordings; matching takes the closer of the two readings (see best_distance)."""
    return [s for s in (sequence_vectors(frames, "Right"), sequence_vectors(frames, "Left")) if s]


def best_distance(variants, template):
    return min(dtw(v, template) for v in variants)


def static_vector(frames, dom):
    """A held hand shape (no movement): the average frame vector over the smoothed frames with hands."""
    frames = [f for f in smooth_hands(frames) if f["hands"]]
    if not frames:
        return None
    vecs = [frame_vector(f, dom) for f in frames]
    return [sum(col) / len(vecs) for col in zip(*vecs)]


def static_variants(frames):
    """The held shape read with each hand as the dominant one (as query_variants)."""
    return [v for v in (static_vector(frames, "Right"), static_vector(frames, "Left")) if v]


def _slot_cost(a, b, off, both_body):
    pa, pb = a[off] >= 0.5, b[off] >= 0.5
    if not pa and not pb:
        return 0.0
    if pa != pb:
        return MISSING_HAND_COST
    s = sum(abs(a[off + 1 + i] - b[off + 1 + i]) for i in range(N_SHAPE)) / N_SHAPE * 3
    o0 = off + 1 + N_SHAPE
    o = math.sqrt(sum((a[o0 + i] - b[o0 + i]) ** 2 for i in range(N_ORIENT)))
    cost = SHAPE_WEIGHT * s + ORIENT_WEIGHT * o
    if both_body:
        j = o0 + N_ORIENT
        cost += LOC_WEIGHT * math.hypot(a[j] - b[j], a[j + 1] - b[j + 1])
    return cost


def frame_cost(a, b):
    both_body = a[-1] >= 0.5 and b[-1] >= 0.5
    return _slot_cost(a, b, 0, both_body) + NONDOMINANT_WEIGHT * _slot_cost(a, b, SLOT_DIM, both_body)


def dtw(a, b):
    """Average frame cost along the best alignment of two SEQ_LEN sequences."""
    n, inf = len(a), float("inf")
    d = [[inf] * (n + 1) for _ in range(n + 1)]
    steps = [[0] * (n + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - BAND), min(n, i + BAND) + 1):
            c = frame_cost(a[i - 1], b[j - 1])
            # first strict minimum wins (same tie rule as sign_matcher.js)
            best, k = d[i - 1][j - 1], steps[i - 1][j - 1]
            if d[i - 1][j] < best:
                best, k = d[i - 1][j], steps[i - 1][j]
            if d[i][j - 1] < best:
                best, k = d[i][j - 1], steps[i][j - 1]
            d[i][j] = best + c
            steps[i][j] = k + 1
    return d[n][n] / max(steps[n][n], 1)


def params():
    """Matching weights for the browser (sign_templates.json -> sign_matcher.configure)."""
    return {"band": BAND, "shapeWeight": SHAPE_WEIGHT, "orientWeight": ORIENT_WEIGHT, "locWeight": LOC_WEIGHT,
            "missingHandCost": MISSING_HAND_COST, "nondominantWeight": NONDOMINANT_WEIGHT,
            "dtwScale": DTW_SCALE, "embScale": EMB_SCALE, "embWeight": EMB_WEIGHT}


# ---------------------------------------------------------------------------
# Sign embedding: a small network (trained by train_embedding.py on other dictionary words) that turns a
# SEQ_LEN x FRAME_DIM sequence into 64 numbers; the same sign by different signers lands close together.
# sign_matcher.js `embed` runs the same three layers.

def unpack_embedding(emb):
    """sign_embedding.json -> list of networks, each {"layers": [{"w": rows, "b": list}]} (8-bit weights x scale)."""
    import base64
    models = []
    for m in emb["models"]:
        layers = []
        for l in m["layers"]:
            raw = base64.b64decode(l["q"])
            rows, cols = l["shape"]
            q = [x - 256 if x > 127 else x for x in raw]
            layers.append({"w": [[v * l["scale"] for v in q[r * cols:(r + 1) * cols]] for r in range(rows)], "b": l["b"]})
        models.append({"layers": layers})
    return models


def embed(model, seq):
    """model: {"layers": [{"w": [[...]], "b": [...]}, ...]} -> unit-length list."""
    h = [x for fr in seq for x in fr]
    layers = model["layers"]
    for k, layer in enumerate(layers):
        h = [sum(wi * hi for wi, hi in zip(row, h)) + b for row, b in zip(layer["w"], layer["b"])]
        if k < len(layers) - 1:
            h = [max(0.0, x) for x in h]
    n = math.sqrt(sum(x * x for x in h)) or 1e-12
    return [x / n for x in h]


def embed_batch(model, seqs):
    """numpy version of embed for many sequences at once (same numbers up to rounding)."""
    import numpy as np
    h = np.asarray(seqs, dtype=float).reshape(len(seqs), -1)
    layers = model["layers"]
    for k, layer in enumerate(layers):
        h = h @ np.asarray(layer["w"]).T + np.asarray(layer["b"])
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)


def fuse(dtw_distance, emb_distance):
    return dtw_distance / DTW_SCALE + EMB_WEIGHT * emb_distance / EMB_SCALE


# ---------------------------------------------------------------------------
# Vectorised versions (numpy) used by build_index.py for the offline accuracy check.
# They compute exactly the same numbers as frame_cost/dtw above.

def _slot_cost_np(a, b, off, both_body):
    import numpy as np
    pa, pb = a[..., off] >= 0.5, b[..., off] >= 0.5
    s = np.abs(a[..., off + 1: off + 1 + N_SHAPE] - b[..., off + 1: off + 1 + N_SHAPE]).sum(-1) / N_SHAPE * 3
    o0 = off + 1 + N_SHAPE
    o = np.sqrt(((a[..., o0: o0 + N_ORIENT] - b[..., o0: o0 + N_ORIENT]) ** 2).sum(-1))
    j = o0 + N_ORIENT
    loc = np.sqrt(((a[..., j:j + 2] - b[..., j:j + 2]) ** 2).sum(-1))
    cost = SHAPE_WEIGHT * s + ORIENT_WEIGHT * o + np.where(both_body, LOC_WEIGHT * loc, 0.0)
    return np.where(pa & pb, cost, np.where(pa ^ pb, MISSING_HAND_COST, 0.0))


def dtw_batch(query, templates):
    """query: (SEQ_LEN, FRAME_DIM); templates: (N, SEQ_LEN, FRAME_DIM) -> (N,) distances."""
    import numpy as np
    a = query[None, :, None, :]            # (1, n, 1, D)
    b = templates[:, None, :, :]           # (N, 1, n, D)
    both_body = (a[..., -1] >= 0.5) & (b[..., -1] >= 0.5)
    cost = _slot_cost_np(a, b, 0, both_body) + NONDOMINANT_WEIGHT * _slot_cost_np(a, b, SLOT_DIM, both_body)
    N, n = templates.shape[0], query.shape[0]
    d = np.full((N, n + 1, n + 1), np.inf)
    steps = np.zeros((N, n + 1, n + 1))
    d[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - BAND), min(n, i + BAND) + 1):
            cands = np.stack([d[:, i - 1, j - 1], d[:, i - 1, j], d[:, i, j - 1]])
            stp = np.stack([steps[:, i - 1, j - 1], steps[:, i - 1, j], steps[:, i, j - 1]])
            pick = np.argmin(cands, axis=0)  # ties -> first, same order as dtw()
            d[:, i, j] = cands[pick, np.arange(N)] + cost[:, i - 1, j - 1]
            steps[:, i, j] = stp[pick, np.arange(N)] + 1
    return d[:, n, n] / np.maximum(steps[:, n, n], 1)
