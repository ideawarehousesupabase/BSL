"""Build the browser data files from processed clips and landmarks.

Outputs (in demo/static/demo/data/):
  sign_clips.json      word -> trimmed BSL clip URLs, used by Speech -> Sign
  sign_templates.json  landmark templates for the recognisable vocabulary, used by Sign -> Speech
  eval_report.json     held-out accuracy, so the numbers shown to the client are measured

The recognised words come from scripts/vocabulary.txt (chosen by select_words.py). Accuracy is measured
by matching every clip (with small landmark jitter, like live camera input) against all other clips
(never itself), so each test uses a signer or source the word's other examples came from.

Usage:
    python build_index.py --meta bsldict_meta.json --clips <project>/media/signs \
        --landmarks <landmark_dir> --out <project>/demo/static/demo/data --vocab vocabulary.txt \
        [--static-vocab static_vocabulary.txt] [--embedding sign_embedding.json --emb-weight 2]
"""
import argparse
import collections
import json
import math
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sign_features as sf  # noqa: E402

MIN_CLIPS = 2        # a word needs at least two clips so it can be tested
TEST_NOISE = 0.004   # jitter added to test clips, similar to live-camera landmark noise
SELECT_DRAWS = 3     # jittered copies of each clip used to set the accept threshold
REPORT_DRAWS = 3     # fresh jittered copies (different seed) used only to report accuracy
STATIC_HOLD = 5      # a clip gives a held-shape example if the wrist stays still this many frames (0.2 s)
STATIC_STILL = 0.06  # wrist movement per frame (in palm widths, after smoothing) that still counts as held
# Dictionary clips whose held hand shape is one of the client's demo gestures. They are added to the
# held-shape matcher as competitors: a held hand closest to one of these is left to the demo-gesture
# recognisers instead of being read as the nearest held-shape word.
DEMO_SHAPES = {"Peace": ["two", "v", "2"], "Four": ["four", "4"], "Stop": ["five", "5", "hello"],
               "Three": ["three", "3"], "Good job": ["good"], "Help": ["1", "one"]}


def label(word: str) -> str:
    return word.replace("-", " ")


def jitter(frames: list[dict], rng: random.Random) -> list[dict]:
    """Copy of the frames with small random landmark noise, so the test resembles live input."""
    g = lambda: rng.gauss(0, TEST_NOISE)  # noqa: E731
    return [{"aspect": fr["aspect"],
             "hands": [{"label": h["label"], "lm": [[p[0] + g(), p[1] + g(), p[2] + g()] for p in h["lm"]]}
                       for h in fr["hands"]],
             "body": {k: [v[0] + g(), v[1] + g()] for k, v in fr["body"].items()} if fr.get("body") else None}
            for fr in frames]


def load_templates(lmdir: Path) -> list[dict]:
    out = []
    for f in sorted(lmdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        a, b = d["core"]
        frames = d["frames"][a:b + 1]
        seq = sf.sequence_vectors(frames)
        if seq:
            out.append({"word": d["word"], "name": d["name"], "seq": seq, "frames": frames})
    return out


def raw_distances(templates: list[dict], seed: int, model=None) -> tuple[np.ndarray, np.ndarray | None]:
    """D[i, j] = DTW distance from a jittered copy of clip i (read either way round, as live input
    is) to clean template j; E[i, j] the same in the sign embedding (if a model is given).
    A clip is never compared with its own template."""
    rng = random.Random(seed)
    stack = np.array([t["seq"] for t in templates], dtype=float)
    nets = sf.unpack_embedding(model) if model else []
    temb = [sf.embed_batch(n, [t["seq"] for t in templates]) for n in nets]
    rows, erows = [], []
    for t in templates:
        variants = sf.query_variants(jitter(t["frames"], rng)) or [t["seq"]]
        rows.append(np.min([sf.dtw_batch(np.array(v, dtype=float), stack) for v in variants], axis=0))
        if model:   # average over the networks of the closest reading
            erows.append(np.mean([(1 - sf.embed_batch(n, variants) @ te.T).min(0) for n, te in zip(nets, temb)], axis=0))
    D = np.stack(rows)
    np.fill_diagonal(D, np.inf)
    E = None
    if model:
        E = np.stack(erows)
        np.fill_diagonal(E, np.inf)
    return D, E


def distance_matrix(templates: list[dict], seed: int, model=None) -> np.ndarray:
    """Combined distance (sf.fuse: DTW plus the sign embedding when a model is given)."""
    D, E = raw_distances(templates, seed, model)
    return sf.fuse(D, E) if model else D


def held_out(Ds: list[np.ndarray], words: np.ndarray, vocab: set) -> tuple[dict, list[float]]:
    """Per-word [correct, tested] over all jitter draws, matching only within `vocab`."""
    idx = np.where(np.isin(words, list(vocab)))[0]
    per_word = collections.defaultdict(lambda: [0, 0])
    right = []
    for D in Ds:
        sub = D[np.ix_(idx, idx)]
        nearest = np.argmin(sub, axis=1)
        for row, j in enumerate(nearest):
            w = words[idx[row]]
            per_word[w][1] += 1
            if words[idx[j]] == w:
                per_word[w][0] += 1
                right.append(float(sub[row, j]))
    return per_word, right


def held_window(frames: list[dict]) -> tuple[int, int] | None:
    """Longest run of frames where the (smoothed) wrist stays still -> (first, last) or None."""
    best, start, last = None, None, None
    for i, fr in enumerate(sf.smooth_hands(frames)):
        still = False
        if fr["hands"]:
            lm, a = fr["hands"][0]["lm"], fr["aspect"]
            w = (lm[0][0] * a, lm[0][1])
            palm = math.dist(w, (lm[9][0] * a, lm[9][1])) or 1e-6
            still = last is not None and math.dist(w, last) / palm < STATIC_STILL
            last = w
        else:
            last = None
        if still:
            start = i if start is None else start
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
        else:
            start = None
    return best if best and best[1] - best[0] + 1 >= STATIC_HOLD else None


def load_static(lmdir: Path, words: set) -> list[dict]:
    """Held-shape examples: the still part of each clip of `words` (clips without one are skipped)."""
    out = []
    for f in sorted(lmdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["word"] not in words:
            continue
        win = held_window(d["frames"])
        if not win:
            continue
        held = d["frames"][win[0]:win[1] + 1]
        # dominant hand = the one that moves most over the whole sign (both hands are still while held)
        vec = sf.static_vector(held, sf.dominant_label(d["frames"]))
        if vec:
            out.append({"word": d["word"], "name": d["name"], "vec": vec, "held": held})
    return out


def static_distances(items: list[dict], seed: int) -> np.ndarray:
    """D[i, j] = distance from a jittered copy of held shape i (read either way round) to example j."""
    rng = random.Random(seed)
    D = np.full((len(items), len(items)), np.inf)
    for i, it in enumerate(items):
        variants = sf.static_variants(jitter(it["held"], rng)) or [it["vec"]]
        for j, other in enumerate(items):
            if i != j:
                D[i, j] = min(sf.frame_cost(v, other["vec"]) for v in variants)
    return D


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--media-url", default="/media/signs/")
    p.add_argument("--vocab", type=Path, required=True, help="words recognised on Hand -> Speech (vocabulary.txt)")
    p.add_argument("--static-vocab", type=Path, help="words also recognised from a held hand shape (no movement)")
    p.add_argument("--embedding", type=Path, help="sign_embedding.json from train_embedding.py (combined with DTW)")
    p.add_argument("--emb-weight", type=float, default=1.0, help="weight of the embedding against DTW")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(a.meta.read_text(encoding="utf-8"))
    by_name = {v["name"]: v for v in meta["videos"]}

    # Speech -> Sign: every trimmed clip, grouped by word
    clips = collections.defaultdict(list)
    alternatives = {}
    # word = folder name; the client's own recordings ("own_*.mp4", see add_recordings.py) are listed first
    for f in sorted(a.clips.glob("*/*.mp4"), key=lambda f: (f.parent.name, not f.name.startswith("own_"), f.name)):
        word = f.parent.name
        clips[word].append(a.media_url + f"{word}/{f.name}")
        v = by_name.get(f.name)
        if not v:
            continue
        for alt in v.get("alternative_words") or []:
            alt = str(alt).strip().lower()
            if alt and alt != word:
                alternatives.setdefault(alt, word)
    clip_index = {
        "generated": date.today().isoformat(),
        "source": "Own recordings (own_*) and BSLDict (University of Oxford VGG, CC BY 4.0; video copyright "
                  "remains with the original owners)",
        "words": {w: {"label": label(w), "clips": c} for w, c in sorted(clips.items())},
        "alternatives": {k: v for k, v in sorted(alternatives.items()) if v in clips},
    }
    (a.out / "sign_clips.json").write_text(json.dumps(clip_index, indent=0), encoding="utf-8")
    print(f"sign_clips.json: {len(clips)} words, {sum(map(len, clips.values()))} clips")

    # Sign -> Speech: examples of the vocabulary words
    fixed = {w for line in a.vocab.read_text(encoding="utf-8").splitlines()
             if not line.lstrip().startswith("#") for w in line.split()}
    templates = [t for t in load_templates(a.landmarks) if t["word"] in fixed]
    words = np.array([t["word"] for t in templates])
    model = None
    if a.embedding:
        model = json.loads(a.embedding.read_text(encoding="utf-8"))
        for t in templates:   # the browser sees the templates rounded to 4 decimals
            t["seq"] = [[round(x, 4) for x in fr] for fr in t["seq"]]
        D0, E0 = raw_distances(templates, 0, model)
        # each part divided by its typical value (median), so the weight means the same for any data
        sf.DTW_SCALE = round(float(np.median(D0[np.isfinite(D0)])), 4)
        sf.EMB_SCALE = round(float(np.median(E0[np.isfinite(E0)])), 4)
        sf.EMB_WEIGHT = a.emb_weight
        print(f"embedding: weight {sf.EMB_WEIGHT}, scales dtw {sf.DTW_SCALE} / embedding {sf.EMB_SCALE}")
    select_Ds = [distance_matrix(templates, seed, model) for seed in range(SELECT_DRAWS)]
    vocab = {w for w in fixed if w in set(words)}
    _, right = held_out(select_Ds, words, vocab)
    if fixed - vocab:
        print("warning: no clips for", sorted(fixed - vocab))
    # Report accuracy on fresh jitter draws (not the ones that set the threshold).
    report_Ds = [distance_matrix(templates, 100 + seed, model) for seed in range(REPORT_DRAWS)]
    per_word, _ = held_out(report_Ds, words, vocab) if vocab else ({}, [])
    tested = sum(n for _, n in per_word.values())
    correct = sum(ok for ok, _ in per_word.values())
    threshold = round(float(np.quantile(right, 0.9)), 4) if right else 1.0

    signs = collections.defaultdict(list)
    for t in templates:
        if t["word"] in vocab:
            signs[t["word"]].append([[round(x, 4) for x in fr] for fr in t["seq"]])
    template_index = {
        "generated": date.today().isoformat(),
        "seqLen": sf.SEQ_LEN, "frameDim": sf.FRAME_DIM, "threshold": threshold, "params": sf.params(),
        "signs": [{"word": w, "label": label(w), "templates": s} for w, s in sorted(signs.items())],
    }
    if model:
        template_index["embedding"] = model

    # Held-shape words: recognised from the hand shape, orientation and position alone, while held still
    static_report = None
    if a.static_vocab:
        static_words = {w for line in a.static_vocab.read_text(encoding="utf-8").splitlines()
                        if not line.lstrip().startswith("#") for w in line.split()}
        items = load_static(a.landmarks, static_words)
        sw = np.array([it["word"] for it in items])
        found = {w for w in static_words if (sw == w).sum() >= MIN_CLIPS}
        if static_words - found:
            print("warning: fewer than 2 held clips for", sorted(static_words - found))
        _, s_right = held_out([static_distances(items, seed) for seed in range(SELECT_DRAWS)], sw, found)
        s_per, _ = held_out([static_distances(items, 100 + seed) for seed in range(REPORT_DRAWS)], sw, found)
        s_threshold = round(float(np.quantile(s_right, 0.9)), 4) if s_right else 1.0
        by_word = collections.defaultdict(list)
        for it in items:
            if it["word"] in found:
                by_word[it["word"]].append([round(x, 4) for x in it["vec"]])
        demo_of = {w: g for g, ws in DEMO_SHAPES.items() for w in ws if w not in static_words}
        by_demo = collections.defaultdict(list)
        for it in load_static(a.landmarks, set(demo_of)):
            by_demo[demo_of[it["word"]]].append([round(x, 4) for x in it["vec"]])
        template_index["static"] = {
            "threshold": s_threshold,
            "signs": [{"word": w, "label": label(w), "templates": v,
                       # two-handed if most examples show the second hand
                       "twoHanded": sum(t[sf.SLOT_DIM] >= 0.5 for t in v) * 2 > len(v)}
                      for w, v in sorted(by_word.items())],
            "demo": [{"word": g, "label": g, "templates": v} for g, v in sorted(by_demo.items())],
        }
        s_tested = sum(n for _, n in s_per.values())
        s_correct = sum(ok for ok, _ in s_per.values())
        static_report = {
            "method": f"Still part of each clip (wrist still for at least {STATIC_HOLD} frames) averaged into one "
                      "hand-shape example; each one, jittered, matched against all other examples of the held-shape "
                      f"words only (never itself), {REPORT_DRAWS} fresh jitter draws.",
            "examples": len(items), "vocabulary_size": len(found),
            "accuracy_within_vocabulary": round(s_correct / s_tested, 3) if s_tested else None,
            "accept_threshold": s_threshold,
            "vocabulary": {w: {"correct": s_per[w][0], "tested": s_per[w][1]} for w in sorted(found)},
        }
        print(f"held shapes: {len(items)} examples | {len(found)} words "
              f"(held-out top-1 {static_report['accuracy_within_vocabulary']}) | threshold {s_threshold} | "
              f"demo-shape competitors: " + ", ".join(f"{g} {len(v)}" for g, v in sorted(by_demo.items())))

    (a.out / "sign_templates.json").write_text(json.dumps(template_index, separators=(",", ":")), encoding="utf-8")

    report = {
        "generated": date.today().isoformat(),
        "method": f"Vocabulary from {a.vocab.name}. "
                  f"Each clip, with {TEST_NOISE} landmark jitter like live input, is matched against all other "
                  f"clips (never itself); accuracy_within_vocabulary is measured on {REPORT_DRAWS} fresh jitter draws. "
                  "No personal calibration (see scripts/test_calibration.mjs for that).",
        "clips": len(templates),
        "vocabulary_size": len(vocab),
        "accuracy_within_vocabulary": round(correct / tested, 3) if tested else None,
        "accept_threshold": threshold,
        "params": sf.params(),
        "vocabulary": {w: {"correct": per_word[w][0], "tested": per_word[w][1]} for w in sorted(vocab)},
        "held_shapes": static_report,
    }
    (a.out / "eval_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"templates: {len(templates)} | vocabulary: {len(vocab)} words "
          f"(held-out top-1 {report['accuracy_within_vocabulary']}) | threshold {threshold}")
    print("vocabulary:", " ".join(sorted(vocab)))


if __name__ == "__main__":
    main()
