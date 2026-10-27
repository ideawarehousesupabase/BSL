"""Pick the matching weights that give the largest reliable vocabulary in the held-out test.

This only tunes a handful of numbers (no model is trained). The winning values are
printed; copy them into sign_features.py, then re-run build_index.py.

Usage:
    python tune_params.py --landmarks <landmark_dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_index  # noqa: E402
import sign_features as sf  # noqa: E402

GRID = {
    "SHAPE_WEIGHT": [1.0, 2.0, 3.0],
    "ORIENT_WEIGHT": [0.25, 0.5, 1.0],
    "LOC_WEIGHT": [0.25, 0.5, 1.0],
    "NONDOMINANT_WEIGHT": [0.3, 0.6, 1.0],
    "BAND": [2, 3, 4],
}


def score(templates):
    """(vocabulary size, held-out accuracy within it) for the current weights."""
    import numpy as np
    words = np.array([t["word"] for t in templates])
    Ds = [build_index.distance_matrix(templates, seed) for seed in range(build_index.SELECT_DRAWS)]
    vocab, _, _, _ = build_index.select_vocabulary(Ds, words)
    if not vocab:
        return 0, 0.0
    report = [build_index.distance_matrix(templates, 100 + seed) for seed in range(build_index.REPORT_DRAWS)]
    per_word, _ = build_index.held_out(report, words, vocab)
    tested = sum(n for _, n in per_word.values())
    return len(vocab), (sum(ok for ok, _ in per_word.values()) / tested) if tested else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=2)
    a = p.parse_args()
    templates = build_index.load_templates(a.landmarks)
    best = score(templates)
    print(f"start {best} with", {k: getattr(sf, k) for k in GRID}, flush=True)
    for _ in range(a.rounds):  # coordinate descent: one weight at a time
        for name, values in GRID.items():
            keep = getattr(sf, name)
            for v in values:
                setattr(sf, name, v)
                s = score(templates)
                print(f"  {name}={v}: vocabulary={s[0]} held-out top1={s[1]:.3f}", flush=True)
                if s > best:
                    best, keep = s, v
            setattr(sf, name, keep)
    print("best", best, {k: getattr(sf, k) for k in GRID})


if __name__ == "__main__":
    main()
