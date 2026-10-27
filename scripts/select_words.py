"""Choose the Hand -> Speech vocabulary from a large pool of dictionary words (no training).

Moving signs: only words with at least 4 dictionary clips are considered (several signers). Every clip
(with live-like jitter, read either hand as dominant) is compared with all other clips; its margin is how
much closer its nearest same-word example is than its nearest other-word example. Words with the smallest
average margin are removed (10% at a time, then one by one) until TARGET words remain. In tests on
locked-away clips this chose better words than counting right/wrong matches (71.7% vs 66.1%).
Held hand shapes: the word matched worst is removed one at a time. This is done separately for
  - moving signs (dynamic time warping over the whole sign) -> vocabulary.txt
  - held hand shapes (the still moment of the sign)          -> static_vocabulary.txt
    (the client's demo hand shapes take part as competitors that are never removed, so words that look
    like Peace / Four / Stop ... are dropped)

Distance: DTW, optionally combined with the sign embedding (train_embedding.py). The embedding
distances used here are out-of-fold: each clip was embedded by a network that never saw its word.
Several weights are compared on random 40-word vocabularies and the best one is used.

Honest test (--holdout): in each of 4 rounds, one clip of every word is locked away, the words are chosen
without it, and only the locked clips are tested (against the other clips). This figure is not flattered
by the choice; the final words are then chosen from all clips.

Speed: a clip is compared with all others by quick costs first (frame-by-frame, and the embedding); the
exact DTW distance is computed only for the closest SHORTLIST examples plus the word's own other examples.

Usage:
    python select_words.py --landmarks <pool landmark dir> --words wordlist_candidates.txt \
        --out-moving vocabulary.txt --out-static static_vocabulary.txt [--target 40]
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_index as bi  # noqa: E402
import sign_features as sf  # noqa: E402

SHORTLIST = 80
DRAWS = 2
MIN_TEST = 3          # a moving word needs this many clips (so each is tested against 2+ others)
STATIC_BAR = 0.75


def raw_lists(seqs: np.ndarray, frames: list, seed: int, same: list, E: np.ndarray | None) -> list[list]:
    """For each clip i: [(j, dtw, emb)] for its shortlist (never itself)."""
    rng = random.Random(seed)
    out = []
    for i, fr in enumerate(frames):
        variants = sf.query_variants(bi.jitter(fr, rng)) or [seqs[i].tolist()]
        best = {}
        for v in variants:
            q = np.array(v, dtype=float)
            a, b = q[None], seqs
            both = (a[..., -1] >= 0.5) & (b[..., -1] >= 0.5)
            quick = (sf._slot_cost_np(a, b, 0, both) + sf.NONDOMINANT_WEIGHT * sf._slot_cost_np(a, b, sf.SLOT_DIM, both)).mean(1)
            quick[i] = np.inf
            cand = set(np.argsort(quick)[:SHORTLIST].tolist()) | set(same[i])
            if E is not None:
                cand |= set(np.argsort(E[i])[:SHORTLIST].tolist())
            cand.discard(i)
            cand = sorted(cand)
            d = sf.dtw_batch(q, seqs[cand])
            for j, dist in zip(cand, d):
                best[j] = min(best.get(j, np.inf), float(dist))
        out.append([(j, d, float(E[i, j]) if E is not None else 0.0) for j, d in best.items()])
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(frames)}", flush=True)
    return out


def fused_lists(raw: list[list], w: float, dscale: float, escale: float) -> list[list]:
    """Sorted [(distance, j)] with distance = dtw/dscale + w * emb/escale (w = inf: embedding only)."""
    if w == float("inf"):
        return [sorted((e, j) for j, _, e in r) for r in raw]
    return [sorted((d / dscale + w * e / escale, j) for j, d, e in r) for r in raw]


def oof_matrix(paths: str, names: list) -> np.ndarray:
    """Average over the networks (comma-separated oof files) of oof_matrix_one."""
    files = paths.split(",")
    return sum(oof_matrix_one(Path(f), names) for f in files) / len(files)


def oof_matrix_one(path: Path, names: list) -> np.ndarray:
    """E[i, j] = 1 - best cosine similarity between clip i (jittered, both readings) and clip j."""
    oof = json.loads(path.read_text(encoding="utf-8"))
    Q = np.array([np.pad(np.array(oof[n]["q"]), ((0, 2 - len(oof[n]["q"])), (0, 0)), mode="edge") for n in names])
    R = np.array([np.pad(np.array(oof[n]["ref"]), ((0, 2 - len(oof[n]["ref"])), (0, 0)), mode="edge") for n in names])
    E = np.full((len(names), len(names)), np.inf, dtype=np.float32)
    for k in range(0, len(names), 256):
        sims = np.einsum("iad,jbd->ijab", Q[k:k + 256], R).reshape(len(Q[k:k + 256]), len(names), -1).max(-1)
        E[k:k + 256] = 1 - sims
    np.fill_diagonal(E, np.inf)
    return E


def static_lists(items: list, n_test: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    vecs = np.array([it["vec"] for it in items])
    out = []
    for i in range(n_test):
        variants = sf.static_variants(bi.jitter(items[i]["held"], rng)) or [items[i]["vec"]]
        a = np.array(variants)[:, None, :]
        both = (a[..., -1] >= 0.5) & (vecs[None, :, -1] >= 0.5)
        d = (sf._slot_cost_np(a, vecs[None], 0, both) + sf.NONDOMINANT_WEIGHT * sf._slot_cost_np(a, vecs[None], sf.SLOT_DIM, both)).min(0)
        d[i] = np.inf
        out.append(sorted((float(x), j) for j, x in enumerate(d) if np.isfinite(x)))
    return out


def per_word(lists: list[list[dict]], words: list, vocab: set, keep_always: set = frozenset()) -> dict:
    """[correct, tested] per word: nearest example among words in vocab (plus keep_always)."""
    res = collections.defaultdict(lambda: [0, 0])
    for draw in lists:
        for i, near in enumerate(draw):
            w = words[i]
            if w not in vocab:
                continue
            res[w][1] += 1
            # near is sorted by distance: the first example of an allowed word is the match
            hit = next((words[j] for _, j in near if words[j] in vocab or words[j] in keep_always), None)
            if hit == w:
                res[w][0] += 1
    return res


def locked_accuracy(lists, words, sel_words, locked, vocab, keep_always=frozenset()):
    """[correct, tested] per word, for the locked-away clips only (matched against the unlocked examples)."""
    res = collections.defaultdict(lambda: [0, 0])
    for draw in lists:
        for i in locked:
            w = words[i]
            if w not in vocab:
                continue
            res[w][1] += 1
            hit = next((sel_words[j] for _, j in draw[i] if sel_words[j] in vocab or sel_words[j] in keep_always), None)
            if hit == w:
                res[w][0] += 1
    return res


def lock(words: list, need: int) -> tuple[set, list]:
    """Lock away the last clip of every word that has more than `need` clips."""
    by_word = collections.defaultdict(list)
    for i, w in enumerate(words):
        by_word[w].append(i)
    locked = {idx[-1] for idx in by_word.values() if len(idx) > need}
    return locked, [("LOCKED:" + w) if i in locked else w for i, w in enumerate(words)]


def margins_choose(lists, words, byw, candidates, target, locked=frozenset()):
    """Remove the words with the smallest average margin until `target` remain."""
    vocab = set(candidates)
    while len(vocab) > target:
        score = {}
        for w in vocab:
            vals = []
            for i in byw[w]:
                if i in locked:
                    continue
                for draw in lists:
                    s, o = nearest_pair(draw, i, words, vocab, locked)
                    vals.append((o - s) / max(o, 1e-9))
            score[w] = float(np.mean(vals))
        k = max(1, len(vocab) // 10) if len(vocab) > 150 else 1
        for w in sorted(vocab, key=lambda w: (score[w], w))[:k]:
            vocab.discard(w)
    return vocab


def nearest_pair(draw, i, words, vocab, locked):
    """Distances from clip i to its nearest same-word and nearest other-word example (unlocked, in vocab)."""
    s = o = float("inf")
    for d, j in draw[i]:
        if j in locked:
            continue
        if words[j] == words[i]:
            s = min(s, d)
        elif words[j] in vocab:
            o = min(o, d)
        if s < float("inf") and o < float("inf"):
            break
    return s, o


def greedy(lists, words, vocab, freq, target, bar=None, keep_always=frozenset()):
    vocab = set(vocab)
    curve = {}
    while len(vocab) > target:
        res = per_word(lists, words, vocab, keep_always)
        acc = {w: res[w][0] / res[w][1] for w in vocab}
        if bar is not None and min(acc.values()) >= bar:
            break
        worst = min(sorted(vocab), key=lambda w: (acc[w], freq.get(w, 0)))
        vocab.discard(worst)
        if len(vocab) in (150, 100, 80, 60, 50, 45, 40, 35, 30, 25, 20):
            r = per_word(lists, words, vocab, keep_always)
            curve[len(vocab)] = round(sum(c for c, _ in r.values()) / max(1, sum(n for _, n in r.values())), 3)
    return vocab, curve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out-moving", type=Path, required=True)
    p.add_argument("--out-static", type=Path, required=True)
    p.add_argument("--target", type=int, default=40)
    p.add_argument("--static-target", type=int, default=25)
    p.add_argument("--oof", help="oof_embeddings.json from train_embedding.py (k-fold); several (comma-separated) = ensemble")
    p.add_argument("--skip", type=Path, help="words never chosen (e.g. too technical for a demo)")
    p.add_argument("--holdout", action="store_true", help="lock one clip per word away for an honest test")
    a = p.parse_args()

    order = [w for line in a.words.read_text(encoding="utf-8").splitlines()
             if not line.startswith("#") for w in line.split()]
    freq = {w: len(order) - k for k, w in enumerate(order)}   # higher = more common
    pool = set(order)
    if a.skip:
        pool -= {w for line in a.skip.read_text(encoding="utf-8").splitlines()
                 if not line.startswith("#") for w in line.split()}

    # moving signs
    templates = [t for t in bi.load_templates(a.landmarks) if t["word"] in pool]
    if a.oof:
        have = set.intersection(*[set(json.loads(Path(f).read_text(encoding="utf-8"))) for f in a.oof.split(",")])
        templates = [t for t in templates if t["name"] in have]
    # only words with 4+ clips can be chosen, and other words never compete with them, so only those are compared
    per = collections.Counter(t["word"] for t in templates)
    templates = [t for t in templates if per[t["word"]] >= 4]
    words = [t["word"] for t in templates]
    count = collections.Counter(words)
    by_word = collections.defaultdict(list)
    for i, w in enumerate(words):
        by_word[w].append(i)
    same = [[j for j in by_word[w] if j != i] for i, w in enumerate(words)]
    locked, sel_words = lock(words, MIN_TEST) if a.holdout else (set(), words)
    testable = {w for w, n in collections.Counter(sel_words).items() if n >= MIN_TEST and not w.startswith("LOCKED:")}
    print(f"moving: {len(templates)} clips, {len(testable)} testable words")
    seqs = np.array([t["seq"] for t in templates], dtype=float)
    frames = [t["frames"] for t in templates]
    E = oof_matrix(a.oof, [t["name"] for t in templates]) if a.oof else None
    cache = a.out_moving.with_name("raw_lists_cache.json")
    if cache.exists():
        all_raw = json.loads(cache.read_text(encoding="utf-8"))
    else:   # draws 0, 1 to choose; 100, 101 (fresh jitter) to test
        all_raw = [raw_lists(seqs, frames, seed, same, E) for seed in (0, 1, 100, 101)]
        cache.write_text(json.dumps(all_raw), encoding="utf-8")
    sel_raw, rep_raw = all_raw[:2], all_raw[2:]
    dvals = [d for r in sel_raw[0] for _, d, _ in r]
    evals = [e for r in sel_raw[0] for _, _, e in r]
    dscale, escale = float(np.median(dvals)), float(np.median(evals)) if E is not None else 1.0
    best_w = 0.0
    if E is not None:   # compare weights on random 40-word vocabularies of testable words
        rng = random.Random(3)
        sets = [set(rng.sample(sorted(testable), 40)) for _ in range(60)]
        scores = {}
        for w in (0.0, 0.5, 1.0, 2.0, 4.0, float("inf")):
            lists = fused_lists(sel_raw[0], w, dscale, escale)
            ok = tot = 0
            for V in sets:
                r = per_word([lists], sel_words, V)
                ok += sum(c for c, _ in r.values())
                tot += sum(n for _, n in r.values())
            scores[w] = round(ok / tot, 3)
        best_w = max(scores, key=scores.get)
        print("  random 40-word vocabularies, accuracy by embedding weight (0 = DTW only, inf = embedding only):", scores)
    print(f"  using weight {best_w}")
    sel_lists = [fused_lists(r, best_w, dscale, escale) for r in sel_raw]
    rep_lists = [fused_lists(r, best_w, dscale, escale) for r in rep_raw]
    candidates = {w for w, idx in by_word.items() if len(idx) >= 4}
    curve = {}
    if a.holdout:
        ok = tot = 0
        for rnd in range(4):
            locked = {idx[rnd % len(idx)] for w, idx in by_word.items() if w in candidates}
            chosen = margins_choose(sel_lists, words, by_word, candidates, a.target, locked)
            for w in chosen:
                for i in by_word[w]:
                    if i in locked:
                        for draw in rep_lists:
                            s_, o_ = nearest_pair(draw, i, words, chosen, locked)
                            tot += 1
                            ok += s_ < o_
        curve = {"locked_clip_accuracy": round(ok / max(1, tot), 3), "tests": tot}
        print(f"  LOCKED-CLIP accuracy (honest, 4 rounds): {ok}/{tot} = {ok / max(1, tot):.3f}")
    moving = margins_choose(sel_lists, words, by_word, candidates, a.target)
    rep = per_word(rep_lists, words, moving)
    tot, ok = sum(n for _, n in rep.values()), sum(c for c, _ in rep.values())
    print(f"  chosen {len(moving)} moving words from {len(candidates)} with 4+ clips; "
          f"accuracy on their own clips {ok}/{tot} = {ok / max(1, tot):.3f} (flattered by the choice)")
    print("  ", " ".join(f"{w}:{rep[w][0]}/{rep[w][1]}" for w in sorted(moving, key=lambda w: -freq[w])))

    # held hand shapes (demo shapes are permanent competitors)
    demo_of = {w: g for g, ws in bi.DEMO_SHAPES.items() for w in ws}
    s_items = [it for it in bi.load_static(a.landmarks, pool - set(demo_of))]
    n_test = len(s_items)
    s_items += bi.load_static(a.landmarks, set(demo_of))
    s_words = [it["word"] if k < n_test else "DEMO:" + demo_of[it["word"]] for k, it in enumerate(s_items)]
    s_locked, s_sel_words = lock(s_words[:n_test], bi.MIN_CLIPS) if a.holdout else (set(), s_words[:n_test])
    s_words_all, s_words = s_words, s_sel_words + s_words[n_test:]
    s_testable = {w for w, n in collections.Counter(s_sel_words).items() if n >= bi.MIN_CLIPS and not w.startswith("LOCKED:")}
    competitors = {w for w in s_words[n_test:]}
    print(f"held: {n_test} still moments, {len(s_testable)} testable words, {len(competitors)} demo shapes")
    s_sel = [static_lists(s_items, n_test, seed) for seed in range(DRAWS)]
    held, s_curve = greedy(s_sel, s_words, s_testable, freq, a.static_target, bar=STATIC_BAR, keep_always=competitors)
    s_rep_lists = [static_lists(s_items, n_test, 100 + seed) for seed in range(DRAWS)]
    s_rep = per_word(s_rep_lists, s_words, held, competitors)
    tot, ok = sum(n for _, n in s_rep.values()), sum(c for c, _ in s_rep.values())
    print(f"  chosen {len(held)} held words, fresh-draw accuracy {ok}/{tot} = {ok / max(1, tot):.3f}", s_curve)
    if s_locked:
        h_locked = locked_accuracy(s_rep_lists, s_words_all, s_words, s_locked, held, competitors)
        lt, lo = sum(n for _, n in h_locked.values()), sum(c for c, _ in h_locked.values())
        print(f"  LOCKED-CLIP accuracy (honest): {lo}/{lt} = {lo / max(1, lt):.3f} over {len(h_locked)} words")
    print("  ", " ".join(f"{w}:{s_rep[w][0]}/{s_rep[w][1]}" for w in sorted(held, key=lambda w: -freq[w])))

    head = ("# Signs recognised on the Hand -> Speech page (BSLDict spelling), chosen by select_words.py from\n"
            "# the common dictionary words in wordlist_candidates.txt: the ones the matcher tells apart best.\n"
            "# build_index.py --vocab uses exactly these words. The 9 demo gestures are separate.\n")
    a.out_moving.write_text(head + " ".join(sorted(moving)) + "\n", encoding="utf-8")
    a.out_static.write_text(
        "# Signs also recognised from a HELD hand shape (no movement): make the hand shape at its place and\n"
        "# hold it still for about a second. Chosen by select_words.py (held shapes that look like a demo\n"
        "# gesture are left out). build_index.py --static-vocab uses exactly these words.\n"
        + " ".join(sorted(held)) + "\n", encoding="utf-8")
    json.dump({"moving": {w: rep[w] for w in moving}, "held": {w: s_rep[w] for w in held},
               "moving_curve": curve, "held_curve": s_curve, "emb_weight": best_w},
              open(a.out_moving.with_name("selection_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
