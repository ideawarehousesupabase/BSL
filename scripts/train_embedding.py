"""Train a small "sign embedding" network on the dictionary landmarks (offline, CPU, a few minutes).

Why: with only 2-6 dictionary clips per word, a classifier for 40 words cannot be trained directly.
Instead the network is trained on ~1,000 OTHER dictionary words to map a sign (12 frames x 53 hand/body
features, the same input as sign_matcher.js) to 64 numbers, such that clips of the same word by
different signers come out close together and different words far apart. New words are then
recognised by comparing with their dictionary examples in this space (nearest example), exactly like
the DTW matcher, so no clips of the Hand -> Speech words are needed for training.

The browser runs it as plain JavaScript (sign_matcher.js `embed`): three small matrix multiplications.

Evaluation is honest: the reported accuracy for a word always comes from a network that never saw that
word (k-fold over words), compared with DTW on the same test.

Usage:
    python train_embedding.py --landmarks <pool landmark dir> --words wordlist_candidates.txt \
        --out-dir <work dir> [--folds 5] [--exclude vocabulary.txt]   (final model: --folds 0)
"""
import argparse
import base64
import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_index as bi  # noqa: E402
import sign_features as sf  # noqa: E402

COPIES = 6          # jittered + time-cropped copies of each clip computed up front
EMB = 64
HIDDEN = (512, 256)
EPOCHS = 60
SCALE = 16.0        # cosine-softmax temperature
SD = sf.SLOT_DIM
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # an NVIDIA GPU (CUDA build of PyTorch) trains ~5x faster
WEIGHT_DECAY = 1e-3


def load(lmdir: Path, pool: set) -> list[dict]:
    out = []
    for f in sorted(lmdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["word"] in pool:
            a, b = d["core"]
            out.append({"word": d["word"], "name": d["name"], "frames": d["frames"][a:b + 1]})
    return out


def copies(frames: list, rng: random.Random, n: int) -> list:
    """n augmented feature sequences (both dominance readings): jitter + random trim of start/end."""
    seqs = []
    for k in range(n):
        fr = frames
        if k and len(fr) > 8:
            cut = len(fr)
            a = rng.randint(0, int(0.12 * cut))
            b = cut - rng.randint(0, int(0.12 * cut))
            fr = fr[a:b]
        fr = bi.jitter(fr, rng) if k else fr
        seqs += sf.query_variants(fr)
    return seqs


def feature_aug(x: torch.Tensor) -> torch.Tensor:
    """Signer variation on the features: rotate the hand in the image plane, shift its position, bend noise."""
    x = x.clone()
    B = x.shape[0]
    th = (torch.rand(B, 1, device=x.device) - 0.5) * math.radians(24)
    c, s = torch.cos(th), torch.sin(th)
    for off in (0, SD):
        pres = x[:, :, off:off + 1]
        o = off + 19
        for i, j in ((o, o + 1), (o + 2, o + 3)):          # pointing (x, y) and palm normal (x, y)
            xi, xj = x[:, :, i].clone(), x[:, :, j].clone()
            x[:, :, i] = c * xi - s * xj
            x[:, :, j] = s * xi + c * xj
        x[:, :, off + 24:off + 26] += (torch.rand(B, 1, 2, device=x.device) - 0.5) * 0.4 * pres   # location shift
        x[:, :, off + 1:off + 19] += torch.randn(B, 1, 18, device=x.device) * 0.02 * pres         # handshape noise
    return x


class Net(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        dims = [sf.SEQ_LEN * sf.FRAME_DIM, *HIDDEN, EMB]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.drop = nn.Dropout(DROPOUT)
        self.classes = nn.Parameter(torch.randn(n_classes, EMB) * 0.1)

    def embed(self, x):
        h = x.flatten(1)
        for k, layer in enumerate(self.layers):
            h = layer(h)
            if k < len(self.layers) - 1:
                h = self.drop(F.relu(h))
        return F.normalize(h, dim=1)

    def forward(self, x):
        return SCALE * self.embed(x) @ F.normalize(self.classes, dim=1).T


def train(clips: list[dict], seed: int) -> Net:
    torch.manual_seed(seed)
    words = sorted({c["word"] for c in clips})
    idx = {w: i for i, w in enumerate(words)}
    X = torch.tensor(np.array([s for c in clips for s in c["copies"]]), dtype=torch.float32, device=DEVICE)
    Y = torch.tensor([idx[c["word"]] for c in clips for _ in c["copies"]], device=DEVICE)
    net = Net(len(words)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(X), device=DEVICE)
        for k in range(0, len(X), 256):
            b = perm[k:k + 256]
            loss = F.cross_entropy(net(feature_aug(X[b])), Y[b], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return net.cpu()


def embed_all(net: Net, seqs: list) -> np.ndarray:
    with torch.no_grad():
        return net.embed(torch.tensor(np.array(seqs), dtype=torch.float32)).numpy()


def pack(net: Net) -> dict:
    """One network for sign_embedding.json: each weight matrix as 8-bit numbers (base64) times a scale,
    which keeps the download small (about 0.3 MB per network); biases as plain numbers."""
    layers = []
    for l in net.layers:
        w = l.weight.detach().numpy().astype(np.float64)
        scale = float(np.abs(w).max()) / 127 or 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        layers.append({"shape": list(w.shape), "scale": scale, "q": base64.b64encode(q.tobytes()).decode("ascii"),
                       "b": np.round(l.bias.detach().numpy().astype(np.float64), 5).tolist()})
    return {"layers": layers}


def prepare(clips: list[dict], cache: Path, n_copies: int) -> list[dict]:
    """Adds the augmented feature copies ("copies") and the clean readings ("clean") to each clip.
    They take a while to compute, so they are cached (cache file next to the landmarks)."""
    cached = dict(np.load(cache, allow_pickle=False)) if cache.exists() else {}
    rng = random.Random(0)
    for c in clips:
        key = c["name"].replace(".", "_")
        if key + "__copies" in cached:
            c["copies"] = cached[key + "__copies"].tolist()
            c["clean"] = cached[key + "__clean"].tolist()
        else:
            c["copies"] = copies(c["frames"], rng, n_copies)
            c["clean"] = sf.query_variants(c["frames"])
            if c["copies"] and c["clean"]:
                cached[key + "__copies"] = np.array(c["copies"], dtype=np.float32)
                cached[key + "__clean"] = np.array(c["clean"], dtype=np.float32)
    np.savez(cache, **cached)
    return [c for c in clips if c["copies"] and c["clean"]]


def main() -> None:
    global HIDDEN, EPOCHS, DROPOUT, WEIGHT_DECAY
    p = argparse.ArgumentParser()
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--exclude", type=Path, help="words never used for training (the Hand -> Speech vocabulary)")
    p.add_argument("--hidden", default="256,128", help="hidden layer sizes")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--copies", type=int, default=COPIES)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=0, help="changes the network's random start (for ensembles)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--train-extra", type=Path,
                   help="landmarks of more dictionary words used ONLY for training (never tested or recognised)")
    p.add_argument("--ensemble", type=int, default=1, help="final model: number of networks (different seeds)")
    a = p.parse_args()
    HIDDEN = tuple(int(x) for x in a.hidden.split(","))
    EPOCHS, DROPOUT, WEIGHT_DECAY = a.epochs, a.dropout, a.weight_decay
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pool = {w for line in a.words.read_text(encoding="utf-8").splitlines() if not line.startswith("#") for w in line.split()}
    clips = prepare(load(a.landmarks, pool), a.landmarks.parent / f"embedding_features_c{a.copies}.npz", a.copies)
    count = collections.Counter(c["word"] for c in clips)
    extra = []
    if a.train_extra:
        everything = {json.loads(f.read_text(encoding="utf-8"))["word"] for f in a.train_extra.glob("*.json")}
        extra = prepare(load(a.train_extra, everything - set(count)),
                        a.train_extra.parent / f"embedding_features_c{a.copies}.npz", a.copies)
    print(f"{len(clips)} clips, {len(count)} words (+ {len(extra)} training-only clips)")

    if a.folds == 0:   # final model, trained on every word except the excluded vocabulary
        excl = set()
        if a.exclude:
            excl = {w for line in a.exclude.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("#") for w in line.split()}
        models = []
        for k in range(a.ensemble):
            models.append(pack(train([c for c in clips if c["word"] not in excl] + extra, a.seed + k)))
            print(f"  network {k + 1}/{a.ensemble} trained", flush=True)
        (a.out_dir / "sign_embedding.json").write_text(json.dumps(
            {"input": [sf.SEQ_LEN, sf.FRAME_DIM], "models": models}, separators=(",", ":")), encoding="utf-8")
        print("saved", a.out_dir / "sign_embedding.json", "(excluded", len(excl), "words)")
        return

    # k-fold over words: each word's clips are embedded by a network that never saw that word
    words = sorted(count)
    random.Random(1).shuffle(words)
    fold_of = {w: k % a.folds for k, w in enumerate(words)}
    test_rng = random.Random(7)
    out = {}   # clip name -> {"q": [variants of jittered query], "ref": [variants of clean clip]}
    for k in range(a.folds):
        net = train([c for c in clips if fold_of[c["word"]] != k] + extra, k + 10 * a.seed)
        for c in clips:
            if fold_of[c["word"]] == k:
                q = sf.query_variants(bi.jitter(c["frames"], test_rng)) or c["clean"]
                out[c["name"]] = {"word": c["word"], "q": embed_all(net, q).tolist(), "ref": embed_all(net, c["clean"]).tolist()}
        print(f"  fold {k + 1}/{a.folds} done", flush=True)
    (a.out_dir / "oof_embeddings.json").write_text(json.dumps(out), encoding="utf-8")
    print("saved out-of-fold embeddings for", len(out), "clips")


if __name__ == "__main__":
    main()
