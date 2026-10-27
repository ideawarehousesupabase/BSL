"""Merge several sign_embedding.json files (from train_embedding.py --folds 0) into one ensemble file.

The browser averages the networks' distances, which is more accurate than any single network
(tested on unseen words: 4 different network types 52.4% vs the best single one 47.7%).

Usage:
    python merge_embeddings.py a/sign_embedding.json b/sign_embedding.json ... --out sign_embedding.json
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    parts = [json.loads(f.read_text(encoding="utf-8")) for f in a.files]
    merged = {"input": parts[0]["input"], "models": [m for part in parts for m in part["models"]]}
    a.out.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")
    print(f"{len(merged['models'])} networks -> {a.out}")


if __name__ == "__main__":
    main()
