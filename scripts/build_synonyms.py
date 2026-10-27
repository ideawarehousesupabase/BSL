"""Map English words without a BSL clip onto words that have one, using WordNet.

  synonyms:   physician -> doctor, infirmary -> hospital (shared common meaning only)
  irregulars: went -> go, children -> child (WordNet exception lists)

Usage:
    python build_synonyms.py --clips-index <project>/demo/static/demo/data/sign_clips.json \
        --out <project>/demo/static/demo/data/synonyms.json --nltk-data <dir>

Needs the NLTK WordNet corpus: python -c "import nltk; nltk.download('wordnet', download_dir='<dir>')"
"""
import argparse
import json
from pathlib import Path

# Words that are usually names or slang in speech and must never be swapped for a sign.
BLOCK = {"john"}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips-index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nltk-data", required=True)
    a = p.parse_args()

    import nltk
    nltk.data.path.insert(0, a.nltk_data)
    from nltk.corpus import wordnet as wn

    have = set(json.loads(a.clips_index.read_text(encoding="utf-8"))["words"])
    norm = lambda s: s.lower().replace("_", "-")  # noqa: E731

    # For each word we have a clip for, take its most-used meaning and borrow the other
    # words for that meaning, but only ones that are actually used (count >= 1) and are not
    # more common than our word. This keeps physician -> doctor while rejecting rare or
    # misleading senses (worry -> headache, slang senses of "come"). The shared meaning must
    # also be the other word's own main meaning.
    def main_sense(word):
        """The meaning in which this word is used most (WordNet usage counts)."""
        best = None
        for syn in wn.synsets(word.replace("-", "_")):
            c = next((l.count() for l in syn.lemmas() if norm(l.name()) == word), 0)
            if c >= 1 and (best is None or c > best[1]):
                best = (syn, c)
        return best

    synonyms, strength = {}, {}
    for w in have:
        if len(w) == 1 or w.isdigit():
            continue  # letters and digits are only used for fingerspelling / numbers
        best = main_sense(w)
        if not best:
            continue
        syn, c = best
        for lem in syn.lemmas():
            src = norm(lem.name())
            if src == w or src in have or src in BLOCK or not src.isalpha() or len(src) < 3:
                continue
            src_main = main_sense(src)
            if src_main is None or src_main[0] != syn:
                continue  # e.g. "take" mainly means something else, so never show it as "need"
            if 1 <= lem.count() <= c and c > strength.get(src, 0):
                synonyms[src], strength[src] = w, c

    irregular = {}
    for pos_map in wn._exception_map.values():  # e.g. went -> [go], children -> [child]
        for form, bases in pos_map.items():
            for base in bases:
                if norm(base) in have and norm(form) not in have:
                    irregular[norm(form)] = norm(base)
                    break

    out = {"source": "Princeton WordNet 3.0 (WordNet licence)", "synonyms": dict(sorted(synonyms.items())),
           "irregular": dict(sorted(irregular.items()))}
    a.out.write_text(json.dumps(out, indent=0), encoding="utf-8")
    print(f"synonyms: {len(synonyms)}, irregular forms: {len(irregular)}")


if __name__ == "__main__":
    main()
