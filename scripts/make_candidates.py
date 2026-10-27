"""Write the list of BSLDict words worth testing for Hand -> Speech (scripts/wordlist_candidates.txt).

A candidate is an everyday English word or phrase with at least MIN_CLIPS dictionary clips (so it can be
tested against other signers). Specialist glossaries (ICT, science) and religious vocabulary are left out,
since the product is for healthcare, public services, education and retail conversations.
The words actually recognised are then chosen by build_index.py (the ones the matcher tells apart best).

Usage:
    python make_candidates.py --meta bsldict_meta.json --out wordlist_candidates.txt
Needs: pip install wordfreq (word frequencies; used only here, offline).
"""
import argparse
import collections
import json
from pathlib import Path

from wordfreq import zipf_frequency

MIN_CLIPS = 3
MIN_ZIPF = 3.5     # roughly: appears at least 3 times per million words of everyday English
GLOSSARIES = ("QIA Dictionary of ICT", "Scottish Sensory Centre")   # specialist subject glossaries
BLOCK = set("""god lord bible holy-spirit amen priest christian baptism diocese scriptures new-testament
old-testament bless miracle angel grace pray parish church sex kill die dead army war guilty illegal
legal law president government politics""".split())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    videos = json.loads(a.meta.read_text(encoding="utf-8"))["videos"]
    clips = collections.Counter(v["word"] for v in videos)
    sources = collections.defaultdict(collections.Counter)
    for v in videos:
        sources[v["word"]][(v.get("source_site_db") or "").strip(" -")] += 1
    words = []
    for w, n in clips.items():
        if n < MIN_CLIPS or not w.replace("-", "").isalpha() or w in BLOCK:
            continue
        glossary = sum(c for s, c in sources[w].items() if s in GLOSSARIES)
        if glossary * 2 > n:      # mostly from a specialist glossary
            continue
        f = zipf_frequency(w.replace("-", " "), "en")
        if f >= MIN_ZIPF:
            words.append((f, w))
    words.sort(reverse=True)
    a.out.write_text("# Candidate words for Hand -> Speech, most common first (made by make_candidates.py)\n"
                     + "\n".join(w for _, w in words) + "\n", encoding="utf-8")
    print(f"{len(words)} candidate words, {sum(clips[w] for _, w in words)} clips -> {a.out}")


if __name__ == "__main__":
    main()
