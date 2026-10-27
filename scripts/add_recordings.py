"""Add the client's own sign recordings (best way to improve accuracy and to replace third-party clips).

Folder layout (one folder per word, BSLDict spelling, e.g. thank-you):
    recordings/
        doctor/  take1.mp4  take2.mov ...
        name/    ...

For every video this script:
  1. converts it to 25 fps / 360p,
  2. runs MediaPipe hand + body tracking and finds where the sign starts and stops (hand movement),
  3. saves the trimmed sign as media/signs/<word>/own_<file>.mp4 (played first on Speech -> Sign),
  4. writes landmarks to <landmark_dir>/own_<word>_<file>.json (examples for Hand -> Speech).
Then re-run build_index.py (with --vocab) and build_synonyms.py.

Recording tips: plain background, good light, waist-up, start and end with hands down,
one sign per video, 5-10 takes per sign from 2-5 signers.

Usage:
    python add_recordings.py --recordings <dir> --clips <project>/media/signs \
        --landmarks <landmark_dir> --models <model_dir>
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_clips as pc  # noqa: E402

VIDEO_TYPES = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}
MOVING = 0.04   # wrist movement per frame (palm widths) that counts as signing; same as the live page
PAD = 3


def motion_core(frames: list[dict]) -> tuple[int, int] | None:
    """First and last frame where the hand is clearly moving (like BSLDict's temp_interval)."""
    moving, last = [], None
    for i, fr in enumerate(frames):
        if not fr["hands"]:
            last = None
            continue
        lm, a = fr["hands"][0]["lm"], fr["aspect"]
        w = (lm[0][0] * a, lm[0][1])
        palm = math.dist(w, (lm[9][0] * a, lm[9][1])) or 1e-6
        if last is not None and math.dist(w, last) / palm > MOVING:
            moving.append(i)
        last = w
    with_hands = [i for i, fr in enumerate(frames) if fr["hands"]]
    if len(with_hands) < 3:
        return None
    if len(moving) < 2:
        return with_hands[0], with_hands[-1]
    return moving[0], moving[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    a = p.parse_args()
    pc._init(str(a.models))
    a.landmarks.mkdir(parents=True, exist_ok=True)

    added, skipped = 0, []
    for src in sorted(f for f in a.recordings.glob("*/*") if f.suffix.lower() in VIDEO_TYPES):
        word = src.parent.name.strip().lower().replace(" ", "-")
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in src.stem)
        name = f"own_{stem}.mp4"
        out_json = a.landmarks / f"own_{word}_{stem}.json"
        out_clip = a.clips / word / name
        if out_json.exists() and out_clip.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / "full.mp4"
            pc.trim(src, full, 0, 10 ** 6)                 # 25 fps / 360p, whole video
            data = pc.landmarks(full)
            core = motion_core(data["frames"])
            if not core:
                skipped.append(f"{src} (hands not visible)")
                continue
            start, end = max(0, core[0] - PAD), min(len(data["frames"]) - 1, core[1] + PAD)
            pc.trim(full, out_clip, start, end)            # the sign only, for playback
        data.update(word=word, name=name, core=[core[0], core[1]], source="own recording")
        out_json.write_text(json.dumps(data), encoding="utf-8")
        added += 1
        print(f"  {word}: {src.name} -> sign frames {core[0]}-{core[1]}", flush=True)
    print(f"added {added} recordings" + (f", skipped {len(skipped)}: " + "; ".join(skipped) if skipped else ""))


if __name__ == "__main__":
    main()
