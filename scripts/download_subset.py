"""Download the BSLDict videos for the words in a word list.

Usage:
    python download_subset.py --meta bsldict_meta.json --words wordlist_candidates.txt --out <raw_video_dir>

`bsldict_meta.json` is produced from BSLDict's bsldict_v1.pkl (see README_DATA.md).
Hosted videos are fetched over HTTPS; YouTube-hosted ones via yt-dlp.
Already-downloaded files are skipped, so the script can be re-run safely.
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = "Mozilla/5.0 (GestureFlow MVP data pipeline)"


def read_words(path: Path) -> list[str]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.extend(line.split())
    return words


def fetch_hosted(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    if tmp.stat().st_size < 1024:
        tmp.unlink()
        raise RuntimeError("file too small")
    tmp.replace(dest)


def fetch_youtube(video_id: str, dest: Path) -> None:
    ytdlp = shutil.which("yt-dlp") or str(Path(sys.executable).with_name("yt-dlp"))
    cmd = [ytdlp, "-q", "--no-warnings", "-f", "mp4[height<=480]/mp4/best[height<=480]",
           "-o", str(dest), f"https://www.youtube.com/watch?v={video_id}"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    if not dest.exists():
        raise RuntimeError("yt-dlp produced no file")


def download(video: dict, out: Path) -> tuple[str, str | None]:
    dest = out / video["name"]
    if dest.exists():
        return video["name"], None
    try:
        if video["download_method_db"] == "youtube-dl":
            fetch_youtube(video["youtube_identifier_db"], dest)
        else:
            fetch_hosted(video["video_link_db"], dest)
        return video["name"], None
    except Exception as e:  # noqa: BLE001 - log and continue; dead links are expected
        return video["name"], str(e)[:200]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()

    meta = json.loads(a.meta.read_text(encoding="utf-8"))
    wanted = set(read_words(a.words))
    videos = [v for v in meta["videos"] if v["word"] in wanted]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} videos for {len(wanted)} words")

    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(download, v, a.out) for v in videos]
        for i, fut in enumerate(as_completed(futures), 1):
            name, err = fut.result()
            if err:
                failed.append(f"{name}\t{err}")
            if i % 50 == 0:
                print(f"  {i}/{len(videos)} done, {len(failed)} failed", flush=True)

    (a.out / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"finished: {len(videos) - len(failed)} ok, {len(failed)} failed (see failed.txt)")


if __name__ == "__main__":
    main()
