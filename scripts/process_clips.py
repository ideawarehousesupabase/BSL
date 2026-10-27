"""Trim downloaded BSLDict videos to the sign and extract hand/body landmarks.

For every downloaded video:
  1. ffmpeg: resample to 25 fps / 360p (BSLDict's `temp_interval` frame numbers refer to
     that version), cut to the sign (plus a little padding), H.264, no audio
     -> <clips_dir>/<word>/<name>.mp4   (served to the browser for Speech -> Sign)
  2. MediaPipe Hand + Pose landmarkers on the trimmed clip
     -> <landmarks_dir>/<name>.json     (input to build_index.py for Sign -> Speech)

Usage:
    python process_clips.py --meta bsldict_meta.json --raw <raw_video_dir> \
        --clips <project>/media/signs --landmarks <landmark_dir> --models <model_dir>
"""
import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PAD = 3  # frames of padding kept around the sign for natural playback

_hand = _pose = None


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def trim(src: Path, dst: Path, start: int, end: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"fps=25,scale=-2:360,trim=start_frame={start}:end_frame={end + 1},"
          "setpts=PTS-STARTPTS")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-an",
           "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, timeout=300)


def _init(models: str) -> None:
    global _hand, _pose
    import mediapipe as mp
    v = mp.tasks.vision
    base = mp.tasks.BaseOptions
    _hand = (v.HandLandmarker, v.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(Path(models) / "hand_landmarker.task")),
        running_mode=v.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
    _pose = (v.PoseLandmarker, v.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(Path(models) / "pose_landmarker_lite.task")),
        running_mode=v.RunningMode.VIDEO, num_poses=1))


def landmarks(clip: Path) -> dict:
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(clip))
    frames = []
    with _hand[0].create_from_options(_hand[1]) as hand, _pose[0].create_from_options(_pose[1]) as pose:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = i * 40  # 25 fps
            hr = hand.detect_for_video(img, ts)
            pr = pose.detect_for_video(img, ts)
            hands = [{"label": hd[0].category_name,
                      "lm": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in lms]}
                     for hd, lms in zip(hr.handedness, hr.hand_landmarks)]
            body = None
            if pr.pose_landmarks:
                pl = pr.pose_landmarks[0]
                body = {k: [round(pl[j].x, 4), round(pl[j].y, 4)] for k, j in (("nose", 0), ("ls", 11), ("rs", 12))}
            frames.append({"aspect": round(w / h, 4), "hands": hands, "body": body})
            i += 1
    cap.release()
    return {"frames": frames}


def process(video: dict, raw: Path, clips: Path, lmdir: Path) -> tuple[str, str | None]:
    name = video["name"]
    src = raw / name
    if not src.exists():
        return name, "not downloaded"
    a, b = video["temp_interval"]
    start = max(0, a - PAD)
    end = min(video["v360"]["T"] - 1, b + PAD)
    dst = clips / video["word"] / name
    out = lmdir / (Path(name).stem + ".json")
    try:
        if not dst.exists():
            trim(src, dst, start, end)
        if not out.exists():
            data = landmarks(dst)
            data.update(word=video["word"], name=name, core=[a - start, b - start])
            out.write_text(json.dumps(data), encoding="utf-8")
        return name, None
    except Exception as e:  # noqa: BLE001 - keep going, report at the end
        return name, str(e)[:200]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    meta = json.loads(a.meta.read_text(encoding="utf-8"))
    downloaded = {f.name for f in a.raw.glob("*.mp4")}
    videos = [v for v in meta["videos"] if v["name"] in downloaded]
    a.landmarks.mkdir(parents=True, exist_ok=True)
    print(f"processing {len(videos)} videos")

    failed = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(str(a.models),)) as pool:
        futures = [pool.submit(process, v, a.raw, a.clips, a.landmarks) for v in videos]
        for i, fut in enumerate(as_completed(futures), 1):
            name, err = fut.result()
            if err:
                failed.append(f"{name}\t{err}")
            if i % 50 == 0:
                print(f"  {i}/{len(videos)} done, {len(failed)} failed", flush=True)
    (a.landmarks / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"finished: {len(videos) - len(failed)} ok, {len(failed)} failed")


if __name__ == "__main__":
    main()
