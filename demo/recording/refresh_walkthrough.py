import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, "/private/tmp/assignment-video-tools")
import imageio_ffmpeg

from assemble_walkthrough import duration

ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/private/tmp/assignment-recording")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
RETAKES = {1, 4, 5, 9}


def main():
    scenes = json.loads((WORK / "original_timeline.json").read_text())
    files, chapters = [], []
    for scene in scenes:
        index = scene["index"]
        length = scene["video_duration_seconds"]
        clip = WORK / f"retained_{index:02}.mp4"
        if index in RETAKES:
            raw = WORK / f"scene_{index:02}.webm"
            raw_length = duration(raw)
            speed = min(1, length / raw_length)
            clip = WORK / f"refreshed_{index:02}.mp4"
            subprocess.run([
                FFMPEG, "-v", "error", "-y", "-i", str(raw),
                "-i", str(WORK / f"original_audio_{index:02}.wav"),
                "-map", "0:v:0", "-map", "1:a:0",
                "-vf", f"setpts={speed}*PTS,tpad=stop_mode=clone:stop_duration=3",
                "-af", "apad", "-t", str(length), "-r", "25",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(clip),
            ], check=True)
            scene["reshoot_date"] = "2026-10-18"
            scene["raw_reshoot_seconds"] = raw_length
            scene["video_timing_scale"] = speed
        files.append(f"file '{clip}'")
        start = scene["video_start_seconds"]
        title = scene["title"].split(" · ")[0].split(". ", 1)[1]
        chapters.append(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(start*1000)}\nEND={round((start+length)*1000)}\ntitle={title}\n")
    (WORK / "refresh_concat.txt").write_text("\n".join(files))
    (WORK / "refresh_chapters.ffmeta").write_text(
        ";FFMETADATA1\ntitle=Heart Disease MLOps - Live Pipeline Walkthrough\n"
        "comment=Live browser footage; synthetic narration; repository and MLflow views refreshed 2026-10-18.\n"
        + "\n".join(chapters)
    )
    candidate = WORK / "pipeline_walkthrough_refreshed.mp4"
    subprocess.run([
        FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(WORK / "refresh_concat.txt"),
        "-i", str(ROOT / "demo/pipeline_walkthrough_narrated.srt"), "-i", str(WORK / "refresh_chapters.ffmeta"),
        "-map", "0:v:0", "-map", "0:a:0", "-map", "1:0", "-map_metadata", "2", "-map_chapters", "2",
        "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", str(candidate),
    ], check=True)
    (WORK / "refreshed_timeline.json").write_text(json.dumps(scenes, indent=2))
    print(json.dumps({"candidate": str(candidate), "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
