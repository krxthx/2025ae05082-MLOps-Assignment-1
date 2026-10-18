import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap

sys.path.insert(0, "/private/tmp/assignment-video-tools")
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/private/tmp/assignment-recording")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def duration(path):
    result = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError(f"Cannot inspect {path}: {result.stderr}")
    h, m, s = map(float, match.groups())
    return h * 3600 + m * 60 + s


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    h, milliseconds = divmod(milliseconds, 3600000)
    m, milliseconds = divmod(milliseconds, 60000)
    s, milliseconds = divmod(milliseconds, 1000)
    return f"{h:02}:{m:02}:{s:02},{milliseconds:03}"


def main():
    scenes = json.loads((WORK / "scenes.json").read_text())
    subtitles, chapters, files = [], [], []
    offset = 0
    for scene in scenes:
        i = scene["index"]
        raw = WORK / f"scene_{i:02}.webm"
        length = duration(raw)
        filters = []
        event_path = WORK / f"scene_{i:02}_events.json"
        if i == 3 and event_path.exists():
            events = json.loads(event_path.read_text())
            training = next(e for e in events if e["command"] == "python3 src/train.py")
            cut_start = training["start"] + 3
            cut_end = training["completed"] - 3
            if cut_end > cut_start + 2:
                filters = ["-filter_complex", f"[0:v]split[a][b];[a]trim=end={cut_start},setpts=PTS-STARTPTS[a1];[b]trim=start={cut_end},setpts=PTS-STARTPTS[b1];[a1][b1]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "1:a"]
                length -= cut_end - cut_start
                scene["trimmed_training_wait_seconds"] = cut_end - cut_start
        padding = max(0, scene["duration"] + 3 - length)
        if filters:
            filters[1] = filters[1].replace("concat=n=2:v=1:a=0[v]", f"concat=n=2:v=1:a=0,tpad=stop_mode=clone:stop_duration={padding}[v]")
        elif padding:
            filters = ["-vf", f"tpad=stop_mode=clone:stop_duration={padding}"]
        length += padding
        clip = WORK / f"final_{i:02}.mp4"
        title = scene["title"].split(" · ")[0].split(". ", 1)[1]
        print(f"Encoding {i}: {length:.2f}s", flush=True)
        if not clip.exists() or clip.stat().st_mtime < raw.stat().st_mtime:
            subprocess.run([
            FFMPEG, "-v", "error", "-y", "-i", str(raw), "-i", str(WORK / f"{i:02}.wav"),
            *filters, "-af", "adelay=800,apad", "-t", str(length), "-r", "25", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(clip)
        ], check=True)
        files.append("file '" + str(clip) + "'")
        words = scene["speech"].split()
        for start in range(0, len(words), 12):
            end = min(start + 12, len(words))
            begin_time = offset + .8 + scene["duration"] * start / len(words)
            end_time = offset + .8 + scene["duration"] * end / len(words)
            text = textwrap.fill(" ".join(words[start:end]), width=56)
            subtitles.append(f"{len(subtitles)+1}\n{timestamp(begin_time)} --> {timestamp(end_time)}\n{text}\n")
        chapters.append(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(offset*1000)}\nEND={round((offset+length)*1000)}\ntitle={title}\n")
        scene["video_start_seconds"] = offset
        scene["video_duration_seconds"] = length
        offset += length
    (WORK / "concat.txt").write_text("\n".join(files))
    subtitle_path = ROOT / "demo/pipeline_walkthrough_narrated.srt"
    subtitle_path.write_text("\n".join(subtitles))
    (WORK / "chapters.ffmeta").write_text(";FFMETADATA1\ntitle=Heart Disease MLOps - Live Pipeline Walkthrough\ncomment=Real browser interactions and command execution; synthetic Samantha narration.\n" + "\n".join(chapters))
    output = ROOT / "demo/pipeline_walkthrough_narrated.mp4"
    subprocess.run([
        FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(WORK / "concat.txt"),
        "-i", str(subtitle_path), "-i", str(WORK / "chapters.ffmeta"), "-map", "0:v", "-map", "0:a",
        "-map", "1:0", "-map_metadata", "2", "-map_chapters", "2", "-c", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", str(output)
    ], check=True)
    (ROOT / "demo/recording/timeline.json").write_text(json.dumps(scenes, indent=2))
    (ROOT / "demo/NARRATION.txt").write_text("\n\n".join(scene["speech"] for scene in scenes) + "\n")
    print(f"Exported {output}: {offset:.2f}s", flush=True)


if __name__ == "__main__":
    main()
