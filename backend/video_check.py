import os
import uuid
import shutil
import subprocess
import requests
import cv2
import yt_dlp
import imageio_ffmpeg
from dotenv import load_dotenv

from search_engine import search_claim
from trust_score import calculate_trust_score

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

HF_API_KEY = os.getenv("HF_API_KEY")
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

DEEPFAKE_MODEL_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/Deep-Fake-Detector-v2-Model"
WHISPER_MODEL_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"

FRAME_SAMPLE_EVERY = 5  # matches confirmed-working sampling rate from testing

# Below this length, a clip is far more likely to be a snippet cut from a
# longer video rather than a complete, self-contained piece of footage --
# a very common way real footage gets used to spread misinformation
# (true clip, missing context that would change its meaning).
SHORT_CLIP_THRESHOLD_SEC = 25


def _download_video(url, workdir):
    output_path = os.path.join(workdir, "video.mp4")
    ydl_opts = {
        "outtmpl": output_path,
        "format": "bestvideo[vcodec^=avc1][height<=480]+bestaudio/best[height<=480]",
        "ffmpeg_location": FFMPEG_PATH,
        "quiet": True,
        # Pull top comments too, capped for speed -- [max_total, max_parents,
        # max_replies, max_replies_per_thread]. 15 top-level, sorted by likes.
        "getcomments": True,
        "extractor_args": {
            "youtube": {"comment_sort": ["top"], "max_comments": ["15", "15", "0", "0"]}
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        video_path = None
        for f in os.listdir(workdir):
            if f.startswith("video."):
                video_path = os.path.join(workdir, f)
                break
        if not video_path:
            raise FileNotFoundError("Downloaded video file not found after yt-dlp run")

        title = info.get("title", "")
        description = info.get("description", "")

        raw_comments = info.get("comments") or []
        comments = [
            {
                "text": c.get("text", ""),
                "author": c.get("author", ""),
                "likes": c.get("like_count", 0),
            }
            for c in raw_comments[:15]
        ]

        return {
            "video_path": video_path,
            "title": title,
            "description": description,
            "comments": comments,
        }


def _get_duration_sec(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return None
    return frame_count / fps


def _analyze_context_risk(duration_sec):
    """
    Heuristic flag for 'edited / cut out of context' videos.
    NOTE: this is a duration-based heuristic, not true context verification
    (which would require reverse video search against the original source --
    a separate, larger feature). It flags the *pattern* common to
    out-of-context clips so a human reviewer knows to look closer,
    it does not claim to prove the video was manipulated.
    """
    if duration_sec is None:
        return {"status": "error", "message": "Could not determine video duration"}

    is_short = duration_sec < SHORT_CLIP_THRESHOLD_SEC

    return {
        "duration_sec": round(duration_sec, 1),
        "flag": "possible_out_of_context_clip" if is_short else "normal_length",
        "note": (
            f"Clip is under {SHORT_CLIP_THRESHOLD_SEC}s -- short isolated clips are "
            "commonly used to spread misinformation by stripping away context that "
            "would change how a statement is understood. Recommend checking for the "
            "original, full-length source before trusting this clip alone."
            if is_short else
            "Clip length is not unusually short on its own."
        )
    }


def _extract_frames(video_path, workdir, interval_sec=2):
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(fps * interval_sec), 1)

    count, saved = 0, 0
    paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(frames_dir, f"frame_{saved}.jpg")
            cv2.imwrite(filename, frame)
            paths.append(filename)
            saved += 1
        count += 1
    cap.release()
    return paths


def _classify_frame(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "image/jpeg"}
    resp = requests.post(DEEPFAKE_MODEL_URL, headers=headers, data=data, timeout=30)
    if resp.status_code != 200:
        return None
    result = resp.json()
    fake_entry = next((r for r in result if r["label"].lower() in ("deepfake", "fake")), None)
    return fake_entry["score"] if fake_entry else None


def _analyze_frames(frame_paths):
    sampled = frame_paths[::FRAME_SAMPLE_EVERY] or frame_paths
    scores = [s for s in (_classify_frame(p) for p in sampled) if s is not None]

    if not scores:
        return {"status": "error", "message": "No frames could be analyzed"}

    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5

    flag = "high_variance_suspicious" if std_dev > 0.15 else "consistent_low_risk"

    return {
        "status": "ok",
        "frames_analyzed": len(scores),
        "average_score": round(avg, 3),
        "std_dev": round(std_dev, 3),
        "flag": flag
    }


def _extract_and_transcribe_audio(video_path, workdir):
    audio_path = os.path.join(workdir, "audio.mp3")
    cmd = [
        FFMPEG_PATH, "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        "-y", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(audio_path):
        return None

    with open(audio_path, "rb") as f:
        data = f.read()
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "audio/mpeg"}
    resp = requests.post(WHISPER_MODEL_URL, headers=headers, data=data, timeout=120)
    if resp.status_code != 200:
        return None
    return resp.json().get("text", "").strip()


def process_video(url):
    """
    Full video verification pipeline, covering three misinformation patterns:
      1. General false-claim spreading -- transcript + caption fed into fact-check pipeline
      2. AI-generated / deepfake video -- frame-level variance analysis
      3. Edited / clipped out-of-context video -- duration-based heuristic flag
    Also pulls title, description/caption, and top comments so the response
    reflects the full context around the video, not just the raw footage.
    """
    workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_tmp_{uuid.uuid4().hex[:8]}")
    os.makedirs(workdir, exist_ok=True)

    try:
        video_info = _download_video(url, workdir)
        video_path = video_info["video_path"]

        duration_sec = _get_duration_sec(video_path)
        context_risk = _analyze_context_risk(duration_sec)

        frame_paths = _extract_frames(video_path, workdir)
        deepfake_result = _analyze_frames(frame_paths)

        transcript = _extract_and_transcribe_audio(video_path, workdir)

        # Combine caption (title + description) with the spoken transcript --
        # captions often state the explicit claim more clearly than the
        # audio does (e.g. "BREAKING: X did Y" in a caption vs a vaguer
        # spoken clip), so checking both gives a more complete picture.
        claim_text_parts = []
        if video_info["title"]:
            claim_text_parts.append(video_info["title"])
        if video_info["description"]:
            claim_text_parts.append(video_info["description"])
        if transcript:
            claim_text_parts.append(transcript)
        combined_claim_text = " ".join(claim_text_parts).strip()

        claim_result = None
        if combined_claim_text:
            search_results = search_claim(combined_claim_text)
            claim_result = calculate_trust_score(search_results, combined_claim_text)

        return {
            "video_url": url,
            "title": video_info["title"],
            "description": video_info["description"],
            "comments": video_info["comments"],
            "transcript": transcript,
            "deepfake_analysis": deepfake_result,
            "context_risk": context_risk,
            "claim_verdict": claim_result,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)