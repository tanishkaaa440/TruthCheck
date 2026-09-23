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


def _download_video(url, workdir):
    output_path = os.path.join(workdir, "video.mp4")
    ydl_opts = {
        "outtmpl": output_path,
        "format": "bestvideo[vcodec^=avc1][height<=480]+bestaudio/best[height<=480]",
        "ffmpeg_location": FFMPEG_PATH,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp appends the real container extension (e.g. .mkv) -- find actual file
        # rather than assuming output_path is exact (known gotcha from manual testing).
        base = os.path.splitext(output_path)[0]
        for f in os.listdir(workdir):
            if f.startswith("video."):
                return os.path.join(workdir, f)
    raise FileNotFoundError("Downloaded video file not found after yt-dlp run")


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

    # NOTE: absolute % is not well-calibrated (see calibration testing) --
    # flag based on variance/consistency instead, per validated approach.
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
    Full Phase 7 pipeline: download -> frame deepfake analysis + audio transcription
    -> feed transcript into existing claim fact-check pipeline -> combined verdict.
    """
    workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_tmp_{uuid.uuid4().hex[:8]}")
    os.makedirs(workdir, exist_ok=True)

    try:
        video_path = _download_video(url, workdir)

        frame_paths = _extract_frames(video_path, workdir)
        deepfake_result = _analyze_frames(frame_paths)

        transcript = _extract_and_transcribe_audio(video_path, workdir)

        claim_result = None
        if transcript:
            search_results = search_claim(transcript)
            claim_result = calculate_trust_score(search_results, transcript)

        return {
            "video_url": url,
            "transcript": transcript,
            "deepfake_analysis": deepfake_result,
            "claim_verdict": claim_result,
        }
    finally:
        # Clean up temp files -- avoid repeating the earlier disk-space issue
        shutil.rmtree(workdir, ignore_errors=True)