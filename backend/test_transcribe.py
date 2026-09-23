import os
import subprocess
import requests
import imageio_ffmpeg
from dotenv import load_dotenv, dotenv_values

# Explicit path -- guarantees we load the .env sitting next to this script,
# regardless of quirks in dotenv's auto-search when run different ways.
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
print(f"DEBUG: loading .env from {ENV_PATH}")
print(f"DEBUG: that file exists = {os.path.exists(ENV_PATH)}")

# Read the file directly, bypassing os.environ, to see exactly what dotenv parses
parsed = dotenv_values(ENV_PATH)
print(f"DEBUG: keys found by parser = {list(parsed.keys())}")
print(f"DEBUG: HF_API_KEY length via parser = {len(parsed.get('HF_API_KEY') or '')}")

load_dotenv(dotenv_path=ENV_PATH, override=True)

HF_API_KEY = os.getenv("HF_API_KEY")
print(f"DEBUG: HF_API_KEY length after load = {len(HF_API_KEY) if HF_API_KEY else 0}")
MODEL = "openai/whisper-large-v3"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

VIDEO_FILE = "test_video.mp4.mkv"  # matches your confirmed working download filename
AUDIO_FILE = "test_audio.mp3"


def extract_audio():
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Using ffmpeg at: {ffmpeg_path}")

    if not os.path.exists(VIDEO_FILE):
        print(f"ERROR: {VIDEO_FILE} not found. Run test_download.py first.")
        return False

    cmd = [
        ffmpeg_path,
        "-i", VIDEO_FILE,
        "-vn",                 # no video
        "-acodec", "libmp3lame",
        "-q:a", "4",           # reasonable quality, small file
        "-y",                  # overwrite if exists
        AUDIO_FILE
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg audio extraction FAILED:")
        print(result.stderr[-1500:])  # last part of ffmpeg's error output
        return False

    size_kb = os.path.getsize(AUDIO_FILE) / 1024
    print(f"Audio extracted: {AUDIO_FILE} ({size_kb:.1f} KB)")
    return True


def transcribe_audio():
    if not HF_API_KEY:
        print("ERROR: HF_API_KEY not found in .env")
        return None

    with open(AUDIO_FILE, "rb") as f:
        data = f.read()

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "audio/mpeg"}
    print("Sending audio to Whisper API (this can take 10-30s for longer clips)...")
    resp = requests.post(API_URL, headers=headers, data=data, timeout=120)

    if resp.status_code != 200:
        print(f"ERROR {resp.status_code} -> {resp.text}")
        return None

    result = resp.json()
    return result.get("text", "")


def main():
    if not extract_audio():
        return

    transcript = transcribe_audio()
    if transcript is None:
        return

    print("\n--- TRANSCRIPT ---")
    print(transcript.strip())
    print("\n(Next step: feed this transcript into search_engine.py + trust_score.py")
    print("the same way typed/shared claim text is handled now.)")


if __name__ == "__main__":
    main()