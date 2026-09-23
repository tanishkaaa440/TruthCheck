import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "image/jpeg"}

# Known-fake: thispersondoesnotexist.com serves a fresh GAN-generated face every request
FAKE_CONTROL_URL = "https://thispersondoesnotexist.com/"

# Known-real: stable Wikimedia Commons photo (real person, real photo, no edits)
REAL_CONTROL_URL = "https://upload.wikimedia.org/wikipedia/commons/8/89/Barack_Obama_2012_portrait_cropped.jpg"


DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TruthCheckBot/1.0"}


def fetch_image(url, label):
    resp = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=15)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    print(f"{label}: fetched {len(resp.content)} bytes, Content-Type={ctype}")
    if "image" not in ctype:
        print(f"{label}: WARNING -- response doesn't look like an image, likely blocked/HTML error page")
    return resp.content


def classify_bytes(data, label):
    resp = requests.post(API_URL, headers=HEADERS, data=data, timeout=30)
    if resp.status_code != 200:
        print(f"{label}: ERROR {resp.status_code} -> {resp.text}")
        return None
    result = resp.json()
    fake_entry = next((r for r in result if r["label"].lower() in ("deepfake", "fake")), None)
    if fake_entry:
        print(f"{label}: fake_score={fake_entry['score']:.3f}  (raw: {result})")
        return fake_entry["score"]
    print(f"{label}: unexpected format -> {result}")
    return None


def main():
    if not HF_API_KEY:
        print("ERROR: HF_API_KEY not found in .env")
        return

    print("Fetching known-fake control image (AI-generated face)...")
    fake_img = fetch_image(FAKE_CONTROL_URL, "KNOWN-FAKE control")
    fake_score = classify_bytes(fake_img, "KNOWN-FAKE control")

    print("\nFetching known-real control image (real photo)...")
    real_img = fetch_image(REAL_CONTROL_URL, "KNOWN-REAL control")
    real_score = classify_bytes(real_img, "KNOWN-REAL control")

    print("\n--- VERDICT ---")
    if fake_score is not None and real_score is not None:
        gap = fake_score - real_score
        print(f"Fake control score: {fake_score*100:.1f}%")
        print(f"Real control score: {real_score*100:.1f}%")
        print(f"Gap: {gap*100:.1f} points")
        if gap > 0.3:
            print("Model discriminates well. Your 81% video result is likely trustworthy.")
        elif real_score > 0.5:
            print("Model is over-flagging real images as fake too. Your 81% result is")
            print("probably compression-artifact noise, not a real deepfake signal.")
            print("Consider trying a different model (e.g. Wvolf/ViT_Deepfake_Detection).")
        else:
            print("Ambiguous gap -- inspect raw scores above before trusting video result.")


if __name__ == "__main__":
    main()