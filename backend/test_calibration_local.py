import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "Wvolf/ViT_Deepfake_Detection"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "image/jpeg"}

# Save these two files yourself before running:
#   control_fake.jpg -- screenshot/save from https://thispersondoesnotexist.com/
#   control_real.jpg -- any real photo (selfie, phone photo, anything genuine)
FAKE_PATH = "control_fake.jpg"
REAL_PATH = "control_real.jpg"


def classify_file(path, label):
    if not os.path.exists(path):
        print(f"{label}: ERROR -- file not found at {path}")
        return None
    with open(path, "rb") as f:
        data = f.read()
    print(f"{label}: loaded {len(data)} bytes from {path}")
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

    fake_score = classify_file(FAKE_PATH, "KNOWN-FAKE control")
    print()
    real_score = classify_file(REAL_PATH, "KNOWN-REAL control")

    print("\n--- VERDICT ---")
    if fake_score is not None and real_score is not None:
        gap = fake_score - real_score
        print(f"Fake control score: {fake_score*100:.1f}%")
        print(f"Real control score: {real_score*100:.1f}%")
        print(f"Gap: {gap*100:.1f} points")
        if gap > 0.3:
            print("Model discriminates well. Your 81% video result is likely trustworthy.")
        elif real_score > 0.5:
            print("Model is over-flagging real images as fake too. Your 81% video result is")
            print("probably compression-artifact noise, not a real deepfake signal.")
            print("Consider trying a different model (e.g. Wvolf/ViT_Deepfake_Detection).")
        else:
            print("Ambiguous gap -- inspect raw scores above before trusting video result.")


if __name__ == "__main__":
    main()