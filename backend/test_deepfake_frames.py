import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

FRAMES_DIR = "frames"

# Sample every Nth frame instead of all 107 -- keeps API calls + cost down.
# Set to 1 to run every frame.
SAMPLE_EVERY = 5


def classify_frame(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    headers = dict(HEADERS)
    headers["Content-Type"] = "image/jpeg"
    resp = requests.post(API_URL, headers=headers, data=data, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"{resp.status_code} -> {resp.text}")
    result = resp.json()
    # New HF router format: list of {label, score} dicts, sorted descending
    return result


def main():
    if not HF_API_KEY:
        print("ERROR: HF_API_KEY not found in .env")
        return

    frame_files = sorted(
        [f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")],
        key=lambda x: int(x.replace("frame_", "").replace(".jpg", ""))
    )

    if not frame_files:
        print(f"ERROR: no .jpg files found in {FRAMES_DIR}/")
        return

    sampled = frame_files[::SAMPLE_EVERY]
    print(f"Found {len(frame_files)} frames total, sampling {len(sampled)} (every {SAMPLE_EVERY})")

    fake_scores = []
    errors = 0

    for i, fname in enumerate(sampled):
        filepath = os.path.join(FRAMES_DIR, fname)
        try:
            result = classify_frame(filepath)
            # find the "Deepfake" label's score regardless of order
            fake_entry = next(
                (r for r in result if r["label"].lower() in ("deepfake", "fake")),
                None
            )
            if fake_entry:
                score = fake_entry["score"]
                fake_scores.append(score)
                print(f"[{i+1}/{len(sampled)}] {fname}: fake_score={score:.3f}")
            else:
                print(f"[{i+1}/{len(sampled)}] {fname}: unexpected format -> {result}")
                errors += 1
        except Exception as e:
            print(f"[{i+1}/{len(sampled)}] {fname}: ERROR -> {e}")
            errors += 1
        time.sleep(0.2)  # light throttle, avoid rate limits

    if not fake_scores:
        print("\nNo successful classifications. Check API key / model availability.")
        return

    avg_score = sum(fake_scores) / len(fake_scores)
    max_score = max(fake_scores)

    print("\n--- SUMMARY ---")
    print(f"Frames analyzed: {len(fake_scores)} (errors: {errors})")
    print(f"Average fake-likelihood: {avg_score*100:.1f}%")
    print(f"Peak fake-likelihood (worst frame): {max_score*100:.1f}%")
    print(f"\nSuggested verdict: Manipulation Likelihood = {avg_score*100:.1f}%")
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "image/jpeg"}

FRAMES_DIR = "frames"

# Sample every Nth frame instead of all 107 -- keeps API calls + cost down.
# Set to 1 to run every frame.
SAMPLE_EVERY = 5


def classify_frame(filepath):
    with open(filepath, "rb") as f:
        data = f.read()
    headers = dict(HEADERS)
    headers["Content-Type"] = "image/jpeg"
    resp = requests.post(API_URL, headers=headers, data=data, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"{resp.status_code} -> {resp.text}")
    result = resp.json()
    # New HF router format: list of {label, score} dicts, sorted descending
    return result


def main():
    if not HF_API_KEY:
        print("ERROR: HF_API_KEY not found in .env")
        return

    frame_files = sorted(
        [f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")],
        key=lambda x: int(x.replace("frame_", "").replace(".jpg", ""))
    )

    if not frame_files:
        print(f"ERROR: no .jpg files found in {FRAMES_DIR}/")
        return

    sampled = frame_files[::SAMPLE_EVERY]
    print(f"Found {len(frame_files)} frames total, sampling {len(sampled)} (every {SAMPLE_EVERY})")

    fake_scores = []
    errors = 0

    for i, fname in enumerate(sampled):
        filepath = os.path.join(FRAMES_DIR, fname)
        try:
            result = classify_frame(filepath)
            # find the "Deepfake" label's score regardless of order
            fake_entry = next(
                (r for r in result if r["label"].lower() in ("deepfake", "fake")),
                None
            )
            if fake_entry:
                score = fake_entry["score"]
                fake_scores.append(score)
                print(f"[{i+1}/{len(sampled)}] {fname}: fake_score={score:.3f}")
            else:
                print(f"[{i+1}/{len(sampled)}] {fname}: unexpected format -> {result}")
                errors += 1
        except Exception as e:
            print(f"[{i+1}/{len(sampled)}] {fname}: ERROR -> {e}")
            errors += 1
        time.sleep(0.2)  # light throttle, avoid rate limits

    if not fake_scores:
        print("\nNo successful classifications. Check API key / model availability.")
        return

    avg_score = sum(fake_scores) / len(fake_scores)
    max_score = max(fake_scores)
    min_score = min(fake_scores)
    variance = sum((s - avg_score) ** 2 for s in fake_scores) / len(fake_scores)
    std_dev = variance ** 0.5

    print("\n--- SUMMARY ---")
    print(f"Frames analyzed: {len(fake_scores)} (errors: {errors})")
    print(f"Average fake-likelihood: {avg_score*100:.1f}%")
    print(f"Range: {min_score*100:.1f}% - {max_score*100:.1f}%")
    print(f"Std deviation: {std_dev*100:.1f} points")
    print()
    print("NOTE: absolute % from this model is unreliable (see calibration test).")
    print("What matters more here: consistency across frames.")
    if std_dev > 0.15:
        print("HIGH variance -- scores swing a lot frame to frame. This pattern (rather")
        print("than the raw average) is more consistent with localized manipulation")
        print("(e.g. face-swapped region flickering) than a uniformly real OR fake video.")
    else:
        print("LOW variance -- scores stay fairly consistent frame to frame. Whatever the")
        print("model's baseline bias is, it's being applied uniformly -- more consistent")
        print("with genuine, unmanipulated footage (even if the absolute % is unreliable).")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()