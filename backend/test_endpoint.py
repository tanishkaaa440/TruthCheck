import requests
import json

URL = "http://localhost:8000/verify-video"
TEST_VIDEO = "https://www.youtube.com/watch?v=HlVPNg5jPRA"

print(f"Sending request to {URL} ...")
print("This can take 30-60+ seconds (download + frame analysis + transcription + fact-check).")

resp = requests.post(URL, json={"url": TEST_VIDEO}, timeout=300)

print(f"\nStatus code: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))