# TruthCheck Project — Resume Memory File

**Owner:** Tanishka (solo build)
**Last updated:** Session ending ~Sep 14, 2026
**Paste this WHOLE file into a new chat to resume exactly where left off.**

---

## PROJECT SUMMARY

AI-powered misinformation detection system. Started as browser-extension idea, now web app (frontend + backend). Flow: user submit claim → backend search_engine fetch sources → trust_score.py score claim true/false/mixed using HuggingFace zero-shot model + trusted-domain weighting → verdict shown with emoji (🟢 Verified / 🟡 Unverified / 🔴 Red Flag).

## STATUS — DONE

- Backend core done.
- `trust_score.py` done, tested both directions:
  - True claim ("earth is round" type) → score 100, 🟢 Verified.
  - False claim ("vaccines cause autism") → score 0, 🔴 Red Flag.
  - Emoji fixed using Unicode escape codes (`\U0001F7E2` etc) instead of raw emoji chars — raw chars broke on copy-paste/encoding before. Escape-code version confirmed safe.
- `search_engine.py` done (feeds `organic` results list into trust_score).
- Frontend basic test page done (plain HTML, not PWA yet).

## KNOWN GOTCHAS (don't re-debug these)

1. Windows PowerShell terminal does NOT render emoji correctly — cosmetic only, not a bug. Browser/phone renders fine. Do not "fix" this again.
2. Always use `\U0001F7E2` / `\U0001F7E1` / `\U0001F534` Unicode escapes in Python source, never paste raw emoji char directly into file — causes encoding crash.
3. HF_API_KEY loaded via `.env` + `python-dotenv`. Missing key silently degrades to `("neutral", 0)` stance — check `.env` first if scores look flat/wrong.

## CURRENT FILE: trust_score.py (latest working version)

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

TRUSTED_DOMAINS = [
    "nasa.gov", "who.int", "cdc.gov", "gov.in", "wikipedia.org",
    "bbc.com", "reuters.com", "apnews.com", "nature.com",
    "sciencedirect.com", "ncbi.nlm.nih.gov", "britannica.com",
    "nationalgeographic.com", "space.com", "livescience.com",
    "snopes.com", "factcheck.org", "politifact.com", "scientificamerican.com",
    "esa.int", "noaa.gov", "usgs.gov", "un.org"
]

GREEN = "\U0001F7E2"
YELLOW = "\U0001F7E1"
RED = "\U0001F534"

def is_trusted(link: str) -> bool:
    return any(domain in link for domain in TRUSTED_DOMAINS)

def check_stance_ai(claim: str, snippet: str) -> tuple:
    if not snippet or not HF_API_KEY:
        return ("neutral", 0)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": snippet,
        "parameters": {
            "candidate_labels": [
                f"this confirms the claim is true: {claim}",
                f"this shows the claim is false: {claim}",
                "this is unrelated"
            ]
        }
    }

    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=15)
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            print("HF API returned error:", result["error"])
            return ("neutral", 0)

        if not isinstance(result, list) or len(result) == 0:
            print("HF unexpected response format:", result)
            return ("neutral", 0)

        top = result[0]
        top_label = top.get("label", "")
        top_score = top.get("score", 0)

        if top_score < 0.55:
            return ("neutral", top_score)
        if "true" in top_label:
            return ("true", top_score)
        elif "false" in top_label:
            return ("false", top_score)
        else:
            return ("neutral", top_score)
    except Exception as e:
        print("HF API error:", e)
        return ("neutral", 0)

def calculate_trust_score(search_results: dict, claim: str = "") -> dict:
    organic = search_results.get("organic", [])
    if not organic:
        return {"score": 0, "verdict": f"{RED} Red Flag", "sources": []}

    true_weight = 0
    false_weight = 0
    trusted_count = 0
    sources = []

    for item in organic[:5]:
        link = item.get("link", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        combined_text = f"{title}. {snippet}"

        trusted = is_trusted(link)
        stance, confidence = check_stance_ai(claim, combined_text) if claim else ("neutral", 0)

        weight_multiplier = 2 if trusted else 1

        if trusted:
            trusted_count += 1
        if stance == "true":
            true_weight += confidence * weight_multiplier
        elif stance == "false":
            false_weight += confidence * weight_multiplier

        sources.append({
            "title": title,
            "link": link,
            "trusted": trusted,
            "stance": stance,
            "confidence": round(confidence, 2)
        })

    total_checked = min(len(organic), 5)
    trust_ratio = (trusted_count / total_checked) if total_checked else 0

    if false_weight > true_weight and false_weight >= 0.55:
        score = max(0, 20 - int(false_weight * 15))
        verdict = f"{RED} Red Flag (likely false claim)"
    elif true_weight > false_weight and true_weight >= 0.55:
        score = min(100, 65 + int(true_weight * 20) + int(trust_ratio * 15))
        verdict = f"{GREEN} Verified"
    elif true_weight > 0 and false_weight > 0:
        score = 40 + int(trust_ratio * 15)
        verdict = f"{YELLOW} Unverified (mixed signals)"
    else:
        score = int(trust_ratio * 100)
        if score >= 60:
            verdict = f"{GREEN} Verified"
        elif score >= 30:
            verdict = f"{YELLOW} Unverified"
        else:
            verdict = f"{RED} Red Flag"

    score = max(0, min(100, score))

    return {"score": score, "verdict": verdict, "sources": sources}
```

**Action pending on this file:** paste into `trust_score.py` (Ctrl+A → Delete → paste → Ctrl+S), restart backend, retest both true+false claims once more to confirm emoji render correctly on phone/browser (not PowerShell).

## CURRENT PHASE: Phase 4 — PWA build

Goal: turn frontend into installable Progressive Web App w/ Android share-target support (share text/link from IG/YT straight into TruthCheck).

### Step 1 of 5 — DONE THIS SESSION: manifest.json created

File: `frontend/manifest.json` (new file, same folder as `index.html`)

```json
{
  "name": "TruthCheck",
  "short_name": "TruthCheck",
  "description": "AI-powered misinformation checker",
  "start_url": "/index.html",
  "display": "standalone",
  "background_color": "#121212",
  "theme_color": "#4CAF50",
  "icons": [
    {
      "src": "icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "share_target": {
    "action": "/index.html",
    "method": "GET",
    "params": {
      "title": "title",
      "text": "text",
      "url": "url"
    }
  }
}
```

Note: `share_target` here is basic query-param version. Android puts shared text into URL like `index.html?text=...`. Frontend JS must read that param and auto-fill textarea — **not done yet**, planned as later sub-step once manifest + icons confirmed working.

### Remaining steps (Phase 4, PWA build) — NOT STARTED

2. Register `sw.js` (service worker — offline support + installability requirement).
3. Add Web Share Target config confirmation/testing (Android share menu entry) — manifest field already written above, needs real device test.
4. Test "Add to Home Screen" on phone.
5. Test sharing a caption from IG/YT into the app (end-to-end share-target flow).

### Immediate open item (asked, not yet answered by user)

Icons needed: `icon-192.png` and `icon-512.png`, square, go in `frontend/` folder. Question pending: does Tanishka already have a logo/icon image, or need help finding/making a simple placeholder?

## NEXT ACTION WHEN RESUMING

Reply with icon status (have one / need placeholder), then continue: create/place icons → write `sw.js` → test Add to Home Screen → test share-target flow.
