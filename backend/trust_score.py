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