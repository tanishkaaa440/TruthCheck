import os
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

TRUSTED_DOMAINS = [
    # International news / science
    "nasa.gov", "who.int", "cdc.gov", "gov.in", "wikipedia.org",
    "bbc.com", "reuters.com", "apnews.com", "nature.com",
    "sciencedirect.com", "ncbi.nlm.nih.gov", "britannica.com",
    "nationalgeographic.com", "space.com", "livescience.com",
    "scientificamerican.com", "esa.int", "noaa.gov", "usgs.gov", "un.org",

    # Major international news outlets (previously missing)
    "cnn.com", "nytimes.com", "theguardian.com", "npr.org",
    "aljazeera.com", "washingtonpost.com", "wsj.com", "economist.com",
    "ft.com", "bloomberg.com", "time.com", "usatoday.com",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com", "pbs.org",
    "dw.com", "france24.com", "skynews.com",

    # International fact-checking (IFCN-certified)
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
    "factcheck.afp.com", "africacheck.org", "poynter.org",

    # South Asia -- India (IFCN-certified / established outlets)
    "boomlive.in", "factchecker.in", "factcrescendo.com", "factly.in",
    "vishvasnews.com", "newschecker.in", "thequint.com", "altnews.in",

    # South Asia -- Pakistan (IFCN-certified / established outlets)
    "sochfactcheck.com", "dawn.com", "geo.tv",

    # South Asia -- Bangladesh / Nepal
    "tbsnews.net", "southasiacheck.org",
]

# Official broadcaster / government channel names -- when a youtube.com result's
# title contains one of these, treat it as strong evidence the story is
# genuinely covered by an official source, not just "some video exists".
# Matched against the result title since youtube.com itself isn't a trust
# signal on its own (anyone can upload there).
OFFICIAL_YOUTUBE_PUBLISHERS = [
    # International
    "BBC News", "BBC World", "Al Jazeera English", "Reuters", "AP",
    "Associated Press", "CNN", "DW News", "France 24", "NBC News",
    "CBS News", "ABC News", "Sky News", "CNA",

    # South Asia -- Pakistan
    "Dawn News", "Geo News", "ARY News", "Samaa TV", "Hum News",
    "Express News", "92 News", "Bol News", "Dunya News",

    # South Asia -- India
    "NDTV", "India Today", "ANI", "PTI", "The Quint", "The Wire",

    # Official government / international bodies
    "United Nations", "WHO", "PTV News", "PID Pakistan",
]

GREEN = "\U0001F7E2"
YELLOW = "\U0001F7E1"
RED = "\U0001F534"
WHITE = "\u26AA"  # "unable to verify" -- distinct from Red Flag, means no
                  # signal either way, not "this looks false"

# Generic official-government suffixes -- checked against the actual domain
# (netloc), not a substring of the whole URL, so "mygovernment-scam.com"
# can never match. Covers .gov, .mil, and country-code government variants
# like .gov.uk, .gov.pk, .gov.in without needing to hardcode every single
# government website by name (the earlier bug: war.gov wasn't in the list).
GOV_SUFFIXES = (".gov", ".mil", ".gov.uk", ".gov.in", ".gov.pk", ".gov.au", ".gov.ca")

def is_trusted(link: str, title: str = "") -> bool:
    if any(domain in link for domain in TRUSTED_DOMAINS):
        return True

    try:
        netloc = urlparse(link).netloc.lower()
    except Exception:
        netloc = ""
    if netloc and any(netloc == suf.lstrip(".") or netloc.endswith(suf) for suf in GOV_SUFFIXES):
        return True

    if "youtube.com" in link and title:
        title_lower = title.lower()
        return any(pub.lower() in title_lower for pub in OFFICIAL_YOUTUBE_PUBLISHERS)
    return False

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
        return {"score": 0, "verdict": f"{WHITE} Unable to verify (no sources found)", "sources": []}

    true_weight = 0
    false_weight = 0
    trusted_count = 0
    true_count = 0
    false_count = 0
    sources = []

    TOP_N = 10  # increased from 5 -- search now returns more merged results

    for item in organic[:TOP_N]:
        link = item.get("link", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        combined_text = f"{title}. {snippet}"

        trusted = is_trusted(link, title)
        stance, confidence = check_stance_ai(claim, combined_text) if claim else ("neutral", 0)

        weight_multiplier = 2 if trusted else 1

        if trusted:
            trusted_count += 1
        if stance == "true":
            true_weight += confidence * weight_multiplier
            true_count += 1
        elif stance == "false":
            false_weight += confidence * weight_multiplier
            false_count += 1

        sources.append({
            "title": title,
            "link": link,
            "trusted": trusted,
            "stance": stance,
            "confidence": round(confidence, 2)
        })

    total_checked = min(len(organic), TOP_N)
    trust_ratio = (trusted_count / total_checked) if total_checked else 0

    # Average rather than sum -- prevents several weak/borderline signals from
    # stacking up past a single strong one. Bug found: previously summed
    # confidence across all sources unboundedly, letting 5 weak untrusted
    # "true" calls (~0.6 each) blow past the 100 cap on their own.
    avg_true = (true_weight / true_count) if true_count else 0
    avg_false = (false_weight / false_count) if false_count else 0

    if avg_false > avg_true and avg_false >= 0.55:
        score = max(0, 20 - int(avg_false * 15))
        verdict = f"{RED} Red Flag (likely false claim)"
    elif avg_true > avg_false and avg_true >= 0.55:
        if trusted_count == 0:
            # No trusted source backs this -- cap well below "Verified"
            # regardless of how confident the stance model was.
            score = min(50, 30 + int(avg_true * 15))
            verdict = f"{YELLOW} Unverified (no trusted sources found)"
        else:
            score = min(100, 65 + int(avg_true * 20) + int(trust_ratio * 15))
            verdict = f"{GREEN} Verified"
    elif avg_true > 0 and avg_false > 0:
        score = 40 + int(trust_ratio * 15)
        verdict = f"{YELLOW} Unverified (mixed signals)"
    elif true_count == 0 and false_count == 0:
        # Bug fixed: every source came back neutral (no stance model matched
        # true or false strongly). This is NOT the same as "likely false" --
        # it just means no source explicitly confirmed or contradicted the
        # specific claim wording. Previously this fell through to the same
        # scoring as a real red flag, which wrongly implied "looks false"
        # for things like accurately-transcribed neutral news coverage.
        if trusted_count > 0:
            score = 40 + int(trust_ratio * 20)
            verdict = f"{YELLOW} Covered by sources, but no explicit confirmation found"
        else:
            score = 25
            verdict = f"{WHITE} Unable to verify (no clear signal from available sources)"
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