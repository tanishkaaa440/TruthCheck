import os
import requests
from dotenv import load_dotenv
from trust_score import calculate_trust_score

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def _serper_search(query, num=20):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def search_claim(query):
    """
    Runs two searches and merges results:
    1. The claim as-is (broad coverage)
    2. The claim + 'fact check' (biased toward finding actual fact-checkers)
    Deduplicates by link. Increases both volume and quality of sources
    reaching trust_score.py, instead of relying on a single 10-result query.
    """
    primary = _serper_search(query, num=20)
    factcheck_biased = _serper_search(f"{query} fact check", num=20)

    seen_links = set()
    merged_organic = []

    for result_set in (primary, factcheck_biased):
        for item in result_set.get("organic", []):
            link = item.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                merged_organic.append(item)

    return {"organic": merged_organic}


if __name__ == "__main__":
    result = search_claim("Is the earth flat")
    trust = calculate_trust_score(result, "Is the earth flat")
    print(trust)