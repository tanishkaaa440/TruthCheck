import os
import requests
from dotenv import load_dotenv
from trust_score import calculate_trust_score

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def search_claim(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

if __name__ == "__main__":
    result = search_claim("Is the earth flat")
    trust = calculate_trust_score(result)
    print(trust)