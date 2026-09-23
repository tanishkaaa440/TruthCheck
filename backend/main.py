from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search_engine import search_claim
from trust_score import calculate_trust_score
from video_check import process_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    text: str

class VideoRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"status": "TruthCheck backend alive"}

@app.post("/verify")
def verify_claim(request: ClaimRequest):
    search_results = search_claim(request.text)
    result = calculate_trust_score(search_results, request.text)
    return result

@app.post("/verify-video")
def verify_video(request: VideoRequest):
    """
    Phase 7: full video verification.
    Downloads the video, runs deepfake frame analysis, transcribes speech,
    and fact-checks the transcribed claim -- all in one response.
    """
    result = process_video(request.url)
    return result