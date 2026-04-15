"""
Minimal FastAPI wrapper around the sentiment module.

Run with:
    uvicorn app:app --reload

Then:
    curl -X POST http://127.0.0.1:8000/analyze \
         -H "Content-Type: application/json" \
         -d '{"text": "I love this product"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment import analyze, InvalidInputError


app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify text as Positive, Negative or Neutral.",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="English text to classify.")


class AnalyzeResponse(BaseModel):
    text: str
    label: str
    polarity: float
    subjectivity: float


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = analyze(payload.text)
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AnalyzeResponse(
        text=result.text,
        label=result.label,
        polarity=result.polarity,
        subjectivity=result.subjectivity,
    )
