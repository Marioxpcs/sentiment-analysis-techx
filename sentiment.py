"""
Sentiment analysis module.

Classifies English text as Positive, Negative, or Neutral using TextBlob's
lexicon-based polarity score.

Why TextBlob:
  - Lightweight, no model downloads, runs fully offline after `pip install`.
  - Polarity score is interpretable (range -1.0 to +1.0).
  - Listed in the assignment brief as an allowed library.

Class mapping (configurable thresholds):
  polarity >  +0.05  -> Positive
  polarity <  -0.05  -> Negative
  otherwise          -> Neutral

The +/-0.05 band is a common convention (also used by VADER) and avoids
labelling near-zero polarity as a strong sentiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textblob import TextBlob


POSITIVE_THRESHOLD: float = 0.05
NEGATIVE_THRESHOLD: float = -0.05

Sentiment = Literal["Positive", "Negative", "Neutral"]


@dataclass(frozen=True)
class SentimentResult:
    """Structured prediction with raw polarity + subjectivity for transparency."""
    text: str
    label: Sentiment
    polarity: float       # -1.0 (very negative) .. +1.0 (very positive)
    subjectivity: float   #  0.0 (objective)     ..  1.0 (subjective)

    def __str__(self) -> str:
        return (
            f"[{self.label:<8} | polarity={self.polarity:+.2f} "
            f"subjectivity={self.subjectivity:.2f}] {self.text!r}"
        )


class InvalidInputError(ValueError):
    """Raised when input fails validation."""


def _validate(text: object) -> str:
    """Validate input. Raises InvalidInputError on failure, returns cleaned text."""
    if not isinstance(text, str):
        raise InvalidInputError(
            f"Input must be a string, got {type(text).__name__}."
        )
    cleaned = text.strip()
    if not cleaned:
        raise InvalidInputError("Input text is empty or whitespace only.")
    return cleaned


def _classify(polarity: float) -> Sentiment:
    if polarity > POSITIVE_THRESHOLD:
        return "Positive"
    if polarity < NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze(text: str) -> SentimentResult:
    """
    Predict sentiment for a single piece of text.

    Args:
        text: English text to classify.

    Returns:
        SentimentResult with label, polarity and subjectivity scores.

    Raises:
        InvalidInputError: if the input is not a non-empty string.
    """
    cleaned = _validate(text)
    blob = TextBlob(cleaned)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    return SentimentResult(
        text=cleaned,
        label=_classify(polarity),
        polarity=round(polarity, 4),
        subjectivity=round(subjectivity, 4),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python sentiment.py "your text here"')
        sys.exit(1)

    try:
        result = analyze(" ".join(sys.argv[1:]))
        print(result)
    except InvalidInputError as e:
        print(f"Invalid input: {e}")
        sys.exit(2)
