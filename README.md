# Sentiment Analysis Mini Project

A small Python project that classifies English text as **Positive**, **Negative**, or **Neutral** using TextBlob's lexicon-based sentiment scoring.

Built for the TechX AI Engineering take-home assignment.

---

## What's in here

| File | Purpose |
|---|---|
| `sentiment.py` | Core module + CLI. Validates input, runs analysis, returns a structured result. |
| `app.py` | Optional FastAPI wrapper exposing `POST /analyze`. |
| `evaluate.py` | Runs the model on 12 labelled test sentences and prints accuracy + confusion matrix. |
| `test_validation.py` | Unit tests for input validation (6 cases). |
| `test_results.txt` | Saved output from the evaluation run. |
| `results.md` | Written analysis of results, including 2 incorrect predictions. |
| `requirements.txt` | Python dependencies. |

---

## Setup

```bash
git clone <repo-url>
cd sentiment-analysis-techx

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### CLI

```bash
python sentiment.py "I really love this product!"
# [Positive | polarity=+0.50 subjectivity=0.60] 'I really love this product!'
```

### Run the evaluation suite

```bash
python evaluate.py
```

Prints per-sentence predictions, overall accuracy, and a confusion matrix.

### Run the input-validation unit tests

```bash
python -m unittest test_validation.py -v
```

### Run the API (optional)

```bash
uvicorn app:app --reload
```

Then in a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "The food was amazing"}'
```

Interactive docs at `http://127.0.0.1:8000/docs`.

---

## Design notes

**Library choice - TextBlob.**
The brief allows TextBlob, NLTK, or Hugging Face. I chose TextBlob because I wanted something simple, lightweight, and easy to explain. It runs locally without needing a large model download, and the polarity score is easy to inspect when looking at why a sentence was classified a certain way. For a small take-home project, that felt like a practical choice.

I also liked that TextBlob works well as a baseline. It handles straightforward positive and negative sentences reasonably well, but it also has obvious weaknesses on things like sarcasm and more complex negation. That made it useful not only for building the classifier, but also for writing a more honest error analysis.

**Mapping polarity to 3 classes.**
TextBlob outputs a continuous polarity score instead of a final class label, so I added a small `+/-0.05` neutral band around zero. My reasoning was that sentences close to zero should usually be treated as `Neutral` instead of being forced into `Positive` or `Negative`, especially when they are more factual or only mildly opinionated.

```
polarity >  +0.05  -> Positive
polarity <  -0.05  -> Negative
otherwise          -> Neutral
```

**Input validation.**
I added explicit validation so the function fails clearly on bad input. `analyze()` raises `InvalidInputError` for non-string input and for empty or whitespace-only strings. The CLI exits with code `2` on validation failure, and the API returns HTTP `400`.

**Results.**
On the 12-sentence evaluation set the model scored **10/12 (83.3%)**. The two incorrect predictions were on sarcasm and double-negation, which are both cases where a simple lexicon-based approach struggles. I explain those mistakes in more detail in `results.md`.

---

## Possible extensions

- Swap the TextBlob backend for a Hugging Face transformer (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`) for stronger sarcasm and negation handling.
- Batch endpoint for analysing many texts in one request.
- Confidence-calibrated abstention: return `Uncertain` when polarity falls within a tighter deadband.
