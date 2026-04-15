# Evaluation Results

## Method

- **Model:** TextBlob lexicon-based polarity scoring.
- **Mapping:** `polarity > +0.05 → Positive`, `polarity < −0.05 → Negative`, otherwise `Neutral`.
- **Test set:** 12 hand-written English sentences (4 Positive, 4 Negative, 4 Neutral). The negative and neutral sets each include one deliberately tricky case (sarcasm and double-negation respectively) to probe known weaknesses of lexicon-based models.
- **Reproduce:** `python evaluate.py`. Raw output saved to `test_results.txt`.

## Headline numbers

| Metric | Value |
|---|---|
| Accuracy | **10 / 12 = 83.3%** |
| Positive recall | 4 / 4 = 100% |
| Negative recall | 3 / 4 = 75% |
| Neutral recall | 3 / 4 = 75% |

## Confusion matrix

Rows = actual class, columns = predicted class.

| actual ＼ pred | Positive | Negative | Neutral |
|---|---|---|---|
| **Positive** | 4 | 0 | 0 |
| **Negative** | 1 | 3 | 0 |
| **Neutral**  | 0 | 1 | 3 |

## Per-sentence results

| # | Expected | Predicted | Polarity | Sentence | Pass |
|---|---|---|---|---|---|
| 1 | Positive | Positive | +0.35 | I absolutely love how smooth the new update feels. | ✅ |
| 2 | Positive | Positive | +1.00 | Best customer service I've ever had — thank you! | ✅ |
| 3 | Positive | Positive | +0.80 | The food was delicious and the staff were so kind. | ✅ |
| 4 | Positive | Positive | +0.20 | She nailed the presentation and the whole room clapped. | ✅ |
| 5 | Negative | Negative | −1.00 | This is the worst app I have ever downloaded. | ✅ |
| 6 | Negative | Negative | −0.75 | I'm really disappointed with how this turned out. | ✅ |
| 7 | Negative | Negative | −0.50 | Waited 45 minutes and the order was still wrong. | ✅ |
| 8 | Negative | **Positive** | **+0.80** | Oh great, another Monday meeting that could've been an email. | ❌ |
| 9 | Neutral  | Neutral  | +0.00 | The package arrived on Tuesday afternoon. | ✅ |
| 10 | Neutral | Neutral  | +0.00 | The meeting is scheduled for 3pm in conference room B. | ✅ |
| 11 | Neutral | Neutral  | +0.00 | It's a film about a man who travels across the country. | ✅ |
| 12 | Neutral | **Negative** | **−0.10** | The movie wasn't terrible, but it wasn't great either. | ❌ |

---

## Analysis of incorrect predictions

### Failure 1 — Sarcasm

> *"Oh great, another Monday meeting that could've been an email."*
> Expected: **Negative** &nbsp;|&nbsp; Predicted: **Positive** &nbsp;|&nbsp; Polarity: **+0.80**

**What went wrong.** TextBlob's polarity is a weighted sum of per-word scores from its lexicon. The word "great" carries a very strong positive weight (+0.8), and there is nothing else in the sentence that the lexicon scores as negative. The result is a confidently *wrong* positive prediction.

**Why this is hard.** Sarcasm is a *pragmatic* phenomenon — the literal sentiment of the words is inverted by speaker intent, context, and shared knowledge ("Monday meetings that could've been an email" is a cultural complaint). A unigram lexicon has no representation of any of that.

**Possible fixes.**
- Swap the backend for a transformer trained on social-media data (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`), which sees enough sarcastic-but-positively-worded examples in training to start learning the inversion pattern.
- Add a lightweight sarcasm-cue rule (sentence-initial *"Oh great"*, *"Just what I needed"*, etc.) as a pre-processing step that flips the predicted label.
- Use an LLM (e.g. Claude) as a fallback classifier on examples where a separate sarcasm detector flags high probability.

### Failure 2 — Double negation with mixed sentiment

> *"The movie wasn't terrible, but it wasn't great either."*
> Expected: **Neutral** &nbsp;|&nbsp; Predicted: **Negative** &nbsp;|&nbsp; Polarity: **−0.10**

**What went wrong.** TextBlob does apply a simple negation rule (it flips the sign of a word's polarity when preceded by *"not"*), but the heuristic is shallow. Here it negates *"terrible"* and *"great"* and combines them, landing at polarity −0.10 — *just* past the −0.05 negative threshold, so the sentence is misclassified as Negative even though the intended meaning is "mediocre / neutral."

**Why this is hard.** The sentence expresses two negated opinions that should approximately cancel. Getting that right requires:
1. Correctly scoping each negation to the right adjective (TextBlob does this).
2. Weighting the two opposing clauses against each other (TextBlob does *not* — it just sums).
3. Recognising the discourse marker "but" as introducing a contrast that should be balanced rather than additive.

**Possible fixes.**
- Widen the neutral deadband (e.g. `±0.15`). On the eval set this would correctly reclassify sentence 12 as Neutral without breaking any other prediction. Trade-off: borderline-but-genuinely-mild opinions also get pushed to Neutral.
- Add a confidence-aware "Uncertain" output for any prediction whose polarity falls inside a tighter inner band (e.g. `±0.15`), and surface it to the caller for human review.
- Use a dependency parser to detect contrastive constructions (`X but Y`) and weight the second clause more heavily, mirroring how humans usually read these sentences.
- Move to a transformer that has learned these constructions implicitly from data.

---

## Reflection

For short, literal text — product reviews, simple feedback — TextBlob with a deadband is a reasonable, low-cost baseline (100% on the unambiguous Positive cases here). Its ceiling is set by the two failure modes above: it cannot model speaker intent (sarcasm) and it cannot weigh contrastive clauses against each other (mixed sentiment). For a production system handling user-generated content I would treat TextBlob as the fast path and route low-confidence or contrast-marker-containing inputs to a transformer or LLM-based classifier.
