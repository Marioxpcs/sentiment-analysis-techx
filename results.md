# Evaluation Results

## Method

- **Model:** TextBlob lexicon-based polarity scoring.
- **Mapping:** `polarity > +0.05 -> Positive`, `polarity < -0.05 -> Negative`, otherwise `Neutral`.
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

| actual / pred | Positive | Negative | Neutral |
|---|---|---|---|
| **Positive** | 4 | 0 | 0 |
| **Negative** | 1 | 3 | 0 |
| **Neutral**  | 0 | 1 | 3 |

## Per-sentence results

| # | Expected | Predicted | Polarity | Sentence | Pass |
|---|---|---|---|---|---|
| 1 | Positive | Positive | +0.35 | I absolutely love how smooth the new update feels. | OK |
| 2 | Positive | Positive | +1.00 | Best customer service I've ever had - thank you! | OK |
| 3 | Positive | Positive | +0.80 | The food was delicious and the staff were so kind. | OK |
| 4 | Positive | Positive | +0.20 | She nailed the presentation and the whole room clapped. | OK |
| 5 | Negative | Negative | -1.00 | This is the worst app I have ever downloaded. | OK |
| 6 | Negative | Negative | -0.75 | I'm really disappointed with how this turned out. | OK |
| 7 | Negative | Negative | -0.50 | Waited 45 minutes and the order was still wrong. | OK |
| 8 | Negative | **Positive** | **+0.80** | Oh great, another Monday meeting that could've been an email. | FAIL |
| 9 | Neutral  | Neutral  | +0.00 | The package arrived on Tuesday afternoon. | OK |
| 10 | Neutral | Neutral  | +0.00 | The meeting is scheduled for 3pm in conference room B. | OK |
| 11 | Neutral | Neutral  | +0.00 | It's a film about a man who travels across the country. | OK |
| 12 | Neutral | **Negative** | **-0.10** | The movie wasn't terrible, but it wasn't great either. | FAIL |

---

## Analysis of incorrect predictions

### Failure 1 - Sarcasm

> *"Oh great, another Monday meeting that could've been an email."*
> Expected: **Negative** | Predicted: **Positive** | Polarity: **+0.80**

**What went wrong.** TextBlob's polarity is a weighted sum of per-word scores from its lexicon. The word "great" carries a very strong positive weight (+0.8), and there is nothing else in the sentence that the lexicon scores as negative. The result is a confidently wrong positive prediction.

**Why this is hard.** Sarcasm is a pragmatic phenomenon: the literal sentiment of the words is inverted by speaker intent, context, and shared knowledge ("Monday meetings that could've been an email" is a common complaint). A unigram lexicon has no representation of any of that.

**Possible fixes.**
- Swap the backend for a transformer trained on social-media data (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`), which sees enough sarcastic-but-positively-worded examples in training to start learning the inversion pattern.
- Add a lightweight sarcasm-cue rule (sentence-initial "Oh great", "Just what I needed", etc.) as a pre-processing step that flips the predicted label.
- Use an LLM as a fallback classifier on examples where a separate sarcasm detector flags high probability.

### Failure 2 - Double negation with mixed sentiment

> *"The movie wasn't terrible, but it wasn't great either."*
> Expected: **Neutral** | Predicted: **Negative** | Polarity: **-0.10**

**What went wrong.** TextBlob does apply a simple negation rule by flipping the sign of a word's polarity when it is preceded by "not", but the heuristic is still shallow. Here it negates "terrible" and "great" and combines them, landing at polarity -0.10, which is just past the -0.05 negative threshold. That pushes the sentence into `Negative` even though the intended meaning is closer to "mediocre" or neutral.

**Why this is hard.** The sentence expresses two negated opinions that should roughly cancel out. Getting that right requires correctly scoping each negation, weighing the two clauses against each other, and recognizing that "but" introduces a contrast instead of just more sentiment to add up.

**Possible fixes.**
- Widen the neutral deadband (e.g. `+/-0.15`). On the eval set this would correctly reclassify sentence 12 as Neutral without breaking any other prediction. Trade-off: borderline but genuinely mild opinions also get pushed to Neutral.
- Add a confidence-aware `Uncertain` output for predictions whose polarity falls inside a tighter inner band, and surface that to the caller for human review.
- Use a dependency parser to detect contrastive constructions (`X but Y`) and weight the second clause more heavily.
- Move to a transformer that has learned these constructions implicitly from data.

---

## Reflection

This project helped me understand both the strengths and the limitations of a simple NLP baseline. For short and direct sentences, TextBlob works pretty well, and it was a good fit for a take-home assignment because it is easy to set up, easy to test, and easy to explain. I also liked that the polarity score makes the model's behavior more transparent, since I can look at the number and get a rough sense of why a prediction happened.

At the same time, the two incorrect predictions showed where this approach starts to break down. Sarcasm depends on tone and intent, not just the literal meaning of the words, and mixed statements like "wasn't terrible, but it wasn't great either" need more context-aware reasoning than a simple polarity sum can provide. One of the biggest takeaways for me was that getting reasonable results on straightforward examples is not too difficult, but handling edge cases is where sentiment analysis becomes much more challenging.

If I were extending this project, I would keep TextBlob as a baseline for comparison, but I would also want to test a transformer-based model and compare the trade-offs in accuracy, speed, and complexity. That would give a better sense of when a lightweight approach is enough and when a stronger model is worth the extra overhead.
