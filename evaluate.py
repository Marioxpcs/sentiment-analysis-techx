"""
Evaluation harness.

Runs the sentiment model on a held-out set of 12 sentences (4 positive,
4 negative, 4 neutral) and prints a confusion matrix + per-sentence report.
"""

from __future__ import annotations

from collections import defaultdict

from sentiment import analyze, Sentiment


# 12 test sentences. A few are deliberately tricky (sarcasm, negation,
# mixed sentiment) to surface model weaknesses.
TEST_CASES: list[tuple[str, Sentiment]] = [
    # Positive (4)
    ("I absolutely love how smooth the new update feels.",        "Positive"),
    ("Best customer service I've ever had — thank you!",          "Positive"),
    ("The food was delicious and the staff were so kind.",        "Positive"),
    ("She nailed the presentation and the whole room clapped.",   "Positive"),

    # Negative (4)
    ("This is the worst app I have ever downloaded.",             "Negative"),
    ("I'm really disappointed with how this turned out.",         "Negative"),
    ("Waited 45 minutes and the order was still wrong.",          "Negative"),
    ("Oh great, another Monday meeting that could've been an email.",  "Negative"),  # sarcasm

    # Neutral (4)
    ("The package arrived on Tuesday afternoon.",                 "Neutral"),
    ("The meeting is scheduled for 3pm in conference room B.",    "Neutral"),
    ("It's a film about a man who travels across the country.",   "Neutral"),
    ("The movie wasn't terrible, but it wasn't great either.",    "Neutral"),  # mixed/negated
]


def run_evaluation() -> None:
    correct = 0
    rows: list[tuple[str, str, str, float, bool]] = []
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    for text, expected in TEST_CASES:
        result = analyze(text)
        is_correct = result.label == expected
        correct += int(is_correct)
        confusion[(expected, result.label)] += 1
        rows.append((text, expected, result.label, result.polarity, is_correct))

    # Per-sentence report
    print("=" * 95)
    print("PER-SENTENCE RESULTS")
    print("=" * 95)
    for text, expected, predicted, polarity, ok in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  expected={expected:<8}  predicted={predicted:<8}  "
              f"polarity={polarity:+.2f}  | {text}")

    # Summary
    total = len(TEST_CASES)
    accuracy = correct / total
    print()
    print("=" * 90)
    print(f"ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print("=" * 90)

    # Confusion matrix
    classes = ["Positive", "Negative", "Neutral"]
    print("\nCONFUSION MATRIX (rows = actual, cols = predicted)\n")
    header = "actual \\ pred".ljust(16) + "".join(c.ljust(12) for c in classes)
    print(header)
    for actual in classes:
        row = actual.ljust(16)
        for pred in classes:
            row += str(confusion[(actual, pred)]).ljust(12)
        print(row)


if __name__ == "__main__":
    run_evaluation()
