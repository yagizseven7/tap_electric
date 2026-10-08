"""
OCR metrics.

  exact_match   share of IDs read completely right. A charger ID with one
                wrong character is the wrong charger, so this is the number
                that matters most.
  cer           Character Error Rate: how many characters must be changed
                (inserted, deleted or replaced) to turn the prediction into
                the truth, divided by the length of the truth.
                0.0 = perfect. 0.1 = 1 in 10 characters wrong.
                More forgiving than exact match, so it shows progress earlier
                and tells us whether a near-miss could still be matched
                against nearby chargers (Step 9).

Both compare IDs the way matching.py does (upper case, asterisks and spaces
ignored), so a metric only counts errors that would actually hurt.
"""

from app.inference.matching import comparison_key


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the minimum number of single-character edits.

    'E12345' -> 'E12845' is 1 (one replacement).
    Classic dynamic programming, keeping only two rows of the table.
    """
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                      # delete
                current[j - 1] + 1,                   # insert
                previous[j - 1] + (char_a != char_b), # replace (free if equal)
            ))
        previous = current
    return previous[-1]


def cer(predictions: list[str], targets: list[str]) -> float:
    """Corpus-level CER: total edits / total target characters."""
    preds = [comparison_key(p) for p in predictions]
    truths = [comparison_key(t) for t in targets]
    total_chars = sum(len(t) for t in truths)
    if total_chars == 0:
        return 0.0
    return sum(edit_distance(p, t) for p, t in zip(preds, truths)) / total_chars


def exact_match(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    hits = sum(comparison_key(p) == comparison_key(t) for p, t in zip(predictions, targets))
    return hits / len(targets)


def ocr_metrics(predictions: list[str], targets: list[str]) -> dict[str, float]:
    return {"cer": cer(predictions, targets), "exact_match": exact_match(predictions, targets)}
