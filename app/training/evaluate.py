"""
Step 12: evaluating a model on the golden test set.

Every golden photo goes through the REAL pipeline (find the text line,
read it, match it against chargers near the driver), with the QR stages
switched off so the model is tested on every photo, not only on the few
where the QR code failed. We measure at three levels:

  1. READING    did the model read the ID correctly?
       exact_match      share of IDs read exactly right
       cer              character error rate
  2. MATCHING   what would the driver experience?
       resolved_rate    right charger found: started directly, or shown
                        FIRST in the "Is it one of these?" suggestions
       auto_match_rate  share started directly, without asking
       gps_baseline_rate  the same, with NO model: how often is simply the
                        nearest charger the right one? A model that doesn't
                        clearly beat this adds nothing over "chargers near
                        you" (we learned this the hard way: a model that
                        only guessed the ID format still put the right
                        charger SOMEWHERE in the list 98% of the time)
       wrong_match_rate share where the WRONG charger was started directly.
                        The most dangerous error: a driver could pay for
                        someone else's session. Must stay close to zero.
  3. SYSTEM     the model together with the QR stages
       qr_rate          share the QR code alone already solves
       rescue_rate      of the scans where the QR code FAILED: share the
                        model now resolves. This is the business value:
                        drivers who used to be stuck and now are helped.

Plus three views that a single average would hide:
  * slices        the same numbers per condition (dark, blurry, ...)
  * calibration   does "90% confident" really mean right 90% of the time?
  * latency       how long the model takes per scan (p50 / p95)
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.inference.enhance import to_gray
from app.inference.matching import comparison_key, extract_evse_id
from app.inference.model import TextRecognizer
from app.inference.pipeline import ResolveStatus, ScanPipeline
from app.storage.chargers import ChargerDirectory, distance_m
from app.training.golden import GoldenSet
from app.training.metrics import edit_distance

CONFIDENCE_BINS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]


@dataclass
class ExampleResult:
    """What happened to one golden photo."""
    scan_id: str
    label: str
    charger_id: str
    conditions: list[str]
    hard: bool
    read_text: str | None
    confidence: float | None
    exact: bool
    char_errors: int
    label_length: int
    status: str
    matched_charger_id: str | None
    candidate_ids: list[str]
    qr_decoded: bool
    qr_correct: bool
    latency_ms: float
    gps_nearest_id: str | None = None   # the nearest charger by GPS alone (the no-model baseline)

    @property
    def correct_match(self) -> bool:
        return self.status == ResolveStatus.MATCHED.value and self.matched_charger_id == self.charger_id

    @property
    def wrong_match(self) -> bool:
        return self.status == ResolveStatus.MATCHED.value and self.matched_charger_id != self.charger_id

    @property
    def resolved(self) -> bool:
        """Started directly, or suggested FIRST. Merely appearing somewhere in
        a list is not counted: GPS alone can produce such a list."""
        return self.correct_match or (self.status == ResolveStatus.CANDIDATES.value
                                      and self.candidate_ids[:1] == [self.charger_id])

    @property
    def suggested(self) -> bool:
        return self.correct_match or self.charger_id in self.candidate_ids

    @property
    def gps_correct(self) -> bool:
        return self.gps_nearest_id == self.charger_id

    @property
    def slices(self) -> list[str]:
        return ["all"] + (self.conditions or ["clear"]) + (["qr_failed"] if not self.qr_decoded else [])


@dataclass
class EvalReport:
    version: str
    golden_id: str
    examples: int
    metrics: dict[str, float]
    slices: dict[str, dict[str, float]]
    calibration: list[dict]
    suggested_min_confidence: float | None
    latency_ms: dict[str, float]
    results: list[ExampleResult] = field(default_factory=list, repr=False)

    def correctness(self) -> dict[str, bool]:
        """Per golden photo: did this model resolve it? (for the paired comparison)"""
        return {r.scan_id: r.resolved for r in self.results}

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("results")
        return data

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.version}.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        with open(out / f"{self.version}_examples.jsonl", "w", encoding="utf-8") as f:
            for r in self.results:  # every single prediction, for error analysis later
                f.write(json.dumps(asdict(r)) + "\n")
        return out / f"{self.version}.json"


# ---------------------------------------------------------------------------
# Running the model over the golden set
# ---------------------------------------------------------------------------

def evaluate_model(
    recognizer: TextRecognizer,
    golden: GoldenSet,
    chargers: ChargerDirectory,
    version: str | None = None,
    pipeline_settings: dict | None = None,
) -> EvalReport:
    """pipeline_settings: the same thresholds as production (min_match_score,
    min_margin, ...), so we measure what drivers would actually get."""
    pipeline = ScanPipeline(chargers, recognizer, **(pipeline_settings or {}))
    results = []
    for ex in golden.examples:
        image = golden.image(ex)

        # What the QR stages alone achieve (doesn't depend on the model)
        qr = pipeline.try_qr(to_gray(image))
        qr_correct = bool(qr) and comparison_key(extract_evse_id(qr[0]) or "") == comparison_key(ex.evse_id)

        # The model on this photo, through the real pipeline but without QR
        started = time.perf_counter()
        out = pipeline.resolve(image, ex.latitude, ex.longitude, ex.gps_accuracy_m, use_qr=False)
        latency = (time.perf_counter() - started) * 1000

        read = extract_evse_id(out.read_text or "") or (out.read_text or "")
        results.append(ExampleResult(
            scan_id=ex.scan_id, label=ex.evse_id, charger_id=ex.charger_id,
            conditions=list(ex.conditions), hard=ex.hard,
            read_text=out.read_text, confidence=out.ocr_confidence,
            exact=comparison_key(read) == comparison_key(ex.evse_id),
            char_errors=edit_distance(comparison_key(read), comparison_key(ex.evse_id)),
            label_length=len(comparison_key(ex.evse_id)),
            status=out.status.value,
            matched_charger_id=out.charger.charger_id if out.charger else None,
            candidate_ids=[m.charger.charger_id for m in out.candidates],
            qr_decoded=qr is not None, qr_correct=qr_correct, latency_ms=latency,
            gps_nearest_id=nearest_charger(chargers, ex.latitude, ex.longitude, ex.gps_accuracy_m),
        ))
    return build_report(version or getattr(recognizer, "version", "unknown"), golden.golden_id, results)


def nearest_charger(chargers: ChargerDirectory, latitude: float | None, longitude: float | None,
                    accuracy_m: float | None) -> str | None:
    """The no-model baseline: which charger is closest to the driver?"""
    if latitude is None or longitude is None:
        return None
    nearby = chargers.find_near(latitude, longitude, ScanPipeline._search_radius(accuracy_m))
    if not nearby:
        return None
    return min(nearby, key=lambda c: distance_m(latitude, longitude, c.latitude, c.longitude)).charger_id


# ---------------------------------------------------------------------------
# Turning per-photo results into numbers (pure functions: easy to test)
# ---------------------------------------------------------------------------

def summarize(results: list[ExampleResult]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    qr_failed = [r for r in results if not r.qr_decoded]
    confidences = [r.confidence for r in results if r.confidence is not None]
    return {
        "n": n,
        "exact_match": sum(r.exact for r in results) / n,
        "cer": sum(r.char_errors for r in results) / max(1, sum(r.label_length for r in results)),
        "resolved_rate": sum(r.resolved for r in results) / n,
        "suggested_rate": sum(r.suggested for r in results) / n,
        "gps_baseline_rate": sum(r.gps_correct for r in results) / n,
        "auto_match_rate": sum(r.status == ResolveStatus.MATCHED.value for r in results) / n,
        "wrong_match_rate": sum(r.wrong_match for r in results) / n,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "qr_rate": sum(r.qr_correct for r in results) / n,
        "rescue_rate": sum(r.resolved for r in qr_failed) / len(qr_failed) if qr_failed else None,  # None: no QR failures to rescue
        "qr_failed": len(qr_failed),
    }


def slice_metrics(results: list[ExampleResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[ExampleResult]] = {}
    for r in results:
        for name in r.slices:
            groups.setdefault(name, []).append(r)
    keep = ("n", "exact_match", "cer", "resolved_rate", "wrong_match_rate")
    return {name: {k: v for k, v in summarize(rs).items() if k in keep} for name, rs in sorted(groups.items())}


def calibration(results: list[ExampleResult], bins: list[float] = CONFIDENCE_BINS) -> tuple[list[dict], float]:
    """Group predictions by confidence and compare with how often they were right.

    Well calibrated: in the 0.9-0.95 bin about 92% are right. Over-confident:
    the model says 0.95 but is right only 60% of the time. Then confidence
    can't be used to decide when to start a session without asking.

    Returns the table and the Expected Calibration Error (ECE): the average
    gap between confidence and accuracy, weighted by bin size. 0 = perfect.
    """
    scored = [r for r in results if r.confidence is not None]
    table, ece = [], 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        in_bin = [r for r in scored if low <= r.confidence < high]
        if not in_bin:
            continue
        mean_conf = float(np.mean([r.confidence for r in in_bin]))
        accuracy = sum(r.exact for r in in_bin) / len(in_bin)
        table.append({"bin": f"{low:.2f}-{min(high, 1.0):.2f}", "n": len(in_bin),
                      "mean_confidence": mean_conf, "accuracy": accuracy})
        ece += len(in_bin) / len(scored) * abs(mean_conf - accuracy)
    return table, ece


def suggest_min_confidence(results: list[ExampleResult], target_precision: float = 0.99,
                           min_support: int = 20) -> float | None:
    """The lowest confidence threshold above which readings are right at least
    `target_precision` of the time (on at least `min_support` examples).

    This turns calibration into an action: set the pipeline's
    min_ocr_confidence to this value, and only readings that are almost
    always right will start a session without asking the driver.
    """
    scored = sorted((r for r in results if r.confidence is not None), key=lambda r: r.confidence, reverse=True)
    best, correct = None, 0
    for i, r in enumerate(scored, start=1):
        correct += r.exact
        if i >= min_support and correct / i >= target_precision:
            best = r.confidence
    return best


def build_report(version: str, golden_id: str, results: list[ExampleResult]) -> EvalReport:
    table, ece = calibration(results)
    latencies = [r.latency_ms for r in results]
    metrics = summarize(results)
    metrics["ece"] = ece
    return EvalReport(
        version=version, golden_id=golden_id, examples=len(results), metrics=metrics,
        slices=slice_metrics(results), calibration=table,
        suggested_min_confidence=suggest_min_confidence(results),
        latency_ms={"p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
                    "p95": float(np.percentile(latencies, 95)) if latencies else 0.0},
        results=results,
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_report(report: EvalReport) -> str:
    """A short human-readable summary (Markdown)."""
    m = report.metrics
    lines = [
        f"### {report.version} on {report.golden_id} ({report.examples} photos)",
        "",
        "| Reading | | Matching | | System | |",
        "|---|---|---|---|---|---|",
        f"| exact match | {m['exact_match']:.1%} | resolved | {m['resolved_rate']:.1%} | QR alone | {m['qr_rate']:.1%} |",
        f"| CER | {m['cer']:.1%} | no-model GPS baseline | {m['gps_baseline_rate']:.1%} | rescue rate | {_pct(m['rescue_rate'])} |",
        f"| calibration error | {m['ece']:.3f} | started directly | {m['auto_match_rate']:.1%} | latency p95 | {report.latency_ms['p95']:.0f} ms |",
        f"| | | **wrong charger** | **{m['wrong_match_rate']:.1%}** | | |",
        "",
        "| slice | n | exact | resolved | wrong |",
        "|---|---|---|---|---|",
    ]
    for name, s in report.slices.items():
        lines.append(f"| {name} | {s['n']} | {s['exact_match']:.1%} | {s['resolved_rate']:.1%} | {s['wrong_match_rate']:.1%} |")
    suggestion = report.suggested_min_confidence
    lines += ["", "Suggested min_ocr_confidence for 99% precision: "
              + (f"{suggestion:.2f}" if suggestion is not None else "none (the model never reaches 99%)")]
    return "\n".join(lines)
