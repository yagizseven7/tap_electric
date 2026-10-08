"""
Tests for Step 12: golden set, evaluation metrics and the promotion gate.

Most tests feed hand-made results into the pure functions (summarize,
calibration, decide), so every number can be checked by hand. One test
runs evaluate_model end to end with a fake model on real sticker images.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from app.storage.chargers import Charger, InMemoryChargerDirectory
from app.storage.model_registry import ARCHIVED, PRODUCTION, REJECTED, InMemoryModelRegistry, ModelVersion
from app.training.dataset import assign_split, station_key
from app.training.evaluate import (
    EvalReport, ExampleResult, build_report, calibration, evaluate_model, format_report, suggest_min_confidence,
    summarize,
)
from app.training.golden import GoldenExample, GoldenSet, build_golden_set, load_golden_set
from app.training.promotion import PromotionPolicy, apply_decision, decide, paired_bootstrap
from tests.helpers import FakeRecognizer, destroyed_qr, png_bytes, sticker

# ---------------------------------------------------------------------------
# Helpers: hand-made results
# ---------------------------------------------------------------------------


def result(i: int = 0, *, exact=True, status="matched", matched="CH-1", candidates=(), confidence=0.9,
           conditions=(), qr_decoded=True, char_errors=0) -> ExampleResult:
    return ExampleResult(
        scan_id=f"s{i}", label="NL*TNM*E12345*1", charger_id="CH-1", conditions=list(conditions), hard=not qr_decoded,
        read_text="NL*TNM*E12345*1" if exact else "NL*TNM*E12845*1", confidence=confidence, exact=exact,
        char_errors=char_errors, label_length=11, status=status, matched_charger_id=matched if status == "matched" else None,
        candidate_ids=list(candidates), qr_decoded=qr_decoded, qr_correct=qr_decoded, latency_ms=50.0,
    )


def report(version: str, results: list[ExampleResult], golden_id: str = "golden-test") -> EvalReport:
    return build_report(version, golden_id, results)


def resolved_results(pattern: list[bool], start: int = 0, **kwargs) -> list[ExampleResult]:
    """True -> right charger started; False -> nothing found."""
    return [result(start + i, status="matched" if ok else "not_found", exact=ok, **kwargs)
            for i, ok in enumerate(pattern)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_summarize_counts_each_outcome_correctly():
    results = [
        result(0),                                                    # right charger started
        result(1, exact=False, char_errors=1, status="candidates", candidates=["CH-1", "CH-2"]),  # right one suggested
        result(2, exact=False, char_errors=1, matched="CH-2"),        # WRONG charger started
        result(3, exact=False, char_errors=11, status="not_found", qr_decoded=False),  # nothing; QR also failed
    ]
    m = summarize(results)
    assert m["exact_match"] == 0.25
    assert m["cer"] == pytest.approx(13 / 44)
    assert m["resolved_rate"] == 0.5          # results 0 and 1
    assert m["auto_match_rate"] == 0.5        # results 0 and 2 started directly
    assert m["wrong_match_rate"] == 0.25      # result 2
    assert m["qr_rate"] == 0.75
    assert m["rescue_rate"] == 0.0            # the only QR failure wasn't rescued


def test_calibration_detects_overconfidence():
    honest = [result(i, confidence=0.92, exact=i < 92) for i in range(100)]          # 92% sure, 92% right
    overconfident = [result(i, confidence=0.92, exact=i < 40) for i in range(100)]   # 92% sure, 40% right
    _, ece_honest = calibration(honest)
    table, ece_over = calibration(overconfident)
    assert ece_honest == pytest.approx(0.0, abs=1e-9)
    assert ece_over == pytest.approx(0.52)
    assert table == [{"bin": "0.90-0.95", "n": 100, "mean_confidence": pytest.approx(0.92), "accuracy": 0.4}]


def test_suggested_threshold_is_where_readings_become_reliable():
    # Above 0.9 the model is always right; below it, often wrong.
    results = [result(i, confidence=0.9 + i / 1000, exact=True) for i in range(30)]
    results += [result(100 + i, confidence=0.5 + i / 100, exact=i % 2 == 0) for i in range(30)]
    assert suggest_min_confidence(results) == pytest.approx(0.9)
    assert suggest_min_confidence(results[30:]) is None  # never 99% reliable


def test_report_slices_and_markdown():
    rep = report("v1", [result(0, conditions=["dark"]), result(1, conditions=["dark"], exact=False,
                                                                status="not_found", qr_decoded=False)])
    assert rep.slices["dark"]["resolved_rate"] == 0.5
    assert rep.slices["qr_failed"]["n"] == 1
    assert "| dark | 2 |" in format_report(rep)


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def as_dict(pattern: list[bool]) -> dict[str, bool]:
    return {f"s{i}": ok for i, ok in enumerate(pattern)}


def test_bootstrap_identical_models_show_no_gain():
    same = as_dict([True, False] * 50)
    gain, low, high = paired_bootstrap(same, same)
    assert gain == 0 and low == 0 and high == 0


def test_bootstrap_clear_improvement_is_significant():
    champion = as_dict([True] * 50 + [False] * 50)
    challenger = as_dict([True] * 80 + [False] * 20)   # fixes 30 photos, breaks none
    gain, low, _ = paired_bootstrap(challenger, champion)
    assert gain == pytest.approx(0.3) and low > 0.2


def test_bootstrap_small_gain_on_few_photos_is_not_trusted():
    champion = as_dict([True] * 10 + [False] * 10)
    challenger = as_dict([False] + [True] * 11 + [False] * 8)  # +1 net on 20 photos: could be luck
    gain, low, _ = paired_bootstrap(challenger, champion)
    assert gain > 0 and low <= 0


# ---------------------------------------------------------------------------
# The promotion decision
# ---------------------------------------------------------------------------

LENIENT = PromotionPolicy(min_exact_match=0.5, min_resolved_rate=0.5, min_slice_size=10)


def checks(decision) -> dict[str, bool]:
    return {c.name: c.passed for c in decision.checks}


def test_first_model_needs_only_the_absolute_bars():
    good = report("v1", resolved_results([True] * 90 + [False] * 10))
    useless = report("v0", resolved_results([False] * 100))
    assert decide(good, None, LENIENT).promote
    rejected = decide(useless, None, LENIENT)
    assert not rejected.promote
    assert checks(rejected)["minimum exact match"] is False
    assert "REJECT v0" in rejected.explain()


def test_a_model_no_better_than_gps_is_rejected():
    # Resolves 90%, but simply picking the nearest charger would get 85%
    results = resolved_results([True] * 90 + [False] * 10)
    for r in results[:85]:
        r.gps_nearest_id = "CH-1"
    decision = decide(report("v1", results), None, LENIENT)
    assert checks(decision)["beats GPS-only baseline"] is False and not decision.promote


def test_wrong_charger_starts_block_promotion():
    risky = report("v2", [result(i) for i in range(95)] + [result(100 + i, matched="CH-9") for i in range(5)])
    decision = decide(risky, None, LENIENT)
    assert checks(decision)["wrong charger rate"] is False and not decision.promote


def test_clearly_better_challenger_is_promoted():
    champion = report("v1", resolved_results([True] * 60 + [False] * 40))
    challenger = report("v2", resolved_results([True] * 90 + [False] * 10))
    decision = decide(challenger, champion, LENIENT)
    assert decision.promote, decision.explain()


def test_not_clearly_better_is_rejected():
    champion = report("v1", resolved_results([True] * 80 + [False] * 20))
    challenger = report("v2", resolved_results([True] * 81 + [False] * 19))
    decision = decide(challenger, champion, LENIENT)
    assert checks(decision)["better than champion"] is False


def test_better_on_average_but_worse_on_dark_photos_is_rejected():
    # Champion: bright photos 50%, dark photos 100%.
    champion = report("v1", resolved_results([True] * 20 + [False] * 20)
                      + resolved_results([True] * 20, start=40, conditions=["dark"]))
    # Challenger: bright photos 100% (much better on average), dark photos 70% (worse).
    challenger = report("v2", resolved_results([True] * 40)
                        + resolved_results([True] * 14 + [False] * 6, start=40, conditions=["dark"]))
    decision = decide(challenger, champion, LENIENT)
    assert checks(decision)["better than champion"] is True
    assert checks(decision)["no slice worse"] is False
    assert not decision.promote
    assert "dark 100% -> 70%" in decision.explain()


def test_reports_from_different_golden_sets_cannot_be_compared():
    a = report("v1", resolved_results([True] * 10), golden_id="golden-a")
    b = report("v2", resolved_results([True] * 10), golden_id="golden-b")
    with pytest.raises(ValueError):
        decide(b, a)


def test_apply_decision_updates_the_registry():
    registry = InMemoryModelRegistry()
    for version in ("v1", "v2", "v3"):
        registry.register(ModelVersion(version, f"models/{version}"))
    registry.set_stage("v1", PRODUCTION)

    good = report("v2", resolved_results([True] * 100))
    apply_decision(registry, decide(good, None, LENIENT))
    bad = report("v3", resolved_results([False] * 100))
    apply_decision(registry, decide(bad, None, LENIENT))

    assert registry.production().version == "v2"
    assert registry.get("v1").stage == ARCHIVED
    assert registry.get("v3").stage == REJECTED


# ---------------------------------------------------------------------------
# Golden set and evaluate_model on real images
# ---------------------------------------------------------------------------

def test_golden_set_contains_only_test_stations_and_is_frozen(synthetic_world, tmp_path):
    scans = synthetic_world.repo.get_labeled_scans()
    golden = build_golden_set(scans, synthetic_world.store, tmp_path / "golden", test_percent=30)

    assert golden.examples
    assert all(assign_split(station_key(e.evse_id), 10, 30) == "test" for e in golden.examples)
    assert all((golden.directory / e.image_file).exists() for e in golden.examples)
    assert all(e.latitude is not None for e in golden.examples)  # location kept for matching

    assert load_golden_set(tmp_path / "golden").examples == golden.examples
    with pytest.raises(FileExistsError):  # never silently replaced
        build_golden_set(scans, synthetic_world.store, tmp_path / "golden")


@pytest.fixture
def two_connector_golden(tmp_path) -> tuple[GoldenSet, InMemoryChargerDirectory]:
    """One station with connectors *1 and *2 at the same spot, plus another
    station. The QR codes of the first two photos are worn away."""
    chargers = [Charger("CH-1", "NL*TNM*E12345*1", 52.37, 4.89), Charger("CH-2", "NL*TNM*E12345*2", 52.37, 4.89),
                Charger("CH-3", "NL*TNM*E99871*1", 52.3702, 4.8903)]
    (tmp_path / "images").mkdir()
    examples = []
    for i, (charger, image) in enumerate([(chargers[0], destroyed_qr("NL*TNM*E12345*1")),
                                          (chargers[1], destroyed_qr("NL*TNM*E12345*2")),
                                          (chargers[2], sticker("NL*TNM*E99871*1"))]):
        file = f"images/{i}.png"
        (tmp_path / file).write_bytes(png_bytes(image))
        examples.append(GoldenExample(f"s{i}", file, charger.evse_id, charger.charger_id,
                                      52.37, 4.89, 10.0, hard=i < 2, conditions=[]))
    return GoldenSet(tmp_path, "golden-test", examples), InMemoryChargerDirectory(chargers)


def test_evaluation_catches_a_wrong_connector_start(two_connector_golden, tmp_path):
    golden, chargers = two_connector_golden
    # The fake model reads "*1" on every photo, 90% confident
    rep = evaluate_model(FakeRecognizer("NL*TNM*E12345*1", version="fake"), golden, chargers)
    m = rep.metrics
    by_id = {r.scan_id: r for r in rep.results}

    assert m["exact_match"] == pytest.approx(1 / 3)      # only photo 0 really says *1
    assert [r.qr_decoded for r in rep.results] == [False, False, True]
    # Photo 1 says *2, but the read "*1" matches CH-1 exactly, and the margin
    # over CH-2 (one character in eleven: 0.09) passes the default margin rule
    # (0.08). With default settings the WRONG connector would be started.
    assert by_id["s0"].correct_match
    assert by_id["s1"].wrong_match
    # Photo 2 shows ANOTHER station 30 m away, but "*1" exists nearby: wrong again
    assert by_id["s2"].wrong_match
    assert m["wrong_match_rate"] == pytest.approx(2 / 3)
    assert not decide(rep, None, PromotionPolicy(min_exact_match=0, min_resolved_rate=0)).promote
    # No-model baseline: all photos were taken at CH-1's spot, so "nearest" is CH-1
    assert m["gps_baseline_rate"] == pytest.approx(1 / 3)

    saved = json.loads(rep.save(tmp_path / "reports").read_text())
    assert saved["version"] == "fake" and "results" not in saved


def test_stricter_confidence_threshold_prevents_the_wrong_start(two_connector_golden):
    golden, chargers = two_connector_golden
    # The fix Step 12 recommends: only start without asking when the model is
    # more confident than a calibrated threshold. Below it, the driver chooses.
    rep = evaluate_model(FakeRecognizer("NL*TNM*E12345*1", version="fake"), golden, chargers,
                         pipeline_settings={"min_ocr_confidence": 0.95})
    by_id = {r.scan_id: r for r in rep.results}
    assert by_id["s0"].status == by_id["s1"].status == "candidates"
    assert by_id["s0"].resolved                                # CH-1 suggested first: right
    assert by_id["s1"].suggested and not by_id["s1"].resolved  # CH-2 only second: doesn't count
    assert rep.metrics["wrong_match_rate"] == 0.0
    assert rep.metrics["rescue_rate"] == 0.5                   # 1 of the 2 QR failures resolved


def test_same_inputs_give_the_same_report():
    results = [result(i, confidence=float(c)) for i, c in enumerate(np.linspace(0.5, 1.0, 30))]
    a, b = report("v", results), report("v", [replace(r) for r in results])
    assert a.to_dict() == b.to_dict()
    assert a.metrics["rescue_rate"] is None  # no QR failures: nothing to rescue
