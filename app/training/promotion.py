"""
Step 12: should the new model (challenger) replace the current one (champion)?

A candidate is promoted to production only if it passes EVERY check:

  1. Minimum quality     good enough on its own, regardless of the champion,
                         AND clearly better than using no model at all (just
                         picking the nearest charger by GPS).
                         (Lesson: a useless model that always wrote the same
                         "average" ID still beat a model that read nothing.)
  2. Safety              almost never starts a session on the WRONG charger.
  3. Better              clearly better than the champion: the improvement must
                         be larger than what luck in the choice of test photos
                         could explain (paired bootstrap, below).
  4. No slice regression not worse on any condition (dark, blurry, ...) that has
                         enough examples. An average can improve while one group
                         of drivers gets worse; this check catches that.
  5. Fast enough         95% of scans answered within the latency budget.

Each check reports its numbers, so a rejected model comes with the reason.

Passing offline validation is not the end. In production the new model
would first run in SHADOW mode: it reads every real failed scan next to the
champion, but only the champion's answer is shown to drivers. If the shadow
results agree with the offline results for a week, it takes over; the
manual-entry rate after failed scans is then monitored, and the previous
model stays in the registry for an instant rollback.
"""

from dataclasses import dataclass, field

import numpy as np

from app.storage.model_registry import PRODUCTION, REJECTED, ModelRegistry
from app.training.evaluate import EvalReport


@dataclass
class PromotionPolicy:
    min_exact_match: float = 0.80        # 1. reads at least 80% of IDs exactly right
    min_resolved_rate: float = 0.85      # 1. finds the right charger in 85% of photos
    min_gain_over_gps: float = 0.10      # 1. and 10 points more often than GPS alone
    max_wrong_match_rate: float = 0.01   # 2. at most 1 in 100 photos starts the wrong charger
    min_gain: float = 0.0                # 3. resolved-rate gain over the champion...
    confidence_level: float = 0.95       # 3. ...that must hold with 95% confidence
    slice_tolerance: float = 0.03        # 4. a slice may drop at most 3 percentage points
    min_slice_size: int = 20             # 4. smaller slices are too noisy to judge
    max_p95_latency_ms: float = 2000.0   # 5. latency budget


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class PromotionDecision:
    challenger: str
    champion: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return all(c.passed for c in self.checks)

    def explain(self) -> str:
        verdict = "PROMOTE" if self.promote else "REJECT"
        lines = [f"{verdict} {self.challenger} (champion: {self.champion or 'none'})"]
        lines += [f"  [{'pass' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(lines)


def paired_bootstrap(challenger: dict[str, bool], champion: dict[str, bool], samples: int = 2000,
                     confidence_level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """How sure are we that the challenger is really better, and not just lucky?

    Both models were tested on the SAME photos, so we compare them photo by
    photo ("paired"). For each photo: +1 if only the challenger got it, -1 if
    only the champion did, 0 if both or neither. The average is the gain.

    Bootstrap: draw the test photos again at random (with replacement)
    thousands of times, recompute the gain each time, and look at the spread.
    If even the pessimistic end of the spread (the lower bound) is above 0,
    the improvement is real. With few photos the spread is wide, so a small
    gain doesn't count; that's the honest answer.

    Returns (gain, lower bound, upper bound).
    """
    shared = sorted(set(challenger) & set(champion))
    if not shared:
        return 0.0, 0.0, 0.0
    diff = np.array([int(challenger[k]) - int(champion[k]) for k in shared], dtype=float)
    rng = np.random.default_rng(seed)
    resampled = diff[rng.integers(0, len(diff), size=(samples, len(diff)))].mean(axis=1)
    tail = (1 - confidence_level) / 2 * 100
    return float(diff.mean()), float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100 - tail))


def decide(challenger: EvalReport, champion: EvalReport | None,
           policy: PromotionPolicy | None = None) -> PromotionDecision:
    policy = policy or PromotionPolicy()
    if champion is not None and champion.golden_id != challenger.golden_id:
        raise ValueError("Both models must be evaluated on the same golden set to be compared")

    m = challenger.metrics
    decision = PromotionDecision(challenger.version, champion.version if champion else None)
    add = decision.checks.append

    # 1. Minimum quality
    add(Check("minimum exact match", m["exact_match"] >= policy.min_exact_match,
              f"{m['exact_match']:.1%} (need >= {policy.min_exact_match:.0%})"))
    add(Check("minimum resolved rate", m["resolved_rate"] >= policy.min_resolved_rate,
              f"{m['resolved_rate']:.1%} (need >= {policy.min_resolved_rate:.0%})"))
    gps = m.get("gps_baseline_rate", 0.0)
    add(Check("beats GPS-only baseline", m["resolved_rate"] >= gps + policy.min_gain_over_gps,
              f"{m['resolved_rate']:.1%} vs nearest-charger {gps:.1%} (need +{policy.min_gain_over_gps:.0%})"))

    # 2. Safety
    wrong_ok = m["wrong_match_rate"] <= policy.max_wrong_match_rate
    detail = f"{m['wrong_match_rate']:.2%} (max {policy.max_wrong_match_rate:.0%})"
    if champion is not None:
        wrong_ok = wrong_ok and m["wrong_match_rate"] <= champion.metrics["wrong_match_rate"] + 0.005
        detail += f", champion {champion.metrics['wrong_match_rate']:.2%}"
    add(Check("wrong charger rate", wrong_ok, detail))

    if champion is not None:
        # 3. Clearly better
        gain, low, high = paired_bootstrap(challenger.correctness(), champion.correctness(),
                                           confidence_level=policy.confidence_level)
        add(Check("better than champion", low > policy.min_gain,
                  f"resolved {gain:+.1%} ({policy.confidence_level:.0%} interval {low:+.1%} to {high:+.1%})"))

        # 4. No slice regression
        worse = []
        for name, theirs in champion.slices.items():
            ours = challenger.slices.get(name)
            if ours is None or min(ours["n"], theirs["n"]) < policy.min_slice_size:
                continue
            drop = theirs["resolved_rate"] - ours["resolved_rate"]
            if drop > policy.slice_tolerance:
                worse.append(f"{name} {theirs['resolved_rate']:.0%} -> {ours['resolved_rate']:.0%}")
        add(Check("no slice worse", not worse, "; ".join(worse) or "all slices within tolerance"))

    # 5. Latency
    p95 = challenger.latency_ms["p95"]
    add(Check("latency p95", p95 <= policy.max_p95_latency_ms, f"{p95:.0f} ms (max {policy.max_p95_latency_ms:.0f})"))
    return decision


def apply_decision(registry: ModelRegistry, decision: PromotionDecision) -> None:
    """Record the outcome in the registry. Promoting archives the old champion
    in the same step, so there is always exactly one production model.
    (The full reports are saved as files next to this decision.)"""
    registry.set_stage(decision.challenger, PRODUCTION if decision.promote else REJECTED)
