"""Deterministic, binary metrics over a ``(GoldenCase, EstimationResult)`` pair.

Each metric returns ``score`` 1.0 (pass) or 0.0 (fail) plus a ``details``
string naming what went wrong. Keeping them in-tree — no LLM judge — means
the suite is cheap, reproducible and explains every failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.estimation import LOW_CONFIDENCE_THRESHOLD, OUT_OF_SCOPE_PREFIX, EstimationResult
from evals.dataset import GoldenCase


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: str


def _result(name: str, problems: list[str], ok_details: str) -> MetricResult:
    return MetricResult(
        name=name,
        score=0.0 if problems else 1.0,
        passed=not problems,
        details="; ".join(problems) or ok_details,
    )


class SchemaAdherenceMetric:
    """The schema validators already run inside the wrapper's re-prompt loop;
    this metric flags if any of those guarantees *regress* (e.g. a future
    change drops a validator or the Boss returns an unvalidated copy). It also
    checks the phase count against the expected range, when provided."""

    name = "schema_adherence"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        phase_cost = sum(p.cost_eur for p in result.phases)
        if phase_cost != result.total_cost_eur:
            problems.append(f"phases sum ({phase_cost}) != total_cost_eur ({result.total_cost_eur})")
        phase_weeks = sum(p.duration_weeks for p in result.phases)
        if phase_weeks != result.total_duration_weeks:
            problems.append(f"phases weeks ({phase_weeks}) != total_duration_weeks ({result.total_duration_weeks})")

        if result.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not result.summary.startswith(OUT_OF_SCOPE_PREFIX):
            problems.append(f"low confidence missing {OUT_OF_SCOPE_PREFIX!r} prefix")

        # The out-of-scope envelope is a single placeholder phase by contract.
        if case.expected_phase_count_range and not case.expected_out_of_scope:
            lo, hi = case.expected_phase_count_range
            if not lo <= len(result.phases) <= hi:
                problems.append(f"phase count {len(result.phases)} outside [{lo}, {hi}]")

        return _result(self.name, problems, "schema ok")


class CostBoundsMetric:
    """Sanity bound for absolute cost and duration. Out-of-scope cases expect
    the placeholder envelope: cost 0 and duration 1 week."""

    name = "cost_bounds"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        problems: list[str] = []

        if case.expected_out_of_scope:
            if result.total_cost_eur != 0 or result.total_duration_weeks != 1:
                problems.append(
                    "expected out-of-scope envelope (cost=0, duration=1), got "
                    f"cost={result.total_cost_eur}, weeks={result.total_duration_weeks}"
                )
        else:
            if result.summary.startswith(OUT_OF_SCOPE_PREFIX):
                problems.append("unexpected out-of-scope answer")
            if case.expected_cost_range_eur:
                lo, hi = case.expected_cost_range_eur
                if not lo <= result.total_cost_eur <= hi:
                    problems.append(f"cost {result.total_cost_eur} EUR outside [{lo}, {hi}]")
            if case.expected_duration_weeks_range:
                lo, hi = case.expected_duration_weeks_range
                if not lo <= result.total_duration_weeks <= hi:
                    problems.append(f"duration {result.total_duration_weeks}w outside [{lo}, {hi}]")

        return _result(self.name, problems, "cost & duration within bounds")


class ContentRecallMetric:
    """Lightweight recall: does the summary / phase prose mention what the
    user actually cared about? Every ``expected_in_summary`` term must appear,
    and at least one of ``expected_technologies_any_of``. Case-insensitive.
    Out-of-scope cases skip this."""

    name = "content_recall"

    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        if case.expected_out_of_scope:
            return MetricResult(self.name, 1.0, True, "skipped — case is out-of-scope by design")

        haystack = " ".join([result.summary] + [f"{p.name} {p.summary}" for p in result.phases]).lower()
        problems: list[str] = []

        missing = [term for term in case.expected_in_summary if term.lower() not in haystack]
        if missing:
            problems.append(f"missing in summary: {missing}")
        techs = case.expected_technologies_any_of
        if techs and not any(t.lower() in haystack for t in techs):
            problems.append(f"none of {techs} mentioned anywhere")

        return _result(self.name, problems, "expected content present")


_DEFAULT_METRICS = (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())
METRIC_NAMES = tuple(m.name for m in _DEFAULT_METRICS)


def run_all_metrics(case: GoldenCase, result: EstimationResult) -> list[MetricResult]:
    return [m.evaluate(case, result) for m in _DEFAULT_METRICS]
