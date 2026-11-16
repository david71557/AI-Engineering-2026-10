"""Evaluation harness for the conversational estimator.

Three pieces:

- ``dataset.py``: typed ``GoldenCase`` records and a JSON loader.
- ``metrics.py``: three deterministic, binary metrics (schema adherence,
  cost bounds, content recall). No LLM judge: the suite tells you exactly
  *why* a case failed.
- ``run.py``: CLI that fires each case at one of the two modes (``actor`` →
  ``/sessions/{id}/estimate``, ``acb`` → ``/sessions/{id}/estimate-acb``) and
  prints a table per case plus a final tally.

``golden_dataset.json`` carries 16 cases mixing project types, scope sizes,
NDA/regulatory flavours, a technical audience, a tight deadline and two
out-of-scope cases (one vague, one adversarial).
"""

from evals.dataset import GoldenCase, load_dataset
from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    MetricResult,
    SchemaAdherenceMetric,
    run_all_metrics,
)

__all__ = [
    "ContentRecallMetric",
    "CostBoundsMetric",
    "GoldenCase",
    "MetricResult",
    "SchemaAdherenceMetric",
    "load_dataset",
    "run_all_metrics",
]
