"""Stress test scaffolding for the conversational (CAG) pipeline — Session 6.

Instruments the conversational estimator under many turns and measures where
the context-augmented approach starts to degrade: latency, cost per turn and
recall drift.

- ``scenarios`` — three synthetic multi-turn profiles with fact-trackers
                  (``growing``, ``pivot``, ``contradiction``).
- ``metrics``   — ``LatencyBudgetMetric``, ``CostBudgetMetric``,
                  ``MemoryDriftMetric``. They read ``TurnObservation`` + the
                  session snapshot, not ``EstimationResult`` — hence kept
                  apart from ``evals.metrics`` (``MetricResult`` is reused).
- ``fixtures``  — ``build_pdfs.py`` regenerates the synthetic attachments.

The next step adds ``run`` (the CLI runner that writes ``results.csv``).
"""
