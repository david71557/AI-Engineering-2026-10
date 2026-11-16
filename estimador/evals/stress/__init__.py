"""Stress test scaffolding for the conversational (CAG) pipeline — Session 6.

Instruments the conversational estimator under many turns and measures where
the context-augmented approach starts to degrade: latency, cost per turn and
recall drift.

- ``scenarios`` — three synthetic multi-turn profiles with fact-trackers
                  (``growing``, ``pivot``, ``contradiction``).

Next steps add ``metrics`` (latency/cost budgets + ``MemoryDriftMetric``) and
``run`` (the CLI runner that writes ``results.csv``).
"""
