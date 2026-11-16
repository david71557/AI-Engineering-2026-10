"""Boss orchestrator — coordinates Actor and Critic.

The Boss is a tiny state machine:

    actor_call → critic_review → decide
                                  ├── accept      → return
                                  ├── needs_iteration AND iterations_left
                                  │                  → re-call actor with feedback → loop
                                  └── reject, or out of iterations
                                                   → synthesize fallback

The Boss makes NO LLM calls of its own — it only chooses what to do next.
That keeps the decision logic provable: every choice is traceable in
``BossTrace.iterations`` and reproducible from the inputs.

The Actor is passed as a callable so the Boss can re-invoke it at iteration
N+1 with the feedback from iteration N::

    actor(critic_feedback: CriticFeedback | None) -> EstimationResult

Wiring lives in ``EstimationService.estimate_with_acb``.
"""

from __future__ import annotations

from typing import Callable

import structlog
from pydantic import ValidationError

from app.schemas.acb import ACBIteration, BossDecision, BossTrace
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import LOW_CONFIDENCE_THRESHOLD, EstimationResult

log = structlog.get_logger()

ActorCallable = Callable[[CriticFeedback | None], EstimationResult]
CriticCallable = Callable[[EstimationResult], CriticFeedback]

_SUMMARY_MAX_CHARS = 1200  # EstimationResult.summary max_length


class Boss:
    """Stateless orchestrator. One ``run`` per estimation."""

    def __init__(self, *, max_iterations: int = 3) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def run(self, *, actor: ActorCallable, critic: CriticCallable) -> tuple[EstimationResult, BossTrace]:
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None
        current: EstimationResult | None = None

        for iteration in range(self.max_iterations):
            current = actor(feedback)
            review = critic(current)
            decision = self._decide(review, iterations_left=self.max_iterations - iteration - 1)

            trace.iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[f"[{i.severity}] {i.category} @ {i.field_path}" for i in review.issues[:5]],
                )
            )
            trace.iterations_run = iteration + 1
            log.info("boss_iteration", iteration=iteration, decision=decision, verdict=review.verdict, issues=len(review.issues))

            if decision == "accept":
                trace.final_decision = "accept"
                return current, trace
            if decision == "synthesize":
                trace.final_decision = "synthesize"
                return self._synthesize_fallback(current, review), trace
            feedback = review  # "iterate": loop with the feedback

        # Unreachable in practice (_decide returns "synthesize" on the last
        # iteration), kept as a safety net.
        trace.final_decision = "synthesize"
        assert current is not None  # max_iterations >= 1
        return self._synthesize_fallback(current, feedback), trace

    # -- decision helpers -------------------------------------------------

    @staticmethod
    def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "reject":
            return "synthesize"
        return "iterate" if iterations_left > 0 else "synthesize"

    @staticmethod
    def _synthesize_fallback(last_result: EstimationResult, last_feedback: CriticFeedback | None) -> EstimationResult:
        """When the Boss cannot accept, return the actor's last draft annotated
        with the Critic's open issues and a reduced confidence — never an
        empty envelope.

        Discarding the draft would hide the best available answer behind an
        "Out of scope" placeholder. The honest signal that the loop did not
        converge already travels in ``BossTrace.final_decision == "synthesize"``;
        the user is better served by an estimation with caveats than by
        zeroed-out numbers.
        """
        # Floor at the low-confidence threshold: going below it would contradict
        # the schema rule that demands an "Out of scope:" summary. A draft that
        # is already out of scope keeps its own (low) confidence.
        if last_result.confidence_pct < LOW_CONFIDENCE_THRESHOLD:
            reduced_confidence = last_result.confidence_pct
        else:
            reduced_confidence = max(LOW_CONFIDENCE_THRESHOLD, last_result.confidence_pct // 2)

        if last_feedback is None or not last_feedback.issues:
            candidates = ["⚠ Open caveats from independent review: (no detail available)\n\n"]
        else:
            header = "⚠ Open caveats from independent review (loop did not fully converge):\n"
            detailed = [
                f"- [{i.severity}] {i.category} ({i.field_path}): {i.description}" for i in last_feedback.issues
            ]
            compact = [f"- [{i.severity}] {i.category} ({i.field_path})" for i in last_feedback.issues]
            candidates = [header + "\n".join(detailed) + "\n\n", header + "\n".join(compact) + "\n\n"]

        # estimador's ``summary_matches_totals`` checks every EUR amount and
        # week count quoted in the summary. A Critic description such as "the
        # summary says 30,000 EUR" would make the annotated draft invalid (and
        # FastAPI re-validates the response), so fall back to the compact
        # caveats (no numbers), and finally to the untouched summary.
        for caveats in candidates:
            candidate = last_result.model_copy(
                update={
                    "summary": (caveats + last_result.summary)[:_SUMMARY_MAX_CHARS],
                    "confidence_pct": reduced_confidence,
                }
            )
            try:
                EstimationResult.model_validate(candidate.model_dump())
            except ValidationError:
                log.info("boss_caveats_rejected_by_schema", caveats_chars=len(caveats))
                continue
            return candidate
        return last_result.model_copy(update={"confidence_pct": reduced_confidence})
