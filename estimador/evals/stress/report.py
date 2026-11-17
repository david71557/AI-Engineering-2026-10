"""Turn the stress runner's CSV into the tables of ``REPORT.md`` (step 6).

Every number in the report comes from ``results.csv`` through this module,
so the tables can never drift from the data. The reading paragraphs are
written by hand; the tables are regenerated between two markers::

    <!-- stress-tables:start -->
    ... generated ...
    <!-- stress-tables:end -->

Run with::

    uv run python -m evals.stress.report --csv evals/stress/results.csv --write evals/stress/REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

START, END = "<!-- stress-tables:start -->", "<!-- stress-tables:end -->"
SCENARIO_ORDER = ["growing", "pivot", "contradiction"]
TOKENS_BUCKETS = [(0, 2_999), (3_000, 5_999), (6_000, 9_999), (10_000, 19_999), (20_000, None)]
CHECKPOINT_TURNS = [1, 2, 5, 6, 7, 8, 10, 15, 20]


def load_rows(csv_path: Path) -> tuple[list[dict], list[dict]]:
    """Return ``(ok_rows, error_rows)`` with numeric columns already parsed."""
    ok, errors = [], []
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                errors.append(row)
                continue
            for col in ("attachment_size_kb", "repeat", "turn_index", "tokens_in", "tokens_out", "latency_ms",
                        "wall_clock_ms", "messages_in_window", "summary_chars", "anchors_count",
                        "attachments_total_chars"):
                row[col] = int(row[col])
            row["cost_usd"] = float(row["cost_usd"])
            row["drift"] = None if row["memory_drift_passed"] == "" else row["memory_drift_passed"] == "True"
            ok.append(row)
    return ok, errors


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (same definition as ``run._percentile``)."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(1, math.ceil(pct / 100 * len(ordered))) - 1]


def _rate(flags: list[bool]) -> str:
    return f"{100 * sum(flags) / len(flags):.1f}" if flags else "—"


def _scenario_key(name: str) -> int:
    return SCENARIO_ORDER.index(name) if name in SCENARIO_ORDER else len(SCENARIO_ORDER)


def summary_table(rows: list[dict]) -> str:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["attachment_size_kb"])].append(r)
    out = [
        "| Scenario | KB | n | P50 latency ms | P95 latency ms | Total cost USD | Exact cache hit % | Semantic cache hit % | Mean recall % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (scenario, kb), group in sorted(groups.items(), key=lambda kv: (_scenario_key(kv[0][0]), kv[0][1])):
        lat = [r["latency_ms"] for r in group]
        drift = [r["drift"] for r in group if r["drift"] is not None]
        out.append(
            f"| {scenario} | {kb} | {len(group)} | {percentile(lat, 50):.0f} | {percentile(lat, 95):.0f} "
            f"| {sum(r['cost_usd'] for r in group):.4f} "
            f"| {_rate([r['cache_hit_kind'] == 'exact' for r in group])} "
            f"| {_rate([r['cache_hit_kind'] == 'semantic' for r in group])} "
            f"| {_rate(drift)} |"
        )
    return "\n".join(out)


def latency_vs_tokens(rows: list[dict]) -> str:
    out = ["| `tokens_in` bucket | n | Median latency ms | P95 latency ms |", "|---|---:|---:|---:|"]
    for lo, hi in TOKENS_BUCKETS:
        lat = [r["latency_ms"] for r in rows if r["tokens_in"] >= lo and (hi is None or r["tokens_in"] <= hi)]
        label = f"{lo:,}+" if hi is None else f"{lo:,}–{hi:,}"
        median = f"{statistics.median(lat):.0f}" if lat else "—"
        p95 = f"{percentile(lat, 95):.0f}" if lat else "—"
        out.append(f"| {label} | {len(lat)} | {median} | {p95} |")
    return "\n".join(out)


def cumulative_cost(rows: list[dict], size_kb: int = 0) -> str:
    """Cumulative cost per session, averaged across repeats, at checkpoint turns."""
    scenarios = sorted({r["scenario"] for r in rows if r["attachment_size_kb"] == size_kb}, key=_scenario_key)
    # (scenario, repeat) → {turn: cost}
    per_session: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["attachment_size_kb"] == size_kb:
            per_session[(r["scenario"], r["repeat"])][r["turn_index"]] = r["cost_usd"]
    out = ["| Turn | " + " | ".join(f"{s} USD" for s in scenarios) + " |", "|---:|" + "---:|" * len(scenarios)]
    for turn in CHECKPOINT_TURNS:
        cells = []
        for s in scenarios:
            sessions = [costs for (sc, _), costs in per_session.items() if sc == s and turn in costs]
            cells.append(
                f"{statistics.mean(sum(c for t, c in costs.items() if t <= turn) for costs in sessions):.4f}"
                if sessions else "—"
            )
        if any(c != "—" for c in cells):
            out.append(f"| {turn} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def drift_vs_turn(rows: list[dict]) -> str:
    """Recall of the turn-1 fact at each turn N, pooled across scenarios and repeats."""
    sizes = sorted({r["attachment_size_kb"] for r in rows})
    out = ["| Turn N | " + " | ".join(f"{kb} KB recall %" for kb in sizes) + " | all sizes recall % | n |",
           "|---:|" + "---:|" * (len(sizes) + 2)]
    for turn in sorted({r["turn_index"] for r in rows if r["drift"] is not None}):
        at_turn = [r for r in rows if r["turn_index"] == turn and r["drift"] is not None]
        per_size = [_rate([r["drift"] for r in at_turn if r["attachment_size_kb"] == kb]) for kb in sizes]
        out.append(f"| {turn} | " + " | ".join(per_size) + f" | {_rate([r['drift'] for r in at_turn])} | {len(at_turn)} |")
    return "\n".join(out)


def window_vs_turn(rows: list[dict], size_kb: int = 0) -> str:
    """Context shape per turn (size 0): shows when the window caps and the summary takes over."""
    out = ["| Turn | messages_in_window | summary_chars | anchors_count | tokens_in (mean) | latency ms (median) |",
           "|---:|---:|---:|---:|---:|---:|"]
    for turn in CHECKPOINT_TURNS:
        at = [r for r in rows if r["turn_index"] == turn and r["attachment_size_kb"] == size_kb]
        if at:
            out.append(
                f"| {turn} | {statistics.mean(r['messages_in_window'] for r in at):.0f} "
                f"| {statistics.mean(r['summary_chars'] for r in at):.0f} "
                f"| {statistics.mean(r['anchors_count'] for r in at):.1f} "
                f"| {statistics.mean(r['tokens_in'] for r in at):.0f} "
                f"| {statistics.median(r['latency_ms'] for r in at):.0f} |"
            )
    return "\n".join(out)


def build_tables(csv_path: Path) -> str:
    rows, errors = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"{csv_path} has no successful rows")
    sessions = {(r["scenario"], r["attachment_size_kb"], r["repeat"]) for r in rows}
    parts = [
        f"_Generated by `evals/stress/report.py` from `{csv_path.as_posix()}`: "
        f"{len(rows)} turns OK, {len(errors)} error rows, {len(sessions)} sessions, "
        f"max turn {max(r['turn_index'] for r in rows)}._",
        "### 1. Summary (scenario × attachment size)",
        summary_table(rows),
        "### 2a. latency_ms vs tokens_in",
        latency_vs_tokens(rows),
    ]
    for kb in sorted({r["attachment_size_kb"] for r in rows}):
        parts += [f"### 2b. Cumulative cost_usd vs turn_index (attachment {kb} KB, mean of repeats)",
                  cumulative_cost(rows, kb)]
    parts += [
        "### 2c. MemoryDriftMetric (turn-1 fact) vs N turns",
        drift_vs_turn(rows),
        "### 2d. Context shape vs turn (attachment 0 KB)",
        window_vs_turn(rows),
    ]
    if errors:
        # Error rows carry no turn_index: the failed turn is the last OK turn + 1.
        last_ok: dict[tuple[str, int, int], int] = defaultdict(int)
        for r in rows:
            key = (r["scenario"], r["attachment_size_kb"], r["repeat"])
            last_ok[key] = max(last_ok[key], r["turn_index"])
        lines = ["| Scenario | KB | Aborted at turn | Error |", "|---|---:|---:|---|"]
        for e in sorted(errors, key=lambda e: (_scenario_key(e["scenario"]), int(e["attachment_size_kb"]))):
            key = (e["scenario"], int(e["attachment_size_kb"]), int(e["repeat"]))
            msg = re.sub(r"[0-9a-f-]{36}", "<id>", e["error"]).replace("|", "/")[:140]
            lines.append(f"| {e['scenario']} | {key[1]} | {last_ok[key] + 1} | `{msg}` |")
        parts += [f"### Errors ({len(errors)} sessions aborted)", "\n".join(lines)]
    return "\n\n".join(parts)


def write_into(report_path: Path, tables: str) -> None:
    text = report_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{report_path} lacks the {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    report_path.write_text(f"{head}{START}\n\n{tables}\n\n{END}{tail}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("evals/stress/results.csv"))
    parser.add_argument("--write", type=Path, default=None, help="REPORT.md to update in place (between markers).")
    args = parser.parse_args(argv)
    tables = build_tables(args.csv)
    if args.write:
        write_into(args.write, tables)
        print(f"Tables written into {args.write}")
    else:
        print(tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
