import re

from app.schemas.estimation import StructureCheck

_OK_FINISH_REASONS = {"stop", "end_turn"}

_TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<task>[^|]+?)\s*\|\s*(?P<hours>[\d.,]+)\s*\|\s*(?P<cost>[\d.,\sEURer]+)\s*\|\s*$",
    re.MULTILINE,
)
_HEADER_ROW_RE = re.compile(r"\|\s*Task\s*\|\s*Hours\s*\|\s*Cost", re.IGNORECASE)
_SEPARATOR_ROW_RE = re.compile(r"^\|\s*[-: ]+\|", re.MULTILINE)
_TOTAL_HOURS_RE = re.compile(r"Total\s+hours[:\*\s]*([\d.,]+)", re.IGNORECASE)
_TOTAL_COST_RE = re.compile(r"Total\s+cost[:\*\s]*([\d.,]+)", re.IGNORECASE)


def _to_int(raw: str) -> int | None:
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else None


def evaluate_estimation_structure(text: str, finish_reason: str) -> StructureCheck:
    """Run the Level-1 structural checks against a generated estimation.

    Pure regex/parsing — no LLM call. Used to give the instructor and the
    students a quick, automatable signal on whether the model produced
    something well-formed.
    """
    has_title = bool(re.search(r"^##\s+\S", text, re.MULTILINE))
    has_breakdown_table = bool(_HEADER_ROW_RE.search(text))
    has_totals_section = bool(_TOTAL_HOURS_RE.search(text)) and bool(_TOTAL_COST_RE.search(text))
    has_team_section = bool(
        re.search(
            r"(Recommended\s+Team|Team(\s+composition)?|^\s*-\s+\d+\s+\w+\s+(Developer|Designer|Engineer))",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    has_duration_section = bool(
        re.search(r"(Estimated\s+Duration|Duration:|\bweeks?\b)", text, re.IGNORECASE)
    )

    sum_hours: int | None = None
    sum_cost: int | None = None
    rows = []
    if has_breakdown_table:
        running_h = 0
        running_c = 0
        for match in _TABLE_ROW_RE.finditer(text):
            task = match.group("task").strip().lower()
            if task in {"task", ""} or _SEPARATOR_ROW_RE.match(match.group(0)):
                continue
            h = _to_int(match.group("hours"))
            c = _to_int(match.group("cost"))
            if h is None or c is None:
                continue
            running_h += h
            running_c += c
            rows.append((h, c))
        if rows:
            sum_hours = running_h
            sum_cost = running_c

    m_h = _TOTAL_HOURS_RE.search(text)
    m_c = _TOTAL_COST_RE.search(text)
    declared_total_hours = _to_int(m_h.group(1)) if m_h else None
    declared_total_cost = _to_int(m_c.group(1)) if m_c else None

    hours_match: bool | None
    if sum_hours is not None and declared_total_hours is not None:
        hours_match = abs(sum_hours - declared_total_hours) <= 1
    else:
        hours_match = None

    cost_match: bool | None
    if sum_cost is not None and declared_total_cost is not None and declared_total_cost > 0:
        cost_match = abs(sum_cost - declared_total_cost) / declared_total_cost <= 0.02
    else:
        cost_match = None
    

    finish_reason_ok = finish_reason in _OK_FINISH_REASONS

    checks: list[bool] = [
        has_title,
        has_breakdown_table,
        has_totals_section,
        has_team_section,
        has_duration_section,
        bool(hours_match),
        bool(cost_match),
        finish_reason_ok,
    ]
    score = round(sum(checks) / len(checks), 3)

    issues: list[str] = []
    if not has_title:
        issues.append("Missing H2 project title")
    if not has_breakdown_table:
        issues.append("Missing breakdown table with 'Task | Hours | Cost' header")
    if not has_totals_section:
        issues.append("Missing totals section ('Total hours' / 'Total cost')")
    if not has_team_section:
        issues.append("Missing recommended team section")
    if not has_duration_section:
        issues.append("Missing estimated duration in weeks")
    if hours_match is False:
        issues.append(
            f"Total hours mismatch: declared {declared_total_hours} vs sum of rows {sum_hours}"
        )
    if cost_match is False:
        issues.append(
            f"Total cost mismatch: declared {declared_total_cost} EUR vs sum of rows {sum_cost} EUR"
        )
    if not finish_reason_ok:
        issues.append(f"Response truncated or unexpected finish_reason='{finish_reason}'")

    return StructureCheck(
        has_title=has_title,
        has_breakdown_table=has_breakdown_table,
        has_totals_section=has_totals_section,
        has_team_section=has_team_section,
        has_duration_section=has_duration_section,
        declared_total_hours=declared_total_hours,
        sum_row_hours=sum_hours,
        hours_match=hours_match,
        declared_total_cost=declared_total_cost,
        sum_row_cost=sum_cost,
        cost_match=cost_match,
        finish_reason_ok=finish_reason_ok,
        score=score,
        issues=issues,
    )
