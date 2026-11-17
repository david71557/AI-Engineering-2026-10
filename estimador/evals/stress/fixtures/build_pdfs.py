"""Generate calibrated synthetic PDFs for the stress runner (Session 6, step 3).

Produces ``attach_{5,20,50,100}kb.pdf`` under this directory, where the
suffix refers to the *extracted text size* (what ``pypdf`` will pull out)
rather than the file size on disk — the runner cares about the prompt
budget, not the byte count. ``attach_0kb`` does not exist: 0 KB means "no
attachment" and the runner simply sends none.

The generator is deterministic: same paragraph, same number of repeats per
target and a fixed creation date (fpdf2 stamps "now" by default, which would
change the bytes on every run). Re-running it yields byte-identical PDFs, so
they are gitignored and only this script is committed. The runner calls
``ensure_pdf`` to regenerate any missing fixture on demand.

Run with::

    uv run python -m evals.stress.fixtures.build_pdfs
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader

FIXTURES_DIR = Path(__file__).parent
TARGETS_KB = (5, 20, 50, 100)

# Fixed metadata date → byte-identical output across runs.
_CREATION_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)

_PARAGRAPH = (
    "The supplier portal must replace the existing manual workflow. Vendors "
    "upload PDFs of compliance certificates which the procurement team reviews "
    "for completeness, expiration, and conformance to the company's standard. "
    "Approved suppliers are unlocked for purchase order generation; rejected "
    "suppliers must remediate before the next review cycle. Every state "
    "transition is recorded in an immutable audit log retained for seven years "
    "in compliance with internal policy. Reviewers may attach comments to a "
    "specific certificate; comments are visible to the supplier and form part "
    "of the formal record. The portal must integrate with the legacy ERP for "
    "vendor master data synchronisation, propagating changes within fifteen "
    "minutes. Single sign-on is mandatory for internal users; vendors use "
    "magic-link email authentication scoped to a single tenant. Performance "
    "expectations are P95 page load under two seconds with two hundred "
    "concurrent users, and the system must remain available under one hour of "
    "downtime per quarter excluding planned maintenance windows scheduled "
    "outside business hours. "
)


def pdf_path(target_kb: int) -> Path:
    return FIXTURES_DIR / f"attach_{target_kb}kb.pdf"


def build_pdf(target_kb: int, output_path: Path) -> tuple[int, int]:
    """Build a PDF whose *extracted text* is approximately ``target_kb`` KB.

    Returns ``(text_chars, file_bytes)``. The text-vs-bytes ratio depends on
    fpdf2's PDF compression.
    """
    pdf = FPDF()
    pdf.set_creation_date(_CREATION_DATE)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    target_chars = target_kb * 1024
    text_chars = 0
    while text_chars < target_chars:
        pdf.multi_cell(0, 5, _PARAGRAPH)
        text_chars += len(_PARAGRAPH)
        if text_chars < target_chars:
            # Blank line as a soft delimiter, also helps pypdf parse paragraphs.
            pdf.ln(3)

    pdf.output(str(output_path))
    return text_chars, output_path.stat().st_size


def extracted_chars(path: Path) -> int:
    """Read the PDF back through pypdf and return the extracted-text length.

    ``app.attachments.extractor`` also uses pypdf, so this is what the
    pipeline will see (before its ``MAX_ATTACHMENT_CHARS`` truncation).
    """
    reader = PdfReader(str(path))
    return len("\n\n".join(page.extract_text() or "" for page in reader.pages))


def ensure_pdf(target_kb: int) -> Path:
    """Return the fixture path for ``target_kb``, building it if missing."""
    if target_kb not in TARGETS_KB:
        raise ValueError(f"no fixture for {target_kb} KB; available: {TARGETS_KB} (0 = no attachment)")
    path = pdf_path(target_kb)
    if not path.exists():
        build_pdf(target_kb, path)
    return path


def main() -> int:
    print(f"Writing PDFs to: {FIXTURES_DIR}")
    for kb in TARGETS_KB:
        path = pdf_path(kb)
        text_chars, file_bytes = build_pdf(kb, path)
        print(
            f"  {path.name}: text_chars={text_chars} "
            f"file_bytes={file_bytes} pypdf_extracted={extracted_chars(path)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
