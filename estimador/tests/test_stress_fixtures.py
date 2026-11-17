"""Session 6, step 3 — the synthetic attachment corpus is calibrated and deterministic."""

from __future__ import annotations

import hashlib

import pytest

from app.attachments.extractor import extract_text
from app.config import get_settings
from evals.stress.fixtures import build_pdfs


@pytest.fixture
def tmp_fixtures(tmp_path, monkeypatch):
    """Point the generator at a temp dir so tests never touch the real fixtures."""
    monkeypatch.setattr(build_pdfs, "FIXTURES_DIR", tmp_path)
    return tmp_path


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_targets_are_5_20_50_100_kb() -> None:
    assert build_pdfs.TARGETS_KB == (5, 20, 50, 100)  # 0 KB = no attachment, no file


@pytest.mark.parametrize("kb", [5, 20])
def test_extracted_text_matches_the_target(tmp_fixtures, kb) -> None:
    path = build_pdfs.ensure_pdf(kb)
    assert path.name == f"attach_{kb}kb.pdf"
    extracted = build_pdfs.extracted_chars(path)
    # At least the target, and at most one extra paragraph (+ line breaks).
    assert kb * 1024 <= extracted <= kb * 1024 + 1_200


def test_build_is_byte_identical(tmp_path) -> None:
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    build_pdfs.build_pdf(5, a)
    build_pdfs.build_pdf(5, b)
    assert _sha(a) == _sha(b)


def test_ensure_pdf_builds_only_when_missing(tmp_fixtures) -> None:
    path = build_pdfs.ensure_pdf(5)
    mtime = path.stat().st_mtime_ns
    assert build_pdfs.ensure_pdf(5).stat().st_mtime_ns == mtime  # reused, not rebuilt
    with pytest.raises(ValueError):
        build_pdfs.ensure_pdf(0)


def test_pipeline_extractor_truncates_the_100kb_fixture(tmp_fixtures) -> None:
    # What the service actually sees: capped at MAX_ATTACHMENT_CHARS.
    path = build_pdfs.ensure_pdf(100)
    cap = get_settings().MAX_ATTACHMENT_CHARS
    text = extract_text(filename=path.name, content=path.read_bytes(), max_chars=cap)
    assert build_pdfs.extracted_chars(path) > cap
    assert len(text) <= cap
