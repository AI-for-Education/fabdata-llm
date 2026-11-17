"""
Fixtures for integration tests.
"""
import os
from io import BytesIO

import pytest

DUMMY_API_KEYS = {"a", "a.b"}


@pytest.fixture
def require_api_key():
    def _require_api_key(env_var):
        api_key = os.getenv(env_var)
        if not api_key or api_key in DUMMY_API_KEYS:
            pytest.skip(f"{env_var} is not configured for integration tests")

    return _require_api_key


@pytest.fixture(scope="session")
def sample_pdf_with_title():
    """
    Generate a minimal PDF with a clear title and body text.
    Returns tuple of (pdf_bytes, expected_title).
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title = "Quantum Fluctuations in Banana Cultivation"
    body = "This document discusses the theoretical implications of subatomic phenomena on tropical fruit farming."

    story = [
        Paragraph(f"<b>TITLE: {title}</b>", styles["Title"]),
        Spacer(1, 24),
        Paragraph(f"BODY: {body}", styles["Normal"]),
    ]

    doc.build(story)
    pdf_bytes = buffer.getvalue()

    return pdf_bytes, title
