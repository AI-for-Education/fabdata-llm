"""
Fixtures for integration tests.
"""
import os
from pathlib import Path

# Load real API keys BEFORE any fdllm imports
HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent
PROJECT_ROOT = TEST_ROOT.parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Now import fdllm modules
import pytest
from io import BytesIO
from fdllm.sysutils import register_models

register_models(TEST_ROOT / "custom_models_test.yaml")


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
