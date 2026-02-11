"""
Tests for LLMDocument class and document support across providers.
Tests are parametrized to run across OpenAI, Anthropic, and Google callers.
"""
import pytest
import base64
from pathlib import Path
from PIL import Image

from dotenv import load_dotenv

from fdllm import OpenAICaller, ClaudeCaller, GoogleGenAICaller
from fdllm.llmtypes import LLMMessage, LLMDocument, LLMImage
from fdllm.sysutils import register_models

HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE

load_dotenv(TEST_ROOT / "test.env", override=True)
register_models(TEST_ROOT / "custom_models_test.yaml")

# Sample PDF bytes (minimal valid PDF structure for testing)
MINIMAL_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"


@pytest.fixture
def sample_pdf_bytes():
    return MINIMAL_PDF


@pytest.fixture
def sample_pdf_path(tmp_path, sample_pdf_bytes):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)
    return pdf_file


@pytest.fixture
def sample_image():
    return Image.new("RGB", (10, 10), color="red")


# ===== LLMDocument Class Tests =====


class TestLLMDocument:
    def test_init_with_path(self, sample_pdf_path):
        doc = LLMDocument(Path=sample_pdf_path)
        assert doc.Data is not None
        assert doc.Filename == "test.pdf"
        assert doc.Path_ == sample_pdf_path

    def test_init_with_bytes(self, sample_pdf_bytes):
        doc = LLMDocument(Data=sample_pdf_bytes)
        assert doc.Data == sample_pdf_bytes
        assert doc.get_filename() == "document.pdf"

    def test_init_with_bytes_and_filename(self, sample_pdf_bytes):
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="custom.pdf")
        assert doc.Data == sample_pdf_bytes
        assert doc.get_filename() == "custom.pdf"

    def test_encode(self, sample_pdf_bytes):
        doc = LLMDocument(Data=sample_pdf_bytes)
        encoded = doc.encode()
        assert isinstance(encoded, str)
        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == sample_pdf_bytes

    def test_encode_no_data_raises(self):
        doc = LLMDocument()
        with pytest.raises(ValueError, match="No document data available"):
            doc.encode()

    def test_get_filename_default(self):
        doc = LLMDocument(Data=b"test")
        assert doc.get_filename() == "document.pdf"

    def test_get_filename_from_path(self, sample_pdf_path):
        doc = LLMDocument(Path=sample_pdf_path)
        assert doc.get_filename() == "test.pdf"

    def test_get_filename_explicit(self, sample_pdf_bytes):
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="explicit.pdf")
        assert doc.get_filename() == "explicit.pdf"

    def test_list_from_paths(self, sample_pdf_path):
        docs = LLMDocument.list_from_paths([sample_pdf_path])
        assert len(docs) == 1
        assert docs[0].Filename == "test.pdf"
        assert docs[0].Data is not None

    def test_list_from_paths_none(self):
        docs = LLMDocument.list_from_paths(None)
        assert docs is None

    def test_list_from_paths_multiple(self, tmp_path, sample_pdf_bytes):
        # Create multiple PDF files
        paths = []
        for i in range(3):
            pdf_file = tmp_path / f"doc{i}.pdf"
            pdf_file.write_bytes(sample_pdf_bytes)
            paths.append(pdf_file)

        docs = LLMDocument.list_from_paths(paths)
        assert len(docs) == 3
        assert docs[0].Filename == "doc0.pdf"
        assert docs[1].Filename == "doc1.pdf"
        assert docs[2].Filename == "doc2.pdf"


# ===== Provider-Specific Format Message Tests =====


class TestOpenAIDocumentFormatMessage:
    def test_format_message_with_document(self, sample_pdf_bytes):
        caller = OpenAICaller(model="gpt-4.1")
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="test.pdf")
        message = LLMMessage(Role="user", Message="Analyze this PDF", Documents=[doc])

        result = caller.format_message(message)

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2  # document + text

        # Check document format
        doc_content = result["content"][0]
        assert doc_content["type"] == "file"
        assert doc_content["file"]["filename"] == "test.pdf"
        assert doc_content["file"]["file_data"].startswith(
            "data:application/pdf;base64,"
        )

        # Check text
        assert result["content"][1] == {"type": "text", "text": "Analyze this PDF"}

    def test_format_message_document_unsupported_model(self, sample_pdf_bytes):
        caller = OpenAICaller(model="gpt-4.1-no-doc")
        doc = LLMDocument(Data=sample_pdf_bytes)
        message = LLMMessage(Role="user", Message="test", Documents=[doc])

        with pytest.raises(NotImplementedError, match="doesn't support documents"):
            caller.format_message(message)

    def test_format_message_combined_documents_and_images(
        self, sample_pdf_bytes, sample_image
    ):
        caller = OpenAICaller(model="gpt-4.1")
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="test.pdf")
        img = LLMImage(Img=sample_image, Detail="low")
        message = LLMMessage(
            Role="user",
            Message="Compare these",
            Documents=[doc],
            Images=[img],
        )

        result = caller.format_message(message)

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 3  # document + image + text

        # Check order: documents first, then images, then text
        assert result["content"][0]["type"] == "file"
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][2]["type"] == "text"

    def test_format_message_multiple_documents(self, sample_pdf_bytes):
        caller = OpenAICaller(model="gpt-4.1")
        doc1 = LLMDocument(Data=sample_pdf_bytes, Filename="doc1.pdf")
        doc2 = LLMDocument(Data=sample_pdf_bytes, Filename="doc2.pdf")
        message = LLMMessage(
            Role="user", Message="Compare these PDFs", Documents=[doc1, doc2]
        )

        result = caller.format_message(message)

        assert len(result["content"]) == 3  # 2 documents + text
        assert result["content"][0]["file"]["filename"] == "doc1.pdf"
        assert result["content"][1]["file"]["filename"] == "doc2.pdf"


class TestAnthropicDocumentFormatMessage:
    def test_format_message_with_document(self, sample_pdf_bytes):
        caller = ClaudeCaller(model="claude-test")
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="test.pdf")
        message = LLMMessage(Role="user", Message="Analyze this PDF", Documents=[doc])

        result = caller.format_message(message)

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2  # document + text

        # Check document format
        doc_content = result["content"][0]
        assert doc_content["type"] == "document"
        assert doc_content["source"]["type"] == "base64"
        assert doc_content["source"]["media_type"] == "application/pdf"
        assert isinstance(doc_content["source"]["data"], str)

        # Check text
        assert result["content"][1] == {"type": "text", "text": "Analyze this PDF"}

    def test_format_message_document_unsupported_model(self, sample_pdf_bytes):
        caller = ClaudeCaller(model="claude-test-no-doc")
        doc = LLMDocument(Data=sample_pdf_bytes)
        message = LLMMessage(Role="user", Message="test", Documents=[doc])

        with pytest.raises(NotImplementedError, match="doesn't support documents"):
            caller.format_message(message)

    def test_format_message_combined_documents_and_images(
        self, sample_pdf_bytes, sample_image
    ):
        caller = ClaudeCaller(model="claude-test")
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="test.pdf")
        img = LLMImage(Img=sample_image, Detail="low")
        message = LLMMessage(
            Role="user",
            Message="Compare these",
            Documents=[doc],
            Images=[img],
        )

        result = caller.format_message(message)

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 3  # document + image + text

        # Check order: documents first, then images, then text
        assert result["content"][0]["type"] == "document"
        assert result["content"][1]["type"] == "image"
        assert result["content"][2]["type"] == "text"

    def test_format_message_multiple_documents(self, sample_pdf_bytes):
        caller = ClaudeCaller(model="claude-test")
        doc1 = LLMDocument(Data=sample_pdf_bytes, Filename="doc1.pdf")
        doc2 = LLMDocument(Data=sample_pdf_bytes, Filename="doc2.pdf")
        message = LLMMessage(
            Role="user", Message="Compare these PDFs", Documents=[doc1, doc2]
        )

        result = caller.format_message(message)

        assert len(result["content"]) == 3  # 2 documents + text
        assert result["content"][0]["type"] == "document"
        assert result["content"][1]["type"] == "document"


class TestGoogleDocumentFormatMessage:
    def test_format_message_with_document(self, sample_pdf_bytes):
        caller = GoogleGenAICaller(model="gemini-test")
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="test.pdf")
        message = LLMMessage(Role="user", Message="Analyze this PDF", Documents=[doc])

        result = caller.format_message(message)

        assert result["role"] == "user"
        assert isinstance(result["parts"], list)
        assert len(result["parts"]) == 2  # document + text

        # Check document format
        doc_content = result["parts"][0]
        assert "inline_data" in doc_content
        assert doc_content["inline_data"]["mime_type"] == "application/pdf"
        assert isinstance(doc_content["inline_data"]["data"], str)

        # Check text
        assert result["parts"][1] == {"text": "Analyze this PDF"}

    def test_format_message_document_unsupported_model(self, sample_pdf_bytes):
        caller = GoogleGenAICaller(model="gemini-test-no-doc")
        doc = LLMDocument(Data=sample_pdf_bytes)
        message = LLMMessage(Role="user", Message="test", Documents=[doc])

        with pytest.raises(NotImplementedError, match="doesn't support documents"):
            caller.format_message(message)

    def test_format_message_combined_documents_and_images(
        self, sample_pdf_bytes, sample_image
    ):
        caller = GoogleGenAICaller(model="gemini-test")
        doc = LLMDocument(Data=sample_pdf_bytes, Filename="test.pdf")
        img = LLMImage(Img=sample_image, Detail="low")
        message = LLMMessage(
            Role="user",
            Message="Compare these",
            Documents=[doc],
            Images=[img],
        )

        result = caller.format_message(message)

        assert result["role"] == "user"
        assert isinstance(result["parts"], list)
        assert len(result["parts"]) == 3  # document + image + text

        # Check order: documents first, then images, then text
        assert result["parts"][0]["inline_data"]["mime_type"] == "application/pdf"
        assert result["parts"][1]["inline_data"]["mime_type"] == "image/png"
        assert "text" in result["parts"][2]

    def test_format_message_multiple_documents(self, sample_pdf_bytes):
        caller = GoogleGenAICaller(model="gemini-test")
        doc1 = LLMDocument(Data=sample_pdf_bytes, Filename="doc1.pdf")
        doc2 = LLMDocument(Data=sample_pdf_bytes, Filename="doc2.pdf")
        message = LLMMessage(
            Role="user", Message="Compare these PDFs", Documents=[doc1, doc2]
        )

        result = caller.format_message(message)

        assert len(result["parts"]) == 3  # 2 documents + text
        assert result["parts"][0]["inline_data"]["mime_type"] == "application/pdf"
        assert result["parts"][1]["inline_data"]["mime_type"] == "application/pdf"


# ===== Parametrized Cross-Provider Tests =====


PROVIDER_CONFIGS = [
    pytest.param("openai", OpenAICaller, "gpt-4.1", id="openai"),
    pytest.param("anthropic", ClaudeCaller, "claude-test", id="anthropic"),
    pytest.param("google", GoogleGenAICaller, "gemini-test", id="google"),
]


@pytest.mark.parametrize("provider,caller_cls,model", PROVIDER_CONFIGS)
class TestDocumentFormatMessageCrossProvider:
    def test_document_message_has_correct_role(
        self, provider, caller_cls, model, sample_pdf_bytes
    ):
        caller = caller_cls(model=model)
        doc = LLMDocument(Data=sample_pdf_bytes)
        message = LLMMessage(Role="user", Message="Test", Documents=[doc])

        result = caller.format_message(message)
        assert result["role"] == "user"

    def test_document_message_includes_text(
        self, provider, caller_cls, model, sample_pdf_bytes
    ):
        caller = caller_cls(model=model)
        doc = LLMDocument(Data=sample_pdf_bytes)
        message = LLMMessage(Role="user", Message="Analyze this", Documents=[doc])

        result = caller.format_message(message)

        # Get content list (different key for Google)
        content_key = "parts" if provider == "google" else "content"
        content = result[content_key]

        # Find text element
        text_found = False
        for item in content:
            if provider == "google" and "text" in item:
                assert item["text"] == "Analyze this"
                text_found = True
            elif provider != "google" and item.get("type") == "text":
                assert item["text"] == "Analyze this"
                text_found = True

        assert text_found, "Text content not found in message"

    def test_document_only_no_text(self, provider, caller_cls, model, sample_pdf_bytes):
        caller = caller_cls(model=model)
        doc = LLMDocument(Data=sample_pdf_bytes)
        message = LLMMessage(Role="user", Message=None, Documents=[doc])

        result = caller.format_message(message)

        # Get content list
        content_key = "parts" if provider == "google" else "content"
        content = result[content_key]

        # Should only have document, no text
        assert len(content) == 1
