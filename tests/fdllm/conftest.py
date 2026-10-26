from pathlib import Path

from dotenv import load_dotenv

from fdllm.sysutils import register_models

TEST_ROOT = Path(__file__).resolve().parent


def _running_integration_only(config):
    marker_expr = (config.getoption("-m") or "").strip()
    return "integration" in marker_expr and "not integration" not in marker_expr


def pytest_configure(config):
    if not _running_integration_only(config):
        load_dotenv(TEST_ROOT / "test.env", override=True)
    register_models(TEST_ROOT / "custom_models_test.yaml")
