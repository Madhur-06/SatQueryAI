"""Real-weights smoke test for the Qwen-VL tool. Opt-in: ``just test-ml`` (--run-ml).

Needs torch/transformers and the model in the local Hugging Face cache; it never downloads.
"""

import importlib.util

import pytest

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(
        importlib.util.find_spec("transformers") is None, reason="needs the [ml] extra"
    ),
]


@pytest.fixture
def vlm_enabled(monkeypatch):
    from satquery.config import get_settings

    monkeypatch.setenv("SATQUERY_VLM_ENABLED", "true")
    monkeypatch.setenv("SATQUERY_VLM_ALLOW_DOWNLOAD", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_qwen_answers_and_captions(vlm_enabled, opt_bundle, temporal_bundle) -> None:
    from satquery.testing.helpers import run_tool
    from satquery.tools.ml.qwen_vl import QwenVLZeroShot
    from satquery.types import Capability

    tool = QwenVLZeroShot()
    assert tool.is_available() == (True, None)

    answer = run_tool(tool, opt_bundle, ["optical"], query="Is there a water body in this image?")
    assert answer.text
    assert not answer.is_mock
    assert len(answer.text.split()) <= 5  # short-answer prompt

    from satquery.tools.base import ToolInput

    caption = tool.invoke(
        ToolInput(bundle=opt_bundle, layers=["optical"], capability=Capability.CAPTION), {}
    )
    assert len(caption.text.split()) > 5

    change = tool.invoke(
        ToolInput(
            bundle=temporal_bundle,
            layers=["t1", "t2"],
            query="Has anything been built between the two images?",
            capability=Capability.CHANGE_ANSWER,
        ),
        {},
    )
    assert change.text
