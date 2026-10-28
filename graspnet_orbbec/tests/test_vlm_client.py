"""VLM 客户端：JSON 解析、坐标映射、密钥脱敏。不打真实 API。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm.vlm_client import bbox_to_pixels, iou_xyxy, parse_vlm_json, redact_secrets


def test_parse_plain_json():
    d = parse_vlm_json('{"found": true, "bbox_2d": [10, 20, 30, 40]}')
    assert d["found"] is True
    assert d["bbox_2d"] == [10.0, 20.0, 30.0, 40.0]


def test_parse_markdown_and_bool_string():
    text = '废话\n```json\n{"found": "true", "bbox_2d": [1,2,3,4]}\n```\n'
    d = parse_vlm_json(text)
    assert d["found"] is True


def test_parse_found_false_without_bbox():
    d = parse_vlm_json('{"found": false}')
    assert d["found"] is False
    assert d["bbox_2d"] == [0.0, 0.0, 0.0, 0.0]


def test_parse_trailing_comma():
    d = parse_vlm_json('{"found": false, "bbox_2d": [0,0,0,0],}')
    assert d["found"] is False


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_vlm_json("not json at all")


def test_parse_truncated_object():
    d = parse_vlm_json('{"found": true, "bbox_2d": [10, 20, 30, 40]')
    assert d["found"] is True
    assert d["bbox_2d"] == [10.0, 20.0, 30.0, 40.0]


def test_parse_found_regex_fallback():
    d = parse_vlm_json('blah "found": false, "bbox_2d": [0,0,0,0] trailing')
    assert d["found"] is False


def test_bbox_norm1000_to_pixels():
    px = bbox_to_pixels([100, 200, 500, 800], 640, 480, coord_space="norm1000")
    assert px == [64, 96, 320, 384]


def test_bbox_pixel_space():
    px = bbox_to_pixels([10, 20, 200, 300], 640, 480, coord_space="pixel")
    assert px[0] == 10 and px[2] == 200


def test_bbox_auto_norm1000_on_hd():
    px = bbox_to_pixels([100, 200, 500, 800], 1280, 800, coord_space="auto")
    assert px == [128, 160, 640, 640]


def test_iou_perfect_and_none():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_redact_sk_and_bearer():
    s = redact_secrets("Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz error")
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in s
    assert "***" in s


def test_eval_image_must_stay_in_testdata(tmp_path):
    from vlm.run_vlm import resolve_eval_image, TESTDATA_DIR

    ok = resolve_eval_image("vlm/testdata/01_floor.png")
    assert ok.parent == TESTDATA_DIR.resolve()
    with pytest.raises(ValueError, match="testdata"):
        resolve_eval_image(str(tmp_path / "sneaky.png"))


def test_client_requires_env(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    from common.config import Config
    from vlm.vlm_client import VLMClient

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        VLMClient(Config({"vlm": {"model": "qwen3-vl-flash-2026-01-22"}}))


def test_detect_retries_then_ok(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-TESTKEYNOTREAL000")
    from common.config import Config
    from vlm import vlm_client as mod

    class FakeMsg:
        content = "not-json"

    class FakeChoice:
        message = FakeMsg()

    class FakeComp:
        choices = [FakeChoice()]

    n = {"i": 0}

    class FakeAPI:
        def create(self, **kwargs):
            assert kwargs["extra_body"]["enable_thinking"] is False
            assert "sk-" not in str(kwargs.get("messages"))[:20] or True
            n["i"] += 1
            if n["i"] < 2:
                FakeMsg.content = "zzz"
                return FakeComp()
            FakeMsg.content = '{"found": false, "bbox_2d": [0,0,0,0]}'
            return FakeComp()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "sk-TESTKEYNOTREAL000"
            self.chat = type("C", (), {"completions": FakeAPI()})()

    monkeypatch.setattr(mod, "OpenAI", FakeOpenAI, raising=False)

    def fake_openai_import():
        class O:
            chat = type("C", (), {"completions": FakeAPI()})()

        return O()

    client = mod.VLMClient(Config({"vlm": {"model": "qwen3-vl-flash-2026-01-22", "max_retries": 2, "enable_thinking": False}}))

    class Completions:
        def create(self, **kwargs):
            return FakeAPI().create(**kwargs)

    class Chat:
        completions = Completions()

    class O:
        chat = Chat()

    client._client = O()
    img = np.zeros((32, 32, 3), np.uint8)
    r = client.detect(img, "香蕉")
    assert r.found is False
    assert r.retries >= 1


def test_logger_does_not_contain_key(caplog, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-SUPERSECRETKEYVALUE")
    caplog.set_level(logging.INFO)
    logging.getLogger("graspnet_orbbec.vlm").info("key=%s", "sk-SUPERSECRETKEYVALUE")
    # 业务代码不会这么打；脱敏函数必须能清掉
    assert "sk-SUPERSECRETKEYVALUE" not in redact_secrets(caplog.text)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_api_error_is_not_retried(monkeypatch, status):
    from types import SimpleNamespace as NS
    from common.config import Config
    from vlm.vlm_client import VLMClient, VLMConfigurationError

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-TESTKEYNOTREAL000")
    client = VLMClient(Config({"vlm": {"max_retries": 3}}))
    calls = []

    class APIError(Exception):
        status_code = status

    def create(**kwargs):
        calls.append(kwargs)
        raise APIError("model_not_found sk-TESTKEYNOTREAL000")

    client._client = NS(chat=NS(completions=NS(create=create)))
    with pytest.raises(VLMConfigurationError) as ei:
        client.detect(np.zeros((32, 32, 3), np.uint8), "白色矩形")
    assert len(calls) == 1
    assert "sk-TESTKEYNOTREAL000" not in str(ei.value)
    assert f"HTTP {status}" in str(ei.value)


def test_sdk_retries_are_disabled(monkeypatch):
    from types import SimpleNamespace as NS
    from common.config import Config
    from vlm.vlm_client import VLMClient

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-TESTKEYNOTREAL000")
    constructor_args = {}
    def create_client(**kwargs):
        constructor_args.update(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=create_client))
    VLMClient(Config({"vlm": {"timeout_s": 5}}))._openai()
    assert constructor_args["max_retries"] == 0


def test_opaque_key_is_redacted(monkeypatch):
    from common.config import Config
    from vlm.vlm_client import VLMClient

    token = "example_opaque_credential_not_real"
    monkeypatch.setenv("DASHSCOPE_API_KEY", token)
    client = VLMClient(Config({"vlm": {"max_retries": 0}}))
    assert token not in client._redact(f"upstream error contains {token}")
