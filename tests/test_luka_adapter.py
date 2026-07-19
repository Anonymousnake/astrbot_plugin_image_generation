import sys
import unittest
import os
import json
import time
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_ROOT_CANDIDATES = [
    Path(os.environ["ASTRBOT_ROOT"]) if os.environ.get("ASTRBOT_ROOT") else None,
    Path("/home/ubuntu/AstrBot"),
    Path(r"D:\Codex\AstrBot"),
]
for path in [PLUGIN_ROOT.parent, *ASTRBOT_ROOT_CANDIDATES]:
    if path and path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import astrbot_plugin_image_generation.main  # noqa: F401
from astrbot_plugin_image_generation.adapter.luka_adapter import LukaAdapter
from astrbot_plugin_image_generation.core.config_manager import ConfigManager
from astrbot_plugin_image_generation.core.generator import ImageGenerator
from astrbot_plugin_image_generation.core.types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    ImageData,
)


class DictConfig(dict):
    schema = None


class LukaAdapterTests(unittest.TestCase):
    def test_config_manager_parses_luka_provider(self) -> None:
        config = DictConfig(
            {
                "api_providers": [
                    {
                        "__template_key": "luka",
                        "name": "LukaLeng",
                        "base_url": "https://art.luka77.cc/api/v1",
                        "api_keys": ["secret-token"],
                        "available_models": ["gpt-image-2"],
                        "capability_options": ["文生图", "图生图", "宽高比"],
                    }
                ],
                "generation": {"model": "LukaLeng/gpt-image-2"},
            }
        )

        manager = ConfigManager(config)

        self.assertIsNotNone(manager.adapter_config)
        self.assertEqual(manager.adapter_config.type, AdapterType.LUKA)
        self.assertEqual(manager.adapter_config.name, "LukaLeng")
        self.assertEqual(manager.adapter_config.model, "gpt-image-2")

    def test_generator_creates_luka_adapter(self) -> None:
        generator = ImageGenerator(
            AdapterConfig(
                type=AdapterType.LUKA,
                name="LukaLeng",
                api_keys=["secret-token"],
                model="gpt-image-2",
                capability_options={"text_to_image": True, "image_to_image": True},
            )
        )

        self.assertIsInstance(generator.adapter, LukaAdapter)

    def test_builds_browser_headers_without_multipart_content_type(self) -> None:
        adapter = LukaAdapter(
            AdapterConfig(
                type=AdapterType.LUKA,
                name="LukaLeng",
                api_keys=["secret-token"],
                model="gpt-image-2",
            )
        )

        json_headers = adapter._build_headers(json_request=True)
        form_headers = adapter._build_headers(json_request=False)

        self.assertEqual(json_headers["Authorization"], "Bearer secret-token")
        self.assertEqual(json_headers["Origin"], "https://art.luka77.cc")
        self.assertEqual(json_headers["Referer"], "https://art.luka77.cc/image")
        self.assertIn("User-Agent", json_headers)
        self.assertEqual(json_headers["Content-Type"], "application/json")
        self.assertNotIn("Content-Type", form_headers)

    def test_token_file_overrides_static_key_and_reloads(self) -> None:
        temp = Path(self._get_temp_dir()) / "luka-token.json"
        temp.write_text(json.dumps({"access_token": "first-token"}), encoding="utf-8")
        adapter = LukaAdapter(
            AdapterConfig(type=AdapterType.LUKA, api_keys=["static-token"], extra={"token_file": str(temp)})
        )
        self.assertEqual(adapter._get_current_api_key(), "first-token")
        time.sleep(0.002)
        temp.write_text(json.dumps({"access_token": "second-token"}), encoding="utf-8")
        self.assertEqual(adapter._get_current_api_key(), "second-token")

    def test_invalid_token_file_falls_back_to_static_key(self) -> None:
        temp = Path(self._get_temp_dir()) / "invalid-token.json"
        temp.write_text("not json", encoding="utf-8")
        adapter = LukaAdapter(
            AdapterConfig(type=AdapterType.LUKA, api_keys=["static-token"], extra={"token_file": str(temp)})
        )
        self.assertEqual(adapter._get_current_api_key(), "static-token")

    def test_auth_failure_is_not_retryable(self) -> None:
        adapter = LukaAdapter(AdapterConfig(type=AdapterType.LUKA, api_keys=["static-token"]))
        self.assertFalse(adapter._is_retryable_error("未登录或权限不足"))

    def _get_temp_dir(self) -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="luka-adapter-")

    def test_uses_edits_endpoint_and_multipart_for_reference_images(self) -> None:
        adapter = LukaAdapter(
            AdapterConfig(
                type=AdapterType.LUKA,
                name="LukaLeng",
                base_url="https://art.luka77.cc/api",
                api_keys=["secret-token"],
                model="gpt-image-2",
            )
        )
        request = GenerationRequest(
            prompt="turn this into watercolor",
            images=[ImageData(data=b"fake-image", mime_type="image/png")],
            aspect_ratio="1:1",
            task_id="unit",
        )

        form, fields = adapter._build_edit_form(request)

        self.assertEqual(adapter._endpoint_url("images/edits"), "https://art.luka77.cc/api/v1/images/edits")
        self.assertIn("model", fields)
        self.assertIn("prompt", fields)
        self.assertIn("source", fields)
        self.assertIn("response_format", fields)
        self.assertIn("image", fields)
        self.assertTrue(form.is_multipart)
        self.assertEqual(adapter._source(), "image-page")


if __name__ == "__main__":
    unittest.main()
