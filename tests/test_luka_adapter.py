import sys
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_ROOT = Path(r"D:\Codex\AstrBot")
for path in (PLUGIN_ROOT.parent, ASTRBOT_ROOT):
    if str(path) not in sys.path:
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
