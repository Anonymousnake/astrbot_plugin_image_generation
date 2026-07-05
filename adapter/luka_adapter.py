from __future__ import annotations

import base64
import time
from typing import Any

import aiohttp
from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import UNSPECIFIED_OPTION
from ..core.logging_utils import (
    safe_log_error_body,
    safe_log_mapping,
    safe_log_text,
    safe_log_url,
)
from ..core.types import GenerationRequest, ImageCapability, ImageData


class LukaAdapter(BaseImageAdapter):
    """LukaLeng 生图站适配器，支持文生图和 multipart 图生图。"""

    DEFAULT_BASE_URL = "https://art.luka77.cc/api/v1"
    DEFAULT_MODEL = "gpt-image-2"
    DEFAULT_REFERER = "https://art.luka77.cc/image"
    DEFAULT_ORIGIN = "https://art.luka77.cc"
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.0.0 Safari/537.36"
    )
    IMAGE_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    def _pre_generate(self, request: GenerationRequest):
        mode = "图片编辑" if request.images else "文本生成图片"
        logger.debug(
            f"{self._get_log_prefix(request.task_id)} 开始{mode}: "
            f"提示词={safe_log_text(request.prompt, 120)}，模型={self._model_name()}"
        )
        return None

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)
        session = self._get_session()

        if request.images:
            form, fields = self._build_edit_form(request)
            url = self._endpoint_url("images/edits")
            headers = self._build_headers(json_request=False)
            kwargs: dict[str, Any] = {"data": form}
            logger.debug(f"{prefix} 请求 URL: {safe_log_url(url)}, Form 字段: {fields}")
        else:
            payload = self._build_generation_payload(request)
            url = self._endpoint_url("images/generations")
            headers = self._build_headers(json_request=True)
            kwargs = {"json": payload}
            logger.debug(
                f"{prefix} 请求 URL: {safe_log_url(url)}, Payload 字段: {list(payload.keys())}"
            )
            self._log_debug_json("请求", payload, request.task_id)

        try:
            async with session.post(
                url,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
                **kwargs,
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): "
                        f"{safe_log_error_body(error_text)}"
                    )
                    return None, f"API 错误 ({resp.status})"

                data = await self._read_response_json(resp, request.task_id)
                logger.debug(f"{prefix} 生成成功 (耗时: {duration:.2f}s)")
                return await self._extract_images(data, request.task_id)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {exc}")
            return None, str(exc)

    def _endpoint_url(self, path: str) -> str:
        """构建 Luka /api/v1 图像接口地址。"""
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        for suffix in ("/api/v1/images/generations", "/api/v1/images/edits"):
            if base.endswith(suffix):
                base = base[: -len(suffix)] + "/api/v1"
                break
        if base.endswith("/images/generations") or base.endswith("/images/edits"):
            base = base.rsplit("/images/", 1)[0]
        if base.endswith("/api/v1"):
            return f"{base}/{path}"
        if base.endswith("/api"):
            return f"{base}/v1/{path}"
        return f"{base}/api/v1/{path}"

    def _build_headers(self, *, json_request: bool) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Accept": "application/json, text/plain, */*",
            "Origin": self._origin(),
            "Referer": self._referer(),
            "User-Agent": self._user_agent(),
        }
        if json_request:
            headers["Content-Type"] = "application/json"
        return headers

    def _build_generation_payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "prompt": self._prompt_text(request.prompt),
            "source": self._source(),
            "n": 1,
            "response_format": self._response_format(),
        }
        if size := self._resolve_size(request):
            payload["size"] = size
        if quality := self._quality():
            payload["quality"] = quality
        return payload

    def _build_edit_form(
        self, request: GenerationRequest
    ) -> tuple[aiohttp.FormData, list[str]]:
        form = aiohttp.FormData()
        fields: list[str] = []

        def add_text(name: str, value: str) -> None:
            form.add_field(name, value)
            fields.append(name)

        add_text("model", self._model_name())
        add_text("prompt", self._prompt_text(request.prompt))
        add_text("source", self._source())
        add_text("n", "1")
        add_text("response_format", self._response_format())
        if size := self._resolve_size(request):
            add_text("size", size)
        if quality := self._quality():
            add_text("quality", quality)

        max_images = self._max_reference_images()
        for index, image in enumerate(request.images[:max_images], start=1):
            form.add_field(
                "image",
                image.data,
                filename=self._image_filename(index, image.mime_type),
                content_type=image.mime_type or "image/png",
            )
            fields.append("image")
        if len(request.images) > max_images:
            logger.debug(
                f"{self._get_log_prefix(request.task_id)} Luka 最多发送 {max_images} 张参考图，"
                f"已忽略 {len(request.images) - max_images} 张"
            )
        return form, fields

    def _resolve_size(self, request: GenerationRequest) -> str | None:
        raw_size = str(self.config.extra.get("size") or "").strip()
        if raw_size:
            return raw_size
        if not request.aspect_ratio or request.aspect_ratio == UNSPECIFIED_OPTION:
            return None
        return request.aspect_ratio

    def _prompt_text(self, prompt: str) -> str:
        system_prompt = str(self.config.extra.get("system_prompt") or "").strip()
        if not system_prompt:
            return prompt
        return f"{system_prompt}\n\n{prompt}"

    def _model_name(self) -> str:
        return self.model or self.DEFAULT_MODEL

    def _source(self) -> str:
        return str(self.config.extra.get("source") or "image-page").strip() or "image-page"

    def _response_format(self) -> str:
        value = str(self.config.extra.get("response_format") or "").strip()
        if value in {"url", "b64_json"}:
            return value
        return "url"

    def _quality(self) -> str:
        value = str(self.config.extra.get("quality") or "").strip()
        if value and value != UNSPECIFIED_OPTION:
            return value
        return ""

    def _origin(self) -> str:
        return str(self.config.extra.get("origin") or self.DEFAULT_ORIGIN).strip()

    def _referer(self) -> str:
        return str(self.config.extra.get("referer") or self.DEFAULT_REFERER).strip()

    def _user_agent(self) -> str:
        return str(
            self.config.extra.get("user_agent") or self.DEFAULT_USER_AGENT
        ).strip()

    def _max_reference_images(self) -> int:
        raw = self.config.extra.get("max_reference_images", 4)
        try:
            return max(1, min(int(raw), 8))
        except (TypeError, ValueError):
            return 4

    def _image_filename(self, index: int, mime_type: str) -> str:
        extension = self.IMAGE_EXTENSIONS.get((mime_type or "").lower(), "png")
        return f"image_{index}.{extension}"

    async def _extract_images(
        self, data: dict[str, Any], task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        prefix = self._get_log_prefix(task_id)
        if isinstance(data.get("error"), dict):
            return None, str(data["error"].get("message") or data["error"])
        if isinstance(data.get("code"), int) and data.get("code") != 0:
            return None, str(data.get("msg") or "请求失败")

        items = data.get("data")
        if not isinstance(items, list):
            return None, f"响应格式错误: {safe_log_mapping(data)}"

        images: list[bytes] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if b64_json := item.get("b64_json"):
                if decoded := self._decode_base64_image(b64_json, task_id):
                    images.append(decoded)
            elif url := item.get("url"):
                url = str(url)
                if url.startswith("data:image/"):
                    if decoded := self._decode_base64_image(url, task_id):
                        images.append(decoded)
                elif downloaded := await self._download_image(url, task_id):
                    images.append(downloaded)
            else:
                logger.warning(
                    f"{prefix} 无法从响应项中提取图像: {safe_log_mapping(item)}"
                )

        if not images:
            return None, "未生成任何图像"
        logger.debug(f"{prefix} 成功提取 {len(images)} 张图像")
        return images, None

    def _decode_base64_image(
        self, value: Any, task_id: str | None = None
    ) -> bytes | None:
        data = str(value or "")
        if ";base64," in data:
            _, _, data = data.partition(";base64,")
        try:
            return base64.b64decode(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{self._get_log_prefix(task_id)} Base64 解码失败: {exc}")
            return None

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url,
                proxy=self.proxy,
                timeout=self._get_download_timeout(),
                headers={"User-Agent": self._user_agent(), "Referer": self._referer()},
            ) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    logger.debug(f"{prefix} 图像下载成功: {len(data)} bytes")
                    return data
                logger.error(
                    f"{prefix} 下载图像失败 ({resp.status}): {safe_log_url(url)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像异常: {exc}")
        return None
