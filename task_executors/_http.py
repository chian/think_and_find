"""Small JSON HTTP boundary shared by the configured providers."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A provider transport or response failure."""


def _http_error_detail(error: HTTPError, api_key: str) -> str:
    """Keep bounded, recognized error fields, not headers or arbitrary echoes."""
    try:
        raw = error.read(8193)
        if len(raw) > 8192:
            return ""
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict):
            return ""
        detail = body.get("error", body)
        if isinstance(detail, str):
            detail = {"message": detail}
        if not isinstance(detail, dict):
            return ""
        fields = {key: value for key in ("message", "type", "code", "param", "detail")
                  if isinstance(value := detail.get(key), (str, int)) and not isinstance(value, bool)}
        if api_key:
            fields = {key: str(value).replace(api_key, "[redacted]") for key, value in fields.items()}
        fields = {key: str(value)[:512] for key, value in fields.items()}
        return json.dumps(fields, ensure_ascii=True) if fields else ""
    except (OSError, ValueError, UnicodeError):
        return ""
    finally:
        error.close()


def http_url(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty HTTP URL")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an HTTP or HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain credentials, query, or fragment")
    return value


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str = "",
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Post one JSON request, preserving failures for the workflow reports."""

    url = http_url(url, "provider URL")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("provider timeout must be positive and finite")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(
            dict(payload), allow_nan=False, ensure_ascii=False,
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        detail = _http_error_detail(exc, api_key)
        suffix = f": {detail}" if detail else ""
        raise ProviderError(f"Provider returned HTTP {exc.code}{suffix}") from exc
    except (URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        raise ProviderError(f"Provider connection failed: {reason}") from exc
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError("Provider response must be UTF-8 JSON") from exc
    if not isinstance(body, dict):
        raise ProviderError("Provider response must be a JSON object")
    if body.get("success") is False or body.get("error"):
        detail = str(body.get("error") or body.get("message") or "request failed")
        if api_key:
            detail = detail.replace(api_key, "[redacted]")
        raise ProviderError(f"Provider rejected request: {detail}")
    return body
