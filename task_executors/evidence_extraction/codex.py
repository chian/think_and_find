"""Text-only Codex Responses client using the current Codex CLI login.

The CLI owns refresh-token rotation. This client reads only its current access
token and fails closed if it expires, so it cannot invalidate the CLI session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .._http import ProviderError, _http_error_detail


_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def native_response_schema(value):
    """Compile the supported schema subset; callers still validate uniqueness."""
    if isinstance(value,list):
        return [native_response_schema(item) for item in value]
    if not isinstance(value,dict):
        return value
    result={key:native_response_schema(item) for key,item in value.items()
            if key!='uniqueItems'}
    if result.get('type')=='array' and result.get('items',{}).get('enum')==[]:
        result['items'].pop('enum')
        result['maxItems']=0
    return result


def _message_record(index, item):
    """Retain output-message provenance, without private reasoning or credentials."""
    return {"output_index": index, **{key: item.get(key) for key in
            ("id", "type", "role", "channel", "status")},
            "content": [{"type": part.get("type"), "text": part.get("text", "")}
                        for part in item.get("content", ())
                        if part.get("type") == "output_text"]}


def _final_message(output):
    messages = [(index, item) for index, item in enumerate(output)
                if item.get("type") == "message" and item.get("role") == "assistant"]
    final = [(index, item) for index, item in messages if item.get("channel") == "final"]
    eligible = final or [(index, item) for index, item in messages
                         if item.get("channel") is None]
    if not eligible:
        raise ProviderError("Codex completed response contained no final assistant message")
    # A response can emit intermediate assistant messages. The terminal final
    # message is the answer; concatenating messages changes that answer.
    index, item = eligible[-1]
    text = "".join(part["text"] for part in item.get("content", ())
                   if part.get("type") == "output_text")
    if not text.strip():
        raise ProviderError("Codex final assistant message contained no text")
    return index, item, text


def _access_token() -> tuple[str, dict]:
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    try:
        auth = json.loads((home / "auth.json").read_text(encoding="utf-8"))
        if auth.get("auth_mode") != "chatgpt":
            raise ValueError("Codex is not signed in with ChatGPT")
        token = auth["tokens"]["access_token"]
        if not isinstance(token, str) or not token:
            raise ValueError("Codex access token is missing")
        encoded = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if float(claims.get("exp", 0)) <= time.time() + 120:
            raise ValueError("Codex access token has expired; refresh the Codex login")
        account = claims.get("https://api.openai.com/auth", {})
        if not isinstance(account, dict):
            raise ValueError("Codex access token has invalid account claims")
        return token, account
    except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Codex login is unavailable: {exc}") from exc


@dataclass(frozen=True)
class CodexOAuthClient:
    model: str
    timeout: float = 300.0
    reasoning_effort: str = "medium"
    record_response: Callable[[dict], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Codex model must be a non-empty string")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("Codex timeout must be positive and finite")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Codex reasoning effort is invalid")

    def call(self, prompt: str, system_prompt: str | None = None, *,
             model: str | None = None, reasoning_effort: str | None = None,
             response_schema: dict | None = None) -> str:
        model = model or self.model
        effort = reasoning_effort or self.reasoning_effort
        if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Codex reasoning effort is invalid")
        token, account = _access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "episodic-workflow/0.1",
            "originator": "episodic-workflow",
        }
        if account.get("chatgpt_account_id"):
            headers["ChatGPT-Account-ID"] = account["chatgpt_account_id"]
        residency = account.get("chatgpt_data_residency") or account.get("chatgpt_compute_residency")
        if residency:
            headers["x-openai-internal-codex-residency"] = residency
        payload = {
            "model": model,
            "instructions": system_prompt or "Return a direct text answer to the user's request.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "store": False,
            "stream": True,
            "reasoning": {"effort": effort},
        }
        if response_schema is not None:
            payload["text"] = {"format": {"type": "json_schema", "name": "workflow_response",
                                          "strict": True, "schema": native_response_schema(response_schema)}}
        request = Request(_CODEX_RESPONSES_URL,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers=headers, method="POST")
        chunks, deltas, items = [], {}, {}
        completed_response = None
        selected = None
        agreement = None
        event_counts = {}
        assembly_source = None
        try:
            with urlopen(request, timeout=self.timeout) as response:
                deadline = time.monotonic() + self.timeout
                stream_bytes = 0
                for line in response:
                    stream_bytes += len(line)
                    if time.monotonic() >= deadline:
                        raise ProviderError("Codex response exceeded the request time limit")
                    if stream_bytes > 4 * 1024 * 1024:
                        raise ProviderError("Codex response stream exceeded 4 MiB")
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ProviderError("Codex returned an invalid response event") from exc
                    kind = event.get("type")
                    event_counts[kind] = event_counts.get(kind, 0) + 1
                    if kind == "response.output_text.delta":
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise ProviderError("Codex returned a non-text output delta")
                        chunks.append(delta)
                        address = (event.get("item_id"), event.get("output_index"),
                                   event.get("content_index", 0))
                        deltas.setdefault(address, []).append(delta)
                    elif kind in {"response.output_item.added", "response.output_item.done"}:
                        item = event.get("item", {})
                        if isinstance(item, dict):
                            items[event.get("output_index")] = item
                    elif kind == "response.completed":
                        completed_response = event.get("response", {})
                        break
                    elif kind in {"response.failed", "response.incomplete"}:
                        raise ProviderError(f"Codex response did not complete: {kind}")
            if not completed_response or completed_response.get("status") != "completed":
                raise ProviderError("Codex response stream ended without completion")
            output = completed_response.get("output")
            if not isinstance(output, list):
                raise ProviderError("Codex completed response is missing output items")
            assembly_source = "completed_response"
            if not output:
                # Some Codex transports omit output from the completion marker.
                # Completed streamed items retain the same message boundaries.
                output = [item for _, item in sorted(items.items(), key=lambda pair: pair[0] or 0)]
                assembly_source = "streamed_output_items"
            index, item, result = _final_message(output)
            selected = _message_record(index, item)
            parts = {}
            for (item_id, output_index, content_index), values in deltas.items():
                if ((item_id is not None and item_id == item.get("id"))
                        or (item_id is None and output_index == index)):
                    parts.setdefault(content_index, []).extend(values)
            if parts:
                streamed = "".join("".join(parts[key]) for key in sorted(parts))
                agreement = streamed == result
                if not agreement:
                    raise ProviderError("Codex final message disagrees with its streamed text")
            return result
        except HTTPError as exc:
            detail = _http_error_detail(exc, token)
            raise ProviderError(f"Codex returned HTTP {exc.code}" + (f": {detail}" if detail else "")) from exc
        except (URLError, OSError) as exc:
            raise ProviderError(f"Codex connection failed: {exc}") from exc
        finally:
            if self.record_response is not None:
                output = (completed_response or {}).get("output")
                observed = (list(enumerate(output)) if output
                            else sorted(items.items(), key=lambda pair: pair[0] or 0))
                self.record_response({"event": "model_transport_response", "model": model,
                    "response_id": (completed_response or {}).get("id"),
                    "status": (completed_response or {}).get("status", "interrupted"),
                    "output_items": [_message_record(i, item) for i, item in observed
                                     if item.get("type") == "message"],
                    "selected_message": selected,
                    "selected_delta_agreement": agreement,
                    "assembly_source": assembly_source,
                    "native_response_schema": response_schema is not None,
                    "response_schema_hash": (hashlib.sha256(json.dumps(response_schema,sort_keys=True).encode()).hexdigest()
                                             if response_schema is not None else None),
                    "stream_event_counts": event_counts,
                    "streamed_output_items": [_message_record(i, item) for i, item in
                        sorted(items.items(), key=lambda pair: pair[0] or 0)
                        if item.get("type") == "message"],
                    "delta_addresses": [{"item_id": k[0], "output_index": k[1],
                                         "content_index": k[2], "text": "".join(v)}
                                        for k, v in deltas.items()],
                    "flattened_output_text": "".join(chunks)})

    async def call_async(self, prompt: str, system_prompt: str | None = None) -> str:
        return await asyncio.to_thread(self.call, prompt, system_prompt)
