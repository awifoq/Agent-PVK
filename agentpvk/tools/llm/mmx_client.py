"""MiniMax LLM client used by the analyst and optical-PCE tools.

Supports two backends:

* HTTP — requires ``MINIMAX_API_KEY`` (and optionally ``MINIMAX_GROUP_ID``)
  environment variables; calls the MiniMax chat-completion API.
* CLI — falls back to the ``mmx`` command line client (``mmx.cmd`` on
  Windows) if no API key is configured.

Raise errors loudly so callers can degrade gracefully.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


class MMXClient:
    """Minimal MiniMax chat client with HTTP + CLI backends."""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
        base_url: str = "https://api.minimax.chat/v1/text/chatcompletion_v2",
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.environ.get("MINIMAX_GROUP_ID", "")
        self.base_url = base_url
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Send a chat request and return the assistant text content."""
        if self.api_key:
            return self._chat_http(messages, temperature, max_tokens)
        return self._chat_cli(messages, temperature, max_tokens)

    def _chat_http(self, messages, temperature, max_tokens) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            self.base_url, headers=headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected MiniMax response: {data}")

    def _chat_cli(self, messages, temperature, max_tokens) -> str:
        mmx = "mmx.cmd" if os.name == "nt" else "mmx"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(messages, f, ensure_ascii=False)
            messages_file = f.name
        try:
            cmd = [
                mmx, "text", "chat",
                "--messages-file", messages_file,
                "--model", self.model,
                "--max-tokens", str(max_tokens),
                "--output", "json",
                "--quiet",
                "--non-interactive",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                timeout=self.timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mmx failed (code {result.returncode}): {result.stderr[:300]}"
                )
            content = (result.stdout or "").strip()
            try:
                data = json.loads(content)
                content = data.get("content", "") if isinstance(data, dict) else content
            except json.JSONDecodeError:
                pass
            return content.strip()
        finally:
            Path(messages_file).unlink(missing_ok=True)
