"""
ToolRegistry — named registry of callables used by AgentCore.

All generators, validators, property calculators, filters and screening
tools are registered here so the agent loop can resolve them by name.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional


class ToolRegistry:
    """Register and look up tools by name."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._meta: Dict[str, dict] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        category: str = "",
        description: str = "",
    ) -> None:
        """Register a callable under ``name`` with optional metadata."""
        self._tools[name] = fn
        self._meta[name] = {
            "name": name,
            "category": category,
            "description": description,
        }

    def get(self, name: str) -> Optional[Callable]:
        """Return the registered callable, or ``None`` if absent."""
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self, category: Optional[str] = None) -> List[dict]:
        """List registered tools, optionally filtered by category."""
        items = []
        for name, meta in self._meta.items():
            if category and meta.get("category") != category:
                continue
            items.append(meta)
        return items

    def describe(self, name: str) -> Optional[dict]:
        return self._meta.get(name)
