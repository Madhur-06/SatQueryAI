"""Tool registry: registration, lookup by task / input configuration / capability, export."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.validate import RasterLayer
from satquery.tools.base import Tool, ToolSpec
from satquery.types import Capability, InputConfig, TaskType


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise SatQueryError(
                ErrorCode.NO_TOOL,
                f"No tool named '{name}' is registered.",
                details={"tool": name, "registered": self.names},
            ) from None

    def find(
        self,
        *,
        task: TaskType | None = None,
        config: InputConfig | None = None,
        capability: Capability | None = None,
        layers: list[RasterLayer] | None = None,
        include_unavailable: bool = False,
    ) -> list[Tool]:
        """Tools matching every given filter, best first.

        Ranking: available > real (non-mock) > higher ``priority`` > name. When
        ``layers`` is given, tools whose modality/band requirements those layers
        don't meet are excluded.
        """
        found = []
        for tool in self._tools.values():
            if task is not None and task not in tool.tasks:
                continue
            if config is not None and config not in tool.input_configs:
                continue
            if capability is not None and capability not in tool.produces:
                continue
            if layers is not None and tool.missing_requirements(layers):
                continue
            available, _ = tool.is_available()
            if not available and not include_unavailable:
                continue
            found.append((not available, tool.is_mock, -tool.priority, tool.name, tool))
        return [entry[-1] for entry in sorted(found, key=lambda e: e[:4])]

    def export_specs(self) -> list[ToolSpec]:
        specs = []
        for name in self.names:
            tool = self._tools[name]
            available, reason = tool.is_available()
            specs.append(tool.spec(available, reason))
        return specs

    def export_json(self) -> list[dict[str, Any]]:
        return [s.model_dump(mode="json") for s in self.export_specs()]

    def digest(self) -> str:
        """Short hash of every tool name, version and parameter schema (goes in the trace)."""
        stable = [
            {"name": s.name, "version": s.version, "params": s.params_schema}
            for s in self.export_specs()
        ]
        blob = json.dumps(stable, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def default_registry() -> Registry:
    """A registry with every built-in tool."""
    from satquery.tools.classical import CLASSICAL_TOOLS
    from satquery.tools.ml.qwen_vl import QwenVLZeroShot
    from satquery.tools.mock import MOCK_TOOLS

    registry = Registry()
    for cls in (*CLASSICAL_TOOLS, *MOCK_TOOLS, QwenVLZeroShot):
        registry.register(cls())
    return registry
