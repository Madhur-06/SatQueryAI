"""Small helpers for tests: run a tool directly, compare masks."""

from __future__ import annotations

import numpy as np

from satquery.io.validate import InputBundle
from satquery.tools.base import Tool, ToolInput, ToolOutput


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def run_tool(
    tool: Tool,
    bundle: InputBundle,
    layers: list[str] | None = None,
    upstream: dict[str, ToolOutput] | None = None,
    params: dict | None = None,
    query: str | None = None,
) -> ToolOutput:
    inp = ToolInput(bundle=bundle, layers=layers or [], upstream=upstream or {}, query=query)
    return tool.invoke(inp, params)
