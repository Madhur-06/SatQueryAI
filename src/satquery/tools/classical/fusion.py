"""Rule-based optical + SAR mask fusion."""

from __future__ import annotations

from typing import Literal

import numpy as np

from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.types import Capability, InputConfig, TaskType


class FusionParams(ToolParams):
    rule: Literal["union", "intersection"] = "union"
    """Primary mask: ``union`` misses less (e.g. water under cloud seen only by SAR);
    ``intersection`` keeps only pixels both sensors agree on."""


class OptSarFusionRules(Tool):
    name = "optsar_fusion_rules"
    version = "1.0.0"
    description = (
        "Fuses an optical-derived and a SAR-derived mask of the same class. Reports "
        "agreement (both), optical-only and SAR-only pixels; confidence grows with agreement."
    )
    tasks = frozenset({TaskType.FUSION_EXTRACT, TaskType.VQA})
    input_configs = frozenset({InputConfig.PAIR_OPT_SAR})
    produces = frozenset({Capability.MASK_FUSED})
    requires = ToolRequirements(upstream={"optical": None, "sar": None})
    Params = FusionParams
    priority = 10

    def run(self, inp: ToolInput, params: FusionParams) -> ToolOutput:
        opt_grid = inp.upstream_mask("optical")
        opt = opt_grid.array.astype(bool)
        sar = inp.upstream_mask("sar").array.astype(bool)
        if opt.shape != sar.shape:
            raise ValueError(f"mask shapes differ: {opt.shape} vs {sar.shape}")
        both, either = opt & sar, opt | sar
        primary = either if params.rule == "union" else both
        n_either = int(either.sum())
        agreement = float(both.sum() / n_either) if n_either else 1.0

        in_conf = [
            c for c in (inp.upstream["optical"].confidence, inp.upstream["sar"].confidence) if c
        ]
        base = float(np.mean(in_conf)) if in_conf else 0.5
        warnings = []
        if n_either and agreement < 0.5:
            warnings.append(
                f"Optical and SAR masks agree on only {agreement:.0%} of detected pixels."
            )
        label = inp.upstream["optical"].values.get("class", "target")
        return ToolOutput(
            masks={
                "mask": opt_grid.with_array(primary),
                "agree": opt_grid.with_array(both),
                "optical_only": opt_grid.with_array(opt & ~sar),
                "sar_only": opt_grid.with_array(sar & ~opt),
            },
            values={
                "class": str(label),
                "rule": params.rule,
                "agreement_iou": round(agreement, 4),
                "agree_pixels": int(both.sum()),
                "optical_only_pixels": int((opt & ~sar).sum()),
                "sar_only_pixels": int((sar & ~opt).sum()),
                "pixels": int(primary.sum()),
            },
            confidence=base * (0.5 + 0.5 * agreement),
            confidence_basis="mean input confidence x (0.5 + 0.5 x optical/SAR mask IoU)",
            warnings=warnings,
        )
