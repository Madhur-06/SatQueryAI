"""Classical (non-ML) tools: real results now, cross-checks for learned models later."""

from satquery.tools.classical.area import AreaStats
from satquery.tools.classical.change import ChangeCVA, MaskCompare, SarLogRatioChange
from satquery.tools.classical.fusion import OptSarFusionRules
from satquery.tools.classical.indices import NDBIBuiltupMask, NDVIVegetationMask, NDWIWaterMask
from satquery.tools.classical.sar_water import SarWaterOtsu

CLASSICAL_TOOLS = (
    NDWIWaterMask,
    NDBIBuiltupMask,
    NDVIVegetationMask,
    SarWaterOtsu,
    OptSarFusionRules,
    ChangeCVA,
    SarLogRatioChange,
    MaskCompare,
    AreaStats,
)

__all__ = [
    "CLASSICAL_TOOLS",
    "AreaStats",
    "ChangeCVA",
    "MaskCompare",
    "NDBIBuiltupMask",
    "NDVIVegetationMask",
    "NDWIWaterMask",
    "OptSarFusionRules",
    "SarLogRatioChange",
    "SarWaterOtsu",
]
