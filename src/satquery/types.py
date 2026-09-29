"""Shared vocabulary (enums) used across io, tools, agent and API.

Values are lower-case strings so they serialise cleanly into the trace and API.
"""

from enum import StrEnum


class Modality(StrEnum):
    OPTICAL = "optical"  # RGB / RGB+NIR (<= 4 bands) or panchromatic
    MULTISPECTRAL = "multispectral"  # >= 5 optical bands (e.g. with red-edge / SWIR)
    SAR = "sar"
    UNKNOWN = "unknown"

    @property
    def family(self) -> str:
        """``optical`` for OPTICAL/MULTISPECTRAL, else the value itself."""
        return "optical" if self in (Modality.OPTICAL, Modality.MULTISPECTRAL) else self.value


class InputConfig(StrEnum):
    SINGLE_OPT = "single_opt"
    SINGLE_SAR = "single_sar"
    PAIR_OPT_SAR = "pair_opt_sar"
    PAIR_TEMPORAL = "pair_temporal"


class TaskType(StrEnum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    CHANGE_DESC = "change_desc"
    FUSION_EXTRACT = "fusion_extract"


class AnswerType(StrEnum):
    YES_NO = "yes_no"
    COUNT = "count"
    AREA = "area"
    FREE_TEXT = "free_text"


class TargetClass(StrEnum):
    WATER = "water"
    BUILTUP = "builtup"
    VEGETATION = "vegetation"


class BandRole(StrEnum):
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    RED_EDGE = "red_edge"
    NIR = "nir"
    SWIR1 = "swir1"
    SWIR2 = "swir2"
    PAN = "pan"
    VV = "vv"
    VH = "vh"
    HH = "hh"
    HV = "hv"


# Canonical layer keys inside an InputBundle, per input configuration.
LAYER_KEYS: dict[InputConfig, tuple[str, ...]] = {
    InputConfig.SINGLE_OPT: ("optical",),
    InputConfig.SINGLE_SAR: ("sar",),
    InputConfig.PAIR_OPT_SAR: ("optical", "sar"),
    InputConfig.PAIR_TEMPORAL: ("t1", "t2"),
}
