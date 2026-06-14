from __future__ import annotations

from enum import Enum
from typing import Optional


class BandingShape(str, Enum):
    SOFT = "soft"


class LowerBandMode(str, Enum):
    VSV_HPF = "vsv_hpf"


class BandingParams:
    """
    Placeholder for UFB banding parameters.
    This is a stub to satisfy imports until full implementation is provided.
    """

    def __init__(
        self,
        dt_ms: float,
        design_fs: int,
        shape: BandingShape,
        transform_params: object,
        lower_band_mode: LowerBandMode,
    ) -> None:
        self.dt_ms = dt_ms
        self.design_fs = design_fs
        self.shape = shape
        self.transform_params = transform_params
        self.lower_band_mode = lower_band_mode

    @classmethod
    def Log(
        cls,
        dt_ms: float,
        design_fs: int,
        shape: BandingShape,
        transform_params: object,
        lower_band_mode: LowerBandMode,
    ) -> "BandingParams":
        return cls(dt_ms, design_fs, shape, transform_params, lower_band_mode)

    @classmethod
    def melspace(cls, *args, **kwargs) -> "BandingParams":
        raise NotImplementedError("BandingParams.melspace is not implemented yet.")


class BandingCoefs:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("BandingCoefs is not implemented yet.")


__all__ = [
    "BandingParams",
    "BandingShape",
    "LowerBandMode",
    "BandingCoefs",
]
