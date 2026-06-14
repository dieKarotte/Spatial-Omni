from __future__ import annotations


class SpatialBandingParams:
    """
    Placeholder for spatial banding parameters.
    """

    def __init__(self, base_params, hz_s_per_band=None) -> None:
        self.base_params = base_params
        self.hz_s_per_band = hz_s_per_band

    @classmethod
    def from_banding_params(cls, params, hz_s_per_band=None) -> "SpatialBandingParams":
        return cls(params, hz_s_per_band=hz_s_per_band)


class SpatialBandingCoefs:
    """
    Placeholder for spatial banding coefficients.
    """

    def __init__(self, params, sample_rate: int, dt_ms: float, nch: int) -> None:
        self.params = params
        self.sample_rate = sample_rate
        self.dt_ms = dt_ms
        self.nch = nch
        self.block_size = 1


__all__ = ["SpatialBandingParams", "SpatialBandingCoefs"]
