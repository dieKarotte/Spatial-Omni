"""Spatial-Omni model registrations.

Qwen3-Omni is optional so the Qwen2.5 environment can import this package
without installing the newer Transformers build.
"""

try:
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerConfig  # noqa: F401
except ImportError:
    pass
else:
    from .configuration_qwen3_omni import Qwen3OmniMoeSpatialThinkerConfig
    from .modeling_so_thinker_qwen3 import (
        Qwen3OmniMoeSpatialForConditionalGeneration,
        Qwen3OmniMoeSpatialThinkerForConditionalGeneration,
        register_qwen3_spatial_auto_classes,
    )
    from .processing_so_qwen3 import Qwen3OmniMoeSpatialProcessor

    __all__ = [
        "Qwen3OmniMoeSpatialThinkerConfig",
        "Qwen3OmniMoeSpatialThinkerForConditionalGeneration",
        "Qwen3OmniMoeSpatialForConditionalGeneration",
        "Qwen3OmniMoeSpatialProcessor",
        "register_qwen3_spatial_auto_classes",
    ]
