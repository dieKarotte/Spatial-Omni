"""SO (Spatial-Omni) extension for Microsoft Phi-4-multimodal.

Self-contained package that adds an independent FOA spatial-audio modality
(SO-Encoder / Spatial-BEATs + pixel-shuffle projector) on top of the official
Phi-4-multimodal framework (modality-LoRA + embed_tokens_extend injection),
without modifying the base checkpoint or its vendored code.

Layout:
    processing.py  – SoPhi4Processor: registers + expands <|spatial|> placeholders
    modeling.py    – get_so_phi4_class: SO wrapper factory (dynamic subclass of
                     the official Phi4MMForCausalLM loaded via trust-remote-code
                     machinery)
    collator.py    – SoPhi4QACollator: QA batch builder (prompt/labels/audio/FOA)
"""

from .processing import SoPhi4Processor, SPATIAL_TOKEN
from .modeling import get_so_phi4_class, load_base_phi4mm_class, SO_ADAPTER_NAME
from .collator import SoPhi4QACollator

__all__ = [
    "SoPhi4Processor",
    "SPATIAL_TOKEN",
    "get_so_phi4_class",
    "load_base_phi4mm_class",
    "SO_ADAPTER_NAME",
    "SoPhi4QACollator",
]
