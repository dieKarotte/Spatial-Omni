"""Spatial-aware Qwen3-Omni-MoE Thinker (Spatial-BEATs path).

Subclass of ``Qwen3OmniMoeThinkerForConditionalGeneration`` from the local
transformers fork. Only the Spatial-BEATs encoder path is wired here; IV /
Neural-IV / SELD233 are deferred (see plan.md).

Differences vs the Qwen2.5 Spatial Thinker:

- Parent class lives in the fork; we sys-path inject for safety.
- LLM hidden size is 2048 (vs 4096 for Qwen2.5-Omni-7B). The projector adapts
  via ``output_dim=config.text_config.hidden_size``.
- Qwen3 ``get_rope_index`` does not know about ``<|spatial|>`` tokens, but it
  treats any unknown token id as plain text and assigns sequential position
  ids. We therefore do NOT override ``get_rope_index`` — the upstream
  implementation produces correct positions for spatial tokens out of the box.
- Qwen3 forward takes ``input_features`` (Whisper-style mel) instead of
  ``feature_attention_mask + input_features``-only Qwen2.5 conventions. We
  pass through unchanged.
- Talker is ignored entirely; we wrap the Thinker only.
- ``prepare_inputs_for_generation`` MUST be overridden so that
  ``generate()`` forwards spatial_audio / spatial_tokens / spatial_*_lengths
  through to ``forward()``. Upstream's signature does not list these
  kwargs, so without an override they get swallowed by ``**kwargs`` and
  the spatial signal is silently dropped at inference time, while
  training-time ``forward()`` (called directly, not via ``generate()``)
  remains unaffected. This is the bug that caused azimuth/elevation/
  detect_time to collapse to ~chance levels in early Qwen3 benches even
  though valid_loss decreased normally during training.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Bootstrap fork import path.
# ---------------------------------------------------------------------------
_FORK = os.environ.get("QWEN3_OMNI_FORK", os.environ.get("QWEN3_TRANSFORMERS_FORK", ""))
if _FORK and os.path.isdir(_FORK) and _FORK not in sys.path:
    sys.path.insert(0, _FORK)

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from .configuration_qwen3_omni import Qwen3OmniMoeSpatialThinkerConfig  # noqa: E402
from ..modules.so_encoder import SOEncoder  # noqa: E402
from ..modules.so_token_projector import build_so_token_projector  # noqa: E402


class Qwen3OmniMoeSpatialThinkerForConditionalGeneration(
    Qwen3OmniMoeThinkerForConditionalGeneration
):
    """Spatial-BEATs–augmented Qwen3-Omni-MoE Thinker."""

    config_class = Qwen3OmniMoeSpatialThinkerConfig

    def __init__(self, config):
        super().__init__(config)
        self._validate_spatial_config(config)

        # We only support the BEATs path on Qwen3 for now. Init non-BEATs
        # branches to None so state_dict / attribute checks behave.
        self.so_encoder = None
        self.so_projector = None

        encoder_type = getattr(config, "spatial_encoder_type", "so_backbone")
        if encoder_type != "so_backbone":
            raise NotImplementedError(
                f"Qwen3 spatial path only supports spatial_encoder_type='so_backbone', "
                f"got '{encoder_type}'."
            )

        shuffle_factor = int(getattr(config, "so_projector_shuffle_factor", 4))
        encoder_rate = float(getattr(config, "so_encoder_token_rate", 10.0))
        llm_rate = float(getattr(config, "so_backbone_target_token_rate", 2.5))
        expected_llm_rate = encoder_rate / max(shuffle_factor, 1)
        if abs(expected_llm_rate - llm_rate) > 1e-6:
            raise ValueError(
                f"so_backbone rate mismatch: encoder_token_rate={encoder_rate} / "
                f"projector_shuffle_factor={shuffle_factor} = {expected_llm_rate}, "
                f"but so_backbone_target_token_rate={llm_rate}. "
                f"Set shuffle_factor={int(round(encoder_rate / llm_rate))} or "
                f"target_token_rate={expected_llm_rate}."
            )

        self.so_encoder = SOEncoder(
            checkpoint_path=config.so_backbone_checkpoint_path,
            beats_repo_path=config.so_backbone_repo_path,
            freeze_backbone=config.so_backbone_freeze_backbone,
            max_audio_seconds=config.so_backbone_max_audio_seconds,
            encoder_token_rate=encoder_rate,
        )
        self.so_projector = build_so_token_projector(
            projector_type=getattr(config, "so_projector_type", "pixel_shuffle"),
            input_dim=config.so_encoder_dim,
            hidden_dim=config.so_projector_hidden_dim,
            output_dim=config.text_config.hidden_size,  # 2048 for Qwen3-30B-A3B
            shuffle_factor=shuffle_factor,
        )

        # Re-run post_init so newly added modules get their initialization
        # (parent __init__ already called post_init once before our submodules
        # were added; calling again is safe and only initializes new params).
        self.post_init()

    # ------------------------------------------------------------------
    def _validate_spatial_config(self, config) -> None:
        encoder_type = getattr(config, "spatial_encoder_type", "so_backbone")
        if encoder_type == "so_backbone" and not config.so_backbone_checkpoint_path:
            raise ValueError(
                "so_backbone_checkpoint_path is required when spatial_encoder_type='so_backbone'"
            )

    # ------------------------------------------------------------------
    # Tokenizer / embedding sync (called by the spatial processor)
    # ------------------------------------------------------------------
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ------------------------------------------------------------------
    # Override get_audio_features to cast input_features to the audio_tower
    # dtype. The processor emits float32 mel; the audio_tower weights are
    # bf16 when from_pretrained loaded the model with torch_dtype=bf16.
    # Without this cast, F.conv2d crashes with
    # "Input type (float) and bias type (BFloat16) should be the same".
    # ------------------------------------------------------------------
    def get_audio_features(self, input_features, feature_attention_mask=None, audio_feature_lengths=None):
        try:
            tower_dtype = next(self.audio_tower.parameters()).dtype
        except StopIteration:
            tower_dtype = input_features.dtype
        if input_features.dtype != tower_dtype:
            input_features = input_features.to(dtype=tower_dtype)
        return super().get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )

    def sync_spatial_tokenizer(self, tokenizer, spatial_token: str = "<|spatial|>") -> int:
        vocab = tokenizer.get_vocab()
        if spatial_token not in vocab:
            tokenizer.add_special_tokens({"additional_special_tokens": [spatial_token]})
        spatial_token_id = int(tokenizer.convert_tokens_to_ids(spatial_token))
        new_vocab_size = len(tokenizer)
        current_vocab_size = int(self.get_input_embeddings().num_embeddings)
        if current_vocab_size != new_vocab_size:
            self.resize_token_embeddings(new_vocab_size)
        self.config.spatial_token_index = spatial_token_id
        self.config.text_config.vocab_size = new_vocab_size
        self.vocab_size = new_vocab_size
        return spatial_token_id

    # ------------------------------------------------------------------
    # forward — inject spatial tokens, then delegate to parent
    # ------------------------------------------------------------------
    def forward(
        self,
        *args,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        spatial_tokens: Optional[torch.Tensor] = None,
        spatial_token_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if not self._has_spatial_inputs(
            spatial_audio=spatial_audio, spatial_tokens=spatial_tokens
        ):
            return super().forward(*args, **kwargs)

        if kwargs.get("use_audio_in_video"):
            raise NotImplementedError("Spatial-Omni 30b path requires use_audio_in_video=False.")

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            raise ValueError("input_ids are required when injecting spatial tokens.")

        # Resolve spatial tokens (BEATs only)
        spatial_tokens, spatial_token_lengths = self._resolve_spatial_tokens(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            spatial_tokens=spatial_tokens,
            spatial_token_lengths=spatial_token_lengths,
        )

        # Project to LLM hidden + apply pixel-shuffle length adjustment
        projected_spatial = self.so_projector(spatial_tokens)
        shuffle_factor = int(getattr(self.so_projector, "shuffle_factor", 1))
        if shuffle_factor > 1:
            new_lengths = torch.clamp(
                torch.div(spatial_token_lengths, shuffle_factor, rounding_mode="floor"),
                min=0,
                max=int(projected_spatial.shape[1]),
            )
            nonzero = spatial_token_lengths > 0
            new_lengths = torch.where(
                nonzero & (new_lengths == 0),
                torch.ones_like(new_lengths),
                new_lengths,
            )
            spatial_token_lengths = new_lengths

        projected_spatial, spatial_token_lengths = self._align_projected_spatial_to_placeholders(
            projected_spatial=projected_spatial,
            spatial_token_lengths=spatial_token_lengths,
            input_ids=input_ids,
        )
        flat_spatial = self._flatten_projected_spatial(projected_spatial, spatial_token_lengths)

        # Build inputs_embeds + masked_scatter spatial tokens
        inputs_embeds = self.get_input_embeddings()(input_ids)
        spatial_mask = self._build_spatial_mask(input_ids, inputs_embeds)
        self._validate_spatial_mask_count(spatial_mask, flat_spatial, spatial_token_lengths)
        inputs_embeds = inputs_embeds.masked_scatter(
            spatial_mask,
            flat_spatial.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        kwargs["inputs_embeds"] = inputs_embeds
        # When inputs_embeds is provided, parent forward expects input_ids to
        # also be passed (for audio token id lookup in get_placeholder_mask).
        # Keep input_ids in kwargs/args.
        return super().forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # generate() bridge — forward spatial inputs through to forward()
    # ------------------------------------------------------------------
    # CRITICAL: without this override, upstream
    # Qwen3OmniMoeThinkerForConditionalGeneration.prepare_inputs_for_generation
    # does not know about spatial_audio / spatial_tokens / spatial_*_lengths.
    # Those kwargs reach generate() via collator's gen_* fields, get swallowed
    # by **kwargs in upstream's signature, and never make it back into
    # model_inputs — so our forward() receives spatial_audio=None, sees
    # _has_spatial_inputs == False, and short-circuits straight to
    # super().forward() with NO spatial signal injected. The trained model is
    # forced to answer azimuth/elevation/detect_time questions using only the
    # 16 kHz mono mel — i.e. random.
    #
    # The fix mirrors the Qwen2.5 reference (modeling_so_thinker.py):
    #   - During prefill (no past_key_values, or cache_position == 0): keep
    #     all spatial-related tensors so forward() can run the encoder +
    #     projector + masked_scatter exactly once on the prompt.
    #   - During decode (past_key_values is active and prepared seq len <= 1):
    #     clear spatial tensors so we don't recompute the encoder each token.
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        spatial_audio=None,
        spatial_audio_attention_mask=None,
        spatial_audio_lengths=None,
        spatial_tokens=None,
        spatial_token_lengths=None,
        # ``projected_spatial_tokens`` is accepted for forward-compat with
        # collators that ship the LLM-side projected embeddings directly. The
        # Qwen3 forward() below does not currently consume it (only
        # spatial_audio / spatial_tokens), but we forward it anyway so future
        # extensions don't need yet another generate-path patch.
        projected_spatial_tokens=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Splice spatial inputs back into model_inputs for forward().
        model_inputs["spatial_audio"] = spatial_audio
        model_inputs["spatial_audio_attention_mask"] = spatial_audio_attention_mask
        model_inputs["spatial_audio_lengths"] = spatial_audio_lengths
        model_inputs["spatial_tokens"] = spatial_tokens
        model_inputs["spatial_token_lengths"] = spatial_token_lengths
        model_inputs["projected_spatial_tokens"] = projected_spatial_tokens

        # Detect decode step. Two complementary signals:
        #   (a) Upstream's own decode-step detection: when use_cache is on and
        #       this is NOT the first iteration, upstream itself zeroes
        #       input_features (line 2306-2309 of upstream
        #       modeling_qwen3_omni_moe.py). We mirror that signal for spatial.
        #   (b) Defensive seq-len check: if prepared input_ids has length <= 1
        #       (autoregressive single-step) AND a cache exists AND no
        #       inputs_embeds were provided directly, treat as decode.
        # Either signal suffices — we want maximally aggressive clearing during
        # decode so the encoder is not invoked on every new token.
        prepared_input_ids = model_inputs.get("input_ids")
        prepared_inputs_embeds = model_inputs.get("inputs_embeds")
        prepared_seq_len = None
        if (
            prepared_input_ids is not None
            and hasattr(prepared_input_ids, "shape")
            and prepared_input_ids.ndim >= 2
        ):
            prepared_seq_len = int(prepared_input_ids.shape[1])

        is_decode_by_upstream = (not is_first_iteration) and bool(use_cache)
        is_decode_by_seqlen = bool(
            past_key_values is not None
            and prepared_seq_len is not None
            and prepared_seq_len <= 1
            and prepared_inputs_embeds is None
        )

        if is_decode_by_upstream or is_decode_by_seqlen:
            for key in (
                "spatial_audio",
                "spatial_audio_attention_mask",
                "spatial_audio_lengths",
                "spatial_tokens",
                "spatial_token_lengths",
                "projected_spatial_tokens",
            ):
                model_inputs[key] = None
        return model_inputs

    # ------------------------------------------------------------------
    # helpers (mostly identical to Qwen2.5 spatial impl, BEATs-only)
    # ------------------------------------------------------------------
    @staticmethod
    def _has_spatial_inputs(spatial_audio, spatial_tokens) -> bool:
        return spatial_audio is not None or spatial_tokens is not None

    def _resolve_spatial_tokens(
        self,
        spatial_audio,
        spatial_audio_attention_mask,
        spatial_audio_lengths,
        spatial_tokens,
        spatial_token_lengths,
    ):
        if spatial_tokens is not None:
            if spatial_tokens.ndim != 3:
                raise ValueError(
                    f"spatial_tokens must have shape [B, T_spat, D_spat], got {tuple(spatial_tokens.shape)}"
                )
            if spatial_token_lengths is None:
                spatial_token_lengths = spatial_tokens.new_full(
                    (spatial_tokens.shape[0],),
                    fill_value=spatial_tokens.shape[1],
                    dtype=torch.long,
                )
            return spatial_tokens, spatial_token_lengths

        if spatial_audio is None:
            raise ValueError(
                "spatial_audio is required for the so_backbone encoder path "
                "when spatial_tokens is not provided directly."
            )
        # When the model is loaded with device_map="auto" the parent thinker
        # has accelerate hooks that route activations across GPUs. The
        # so_encoder lives on a single device though, so we must
        # ensure inputs are on the encoder's device before calling forward.
        # Without this, training-time inputs may arrive on `meta` (a relic of
        # low_cpu_mem_usage init) and torchaudio's kaldi.fbank crashes when it
        # asks for an epsilon tensor on the meta device.
        try:
            enc_device = next(self.so_encoder.parameters()).device
        except StopIteration:
            enc_device = spatial_audio.device
        if spatial_audio.device != enc_device:
            spatial_audio = spatial_audio.to(enc_device)
            if spatial_audio_attention_mask is not None:
                spatial_audio_attention_mask = spatial_audio_attention_mask.to(enc_device)
            if spatial_audio_lengths is not None:
                spatial_audio_lengths = spatial_audio_lengths.to(enc_device)
        beats_output = self.so_encoder(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
        )
        return beats_output.spatial_tokens, beats_output.spatial_token_lengths

    def _build_spatial_mask(self, input_ids, inputs_embeds):
        if getattr(self.config, "spatial_token_index", None) is None:
            raise ValueError("config.spatial_token_index must be set before using the spatial thinker.")
        return (
            (input_ids == self.config.spatial_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

    @staticmethod
    def _flatten_projected_spatial(projected_spatial, spatial_token_lengths):
        if projected_spatial.ndim != 3:
            raise ValueError(
                f"projected_spatial must have shape [B, T_spat, D_llm], got {tuple(projected_spatial.shape)}"
            )
        if spatial_token_lengths.ndim != 1 or spatial_token_lengths.shape[0] != projected_spatial.shape[0]:
            raise ValueError(
                f"spatial_token_lengths must have shape [B], got {tuple(spatial_token_lengths.shape)}"
            )
        valid_rows = []
        max_tokens = projected_spatial.shape[1]
        for index, length in enumerate(spatial_token_lengths.tolist()):
            if length < 0 or length > max_tokens:
                raise ValueError(
                    f"spatial_token_lengths[{index}]={length} outside [0, {max_tokens}]"
                )
            if length == 0:
                continue
            valid_rows.append(projected_spatial[index, :length])
        if not valid_rows:
            return projected_spatial.new_zeros((0, projected_spatial.shape[-1]))
        return torch.cat(valid_rows, dim=0)

    def _align_projected_spatial_to_placeholders(
        self, projected_spatial, spatial_token_lengths, input_ids
    ):
        if getattr(self.config, "spatial_token_index", None) is None:
            raise ValueError("config.spatial_token_index must be set before using the spatial thinker.")
        placeholder_counts = (input_ids == self.config.spatial_token_index).sum(dim=1).to(
            device=spatial_token_lengths.device, dtype=torch.long
        )
        if torch.equal(placeholder_counts, spatial_token_lengths):
            return projected_spatial, spatial_token_lengths

        batch_size, _, hidden_dim = projected_spatial.shape
        target_max = int(placeholder_counts.max().item()) if batch_size > 0 else 0
        aligned = projected_spatial.new_zeros((batch_size, target_max, hidden_dim))
        source_max = projected_spatial.shape[1]
        for index, (src_len, tgt_len) in enumerate(
            zip(spatial_token_lengths.tolist(), placeholder_counts.tolist())
        ):
            if src_len < 0 or src_len > source_max:
                raise ValueError(f"spatial_token_lengths[{index}]={src_len} outside [0, {source_max}]")
            if tgt_len <= 0:
                continue
            copy_len = min(src_len, tgt_len)
            if copy_len > 0:
                aligned[index, :copy_len] = projected_spatial[index, :copy_len]
            if tgt_len > src_len and src_len > 0:
                aligned[index, copy_len:tgt_len] = projected_spatial[index, src_len - 1].unsqueeze(0)
        return aligned, placeholder_counts

    def _validate_spatial_mask_count(self, spatial_mask, projected_spatial, spatial_token_lengths):
        expected = int(spatial_token_lengths.sum().item())
        actual = int(spatial_mask[..., 0].sum().item())
        if actual != expected:
            raise ValueError(
                f"Spatial placeholder count does not match projected token count: {actual} vs {expected}"
            )
        if projected_spatial.ndim != 2:
            raise ValueError(
                f"Packed projected spatial tokens must be [sum(T_i), D_llm], got {tuple(projected_spatial.shape)}"
            )


__all__ = [
    "Qwen3OmniMoeSpatialThinkerForConditionalGeneration",
    "Qwen3OmniMoeSpatialForConditionalGeneration",
]


class Qwen3OmniMoeSpatialForConditionalGeneration(
    Qwen3OmniMoeSpatialThinkerForConditionalGeneration
):
    """Top-level wrapper that mimics the Qwen2.5 ``model.thinker`` shape.

    For Qwen3 we wrap the thinker only (no talker). The Qwen2.5 train script
    accesses spatial submodules via ``model.thinker.so_encoder`` and
    calls ``model.disable_talker()``. This wrapper makes both work without
    changing the underlying behavior:

      - ``self.thinker`` returns ``self`` (so ``model.thinker.X`` == ``model.X``)
      - ``disable_talker()`` is a no-op (the Qwen3 talker is never built here)

    Because Qwen3's top-level ``Qwen3OmniMoeForConditionalGeneration`` config
    has known bugs and we don't need the talker, we expose the thinker
    directly as the top-level model.
    """

    config_class = Qwen3OmniMoeSpatialThinkerConfig

    @property
    def thinker(self):
        return self

    def disable_talker(self):
        return None

