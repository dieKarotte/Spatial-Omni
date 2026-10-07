"""SO spatial-modality wrapper around the official Phi4MMForCausalLM.

The official Phi-4-multimodal framework models each modality as:
    placeholder tokens in input_ids  →  modality embed module  →  index_put
    into ``inputs_embeds`` (``Phi4MMImageAudioEmbedding``), plus a
    modality-specific LLM LoRA adapter selected by ``input_mode``.

SO adds a third modality through the same embedding interface:
    ``<|spatial|>`` placeholders (expanded by SoPhi4Processor) are replaced
    with SO-Encoder (Spatial-BEATs, FOA) outputs passed through a
    pixel-shuffle projector (10 Hz → 2.5 Hz LLM-side rate).

Spatial token alignment:
    * placeholders are expanded in text before tokenization, so each
      ``<|spatial|>`` occupies exactly one token id in ``input_ids``;
    * forward asserts sum(spatial_token_lengths) == #spatial positions after
      placeholder-count alignment, then uses index_put with the projected
      encoder embeddings.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import CausalLMOutputWithPast

from spatial_omni.modules.so_encoder import SOEncoder
from spatial_omni.modules.so_token_projector import build_so_token_projector

# InputMode values from the official processing_phi4mm.py
_INPUT_MODE_LANGUAGE = 0
_INPUT_MODE_VISION = 1
_INPUT_MODE_SPEECH = 2
_INPUT_MODE_VISION_SPEECH = 3

SO_ADAPTER_NAME = "so"
_PRETRAINED_ADAPTERS = ("speech", "vision")

_BASE_CLASS_CACHE = {}
_SO_CLASS_CACHE = {}


def load_base_phi4mm_class(model_dir: str):
    """Load the official Phi4MMForCausalLM via HF dynamic-module machinery
    (handles the checkpoint's relative imports without vendoring copies)."""
    if model_dir not in _BASE_CLASS_CACHE:
        inner = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMModel", model_dir
        )
        if not hasattr(inner, "prepare_inputs_for_generation"):
            # peft>=0.17 PeftModelForCausalLM stores this attr from the wrapped
            # decoder; the vendored code predates that requirement. Stored but
            # never invoked on the inner model.
            inner.prepare_inputs_for_generation = lambda self, *a, **kw: {}
        _BASE_CLASS_CACHE[model_dir] = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMForCausalLM", model_dir
        )
    return _BASE_CLASS_CACHE[model_dir]


def get_so_phi4_class(model_dir: str):
    """Build (and cache) the SoPhi4MMForCausalLM subclass for this base."""
    if model_dir in _SO_CLASS_CACHE:
        return _SO_CLASS_CACHE[model_dir]
    base = load_base_phi4mm_class(model_dir)

    class SoPhi4MMForCausalLM(base):  # noqa: D401 - documented at module level
        """Official Phi4MMForCausalLM + SO spatial branch."""

        def __init__(
            self,
            config,
            so_checkpoint_path: str = "",
            so_beats_repo: str = "",
            so_freeze_backbone: bool = True,
            so_encoder_token_rate: float = 10.0,
            so_projector_hidden_dim: int = 768,
            so_projector_shuffle_factor: int = 4,
            so_max_audio_seconds: float = 20.0,
            so_enable_replay: bool = False,
            so_null_alignment_weight: float = 0.05,
            **kwargs,
        ):
            super().__init__(config)
            self.spatial_token_id: Optional[int] = None
            self._so_adapter_added = False
            # Adapter activation modes:
            #   both        - default: "so" stays active alongside the
            #                 requested pretrained adapter on every forward pass,
            #                 including official-forward decode steps.
            #   speech_only - pretrained adapter only (so disabled) [ablation]
            #   so_only     - "so" only [ablation]
            #   legacy      - follow the base-model adapter selection [ablation]
            self.so_adapter_mode = "both"
            self._pretrained_merged = False
            self._last_applied_adapters = None
            self.so_encoder = SOEncoder(
                checkpoint_path=so_checkpoint_path,
                beats_repo_path=so_beats_repo or None,
                freeze_backbone=so_freeze_backbone,
                max_audio_seconds=so_max_audio_seconds,
                encoder_token_rate=so_encoder_token_rate,
            )
            self.so_projector = build_so_token_projector(
                projector_type="pixel_shuffle",
                input_dim=self.so_encoder.encoder_dim,
                hidden_dim=so_projector_hidden_dim,
                output_dim=config.hidden_size,
                shuffle_factor=so_projector_shuffle_factor,
            )
            # Mono-replay support: learned null spatial token bank (mirrors the
            # Qwen SO design). Filled into <|spatial|> placeholders of mono
            # replay samples; the encoder's W-only output is aligned to it with
            # an MSE loss so mono input maps to the null state.
            self.null_alignment_weight = float(so_null_alignment_weight)
            if so_enable_replay:
                num_null = int(round(so_max_audio_seconds * so_encoder_token_rate
                                     / so_projector_shuffle_factor))
                self.spatial_null = torch.nn.Parameter(
                    torch.randn(num_null, config.hidden_size) * 0.02
                )
            else:
                self.spatial_null = None

        # ------------------------------------------------------------------
        # tokenizer sync
        # ------------------------------------------------------------------
        def set_spatial_token_id(self, token_id: int) -> None:
            self.spatial_token_id = int(token_id)

        # ------------------------------------------------------------------
        # LoRA adapter management
        # ------------------------------------------------------------------
        def add_so_adapter(self, r: int = 16, alpha: int = 32, dropout: float = 0.05):
            """Inject a third LoRA adapter ("so") next to the pretrained
            speech/vision adapters, on the same modules (i.e. every existing
            LoraLayer in the decoder)."""
            if self._so_adapter_added:
                return
            from peft.tuners.lora.layer import LoraLayer

            count = 0
            for module in self.model.modules():
                if isinstance(module, LoraLayer):
                    module.update_layer(
                        SO_ADAPTER_NAME,
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        init_lora_weights=True,
                        use_rslora=False,
                        use_dora=False,
                        lora_bias=False,
                    )
                    count += 1
            if count == 0:
                raise RuntimeError("no LoraLayer modules found to attach the 'so' adapter")
            self._so_adapter_added = True
            self._refreeze_pretrained_adapters()

        def set_adapter_mode(self, mode: str) -> None:
            """Switch adapter-activation policy (see __init__ docstring)."""
            allowed = {"both", "speech_only", "so_only", "legacy"}
            if mode not in allowed:
                raise ValueError(f"so_adapter_mode must be one of {allowed}, got {mode!r}")
            self.so_adapter_mode = mode
            self._last_applied_adapters = None  # force re-apply on next forward

        def merge_pretrained_adapters_into_base(self) -> None:
            """Bake the frozen speech LoRA into the base weights and delete
            the pretrained adapters, leaving "so" as the only adapter.

            Must run on CPU right after from_pretrained (fp32 upcast path in
            peft get_delta_weight) and BEFORE .to(device). peft 0.17.1
            pitfalls handled explicitly:
              * merge() appends to module.merged_adapters; the official
                set_lora_adapter auto-unmerges merged modules and
                LoraLayer.forward short-circuits to base-only while
                merged_adapters is non-empty -> the list MUST be cleared.
              * delete_adapter() does not touch merged_adapters.
            """
            if self._pretrained_merged:
                return
            from peft.tuners.lora.layer import LoraLayer

            count = 0
            for module in self.model.modules():
                if not isinstance(module, LoraLayer):
                    continue
                assert hasattr(module, "merged_adapters"), \
                    "peft internals changed: LoraLayer.merged_adapters missing"
                # merge ONLY speech (the always-active audio adapter); vision
                # was never active in the audio pipeline -> delete unmerged,
                # otherwise the function would change vs the dynamic model.
                if "speech" in module.lora_A.keys():
                    module.merge(adapter_names=["speech"])
                module.merged_adapters.clear()
                for name in _PRETRAINED_ADAPTERS:
                    if name in module.lora_A.keys():
                        module.delete_adapter(name)
                if SO_ADAPTER_NAME in module.lora_A.keys():
                    module.set_adapter([SO_ADAPTER_NAME])
                count += 1
            if count == 0:
                raise RuntimeError("merge_pretrained_adapters_into_base: no LoraLayer found")
            self._pretrained_merged = True
            self._last_applied_adapters = None
            print(f"[so-phi4] merged speech LoRA into base on {count} modules; "
                  f"'{SO_ADAPTER_NAME}' is now the only adapter", flush=True)

        def _refreeze_pretrained_adapters(self) -> None:
            """peft's set_adapter() flips requires_grad on every activation;
            keep the pretrained speech/vision adapters frozen regardless."""
            for name, param in self.named_parameters():
                if "lora_" not in name:
                    continue
                if f".{SO_ADAPTER_NAME}." in name:
                    continue
                for pretrained in _PRETRAINED_ADAPTERS:
                    if f".{pretrained}." in name:
                        param.requires_grad_(False)

        def _effective_adapters(self, requested):
            """Map any official/internal activation request to the effective
            adapter set according to ``so_adapter_mode``.

            ``requested`` is a list of adapter names or None (= unset). After
            ``merge_pretrained_adapters_into_base`` only "so" exists, so
            pretrained names are dropped from the mapped set automatically.
            """
            mode = self.so_adapter_mode
            so_ready = self._so_adapter_added
            req = [] if requested is None else [
                a for a in (requested if isinstance(requested, (list, tuple))
                            else [requested])
            ]
            pretrained = [a for a in req if a != SO_ADAPTER_NAME]
            if self._pretrained_merged:
                pretrained = []  # merged into base weights; adapters deleted
            if mode == "legacy" or not so_ready:
                out = req if not self._pretrained_merged else \
                    [a for a in req if a == SO_ADAPTER_NAME]
                return out or None
            if mode == "speech_only":
                return pretrained or None
            if mode == "so_only":
                return [SO_ADAPTER_NAME]
            # Keep the spatial adapter active alongside the requested base adapter.
            return [*pretrained, SO_ADAPTER_NAME]

        def _apply_adapters(self, adapters) -> None:
            key = tuple(adapters) if adapters else None
            if key == self._last_applied_adapters:
                return
            if adapters:
                super().set_lora_adapter(list(adapters))
            else:
                super().unset_lora_adapter()
            self._last_applied_adapters = key
            # peft set_adapter flips requires_grad on activation; keep the
            # pretrained adapters frozen. Skip under no_grad (decode steps).
            if torch.is_grad_enabled():
                self._refreeze_pretrained_adapters()

        def set_lora_adapter(self, adapter_name) -> None:
            self._apply_adapters(self._effective_adapters(adapter_name))

        def unset_lora_adapter(self) -> None:
            self._apply_adapters(self._effective_adapters(None))

        def _select_adapters(self, input_mode) -> str:
            """Mirror the official input_mode → adapter routing, adding the
            "so" adapter so it is active for every forward."""
            if torch.is_tensor(input_mode):
                mode_value = int(input_mode.flatten()[0].item())
            else:
                mode_value = int(input_mode)
            so = [SO_ADAPTER_NAME] if self._so_adapter_added else []
            if mode_value in (_INPUT_MODE_VISION, _INPUT_MODE_VISION_SPEECH):
                self.set_lora_adapter(["vision", *so])
                return "vision"
            if mode_value == _INPUT_MODE_SPEECH:
                self.set_lora_adapter(["speech", *so])
                return "speech"
            if so:
                self.set_lora_adapter(so)
            else:
                self.unset_lora_adapter()
            return "speech"

        # ------------------------------------------------------------------
        # spatial helpers
        # ------------------------------------------------------------------
        def _align_to_placeholders(self, projected, lengths, input_ids):
            """Truncate / edge-pad projected rows so per-sample counts equal
            the actual <|spatial|> placeholder count in input_ids."""
            counts = (input_ids == self.spatial_token_id).sum(dim=1).to(
                device=lengths.device, dtype=torch.long
            )
            if torch.equal(counts, lengths):
                return projected, lengths
            batch, _, hidden = projected.shape
            target_max = int(counts.max().item()) if batch > 0 else 0
            aligned = projected.new_zeros((batch, target_max, hidden))
            src_max = projected.shape[1]
            for i, (src_len, tgt_len) in enumerate(zip(lengths.tolist(), counts.tolist())):
                if src_len < 0 or src_len > src_max:
                    raise ValueError(f"spatial length[{i}]={src_len} outside [0,{src_max}]")
                if tgt_len == 0:
                    continue
                copy_len = min(src_len, tgt_len)
                if copy_len > 0:
                    aligned[i, :copy_len] = projected[i, :copy_len]
                if tgt_len > src_len and src_len > 0:
                    aligned[i, copy_len:tgt_len] = projected[i, src_len - 1].unsqueeze(0)
            return aligned, counts

        # ------------------------------------------------------------------
        # forward with spatial injection
        # ------------------------------------------------------------------
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_mode=None,
            input_audio_embeds: Optional[torch.FloatTensor] = None,
            audio_embed_sizes=None,
            audio_attention_mask=None,
            spatial_audio: Optional[torch.FloatTensor] = None,
            spatial_audio_attention_mask: Optional[torch.FloatTensor] = None,
            spatial_audio_lengths: Optional[torch.LongTensor] = None,
            spatial_token_lengths: Optional[torch.LongTensor] = None,
            has_spatial: Optional[torch.BoolTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            **kwargs,
        ):
            if num_logits_to_keep is None:  # GenerationMixin passes None
                num_logits_to_keep = 0
            has_placeholders = (
                spatial_audio is not None
                and self.spatial_token_id is not None
                and input_ids is not None
                and bool((input_ids == self.spatial_token_id).any())
            )
            if not has_placeholders:
                return super().forward(
                    input_ids=input_ids,
                    input_mode=input_mode,
                    input_audio_embeds=input_audio_embeds,
                    audio_embed_sizes=audio_embed_sizes,
                    audio_attention_mask=audio_attention_mask,
                    labels=labels,
                    num_logits_to_keep=num_logits_to_keep,
                    **kwargs,
                )
            if input_ids is None:
                raise ValueError("input_ids required for spatial injection")

            audio_projection_mode = self._select_adapters(input_mode)

            # --- SO branch: FOA → encoder → pixel-shuffle projector --------
            # waveform stays float32: the encoder's STFT has no bf16 cuFFT
            # kernel; PixelShuffleProjector casts activations to LLM dtype.
            enc = self.so_encoder(
                spatial_audio,
                spatial_audio_attention_mask,
                spatial_audio_lengths,
            )
            projected = self.so_projector(enc.spatial_tokens)
            k = int(getattr(self.so_projector, "shuffle_factor", 1))
            if k > 1:
                lens = torch.div(enc.spatial_token_lengths, k, rounding_mode="floor")
                lens = torch.clamp(lens, min=0, max=int(projected.shape[1]))
                lens = torch.where(
                    (enc.spatial_token_lengths > 0) & (lens == 0),
                    torch.ones_like(lens),
                    lens,
                )
            else:
                lens = enc.spatial_token_lengths
            projected, lens = self._align_to_placeholders(projected, lens, input_ids)

            # --- mixed replay: null fill + W-only MSE alignment ------------
            loss_null = None
            if has_spatial is not None:
                if self.spatial_null is None:
                    raise ValueError(
                        "has_spatial given but model was built without "
                        "so_enable_replay=True (spatial_null missing)"
                    )
                has_spatial = has_spatial.to(device=input_ids.device, dtype=torch.bool)
                null_bank = self.spatial_null.to(
                    device=projected.device, dtype=projected.dtype
                )
                replay_rows = (~has_spatial).nonzero(as_tuple=True)[0]
                # Deterministically mix learned null and W-only encoder fills
                # for replay rows. The ratio is P(null); zero disables null fill.
                ratio = float(getattr(self, "replay_null_ratio", 0.5))
                use_null = {}
                for i in replay_rows.tolist():
                    if self.training:
                        coin = int(input_ids[i].sum().item()) % 100
                        use_null[i] = coin < int(ratio * 100)
                    else:
                        use_null[i] = True  # eval keeps explicit null semantics
                null_rows = [i for i in replay_rows.tolist() if use_null[i]]
                if null_rows:
                    w_flat = torch.cat(
                        [projected[i, : lens[i]] for i in null_rows if lens[i] > 0]
                        or [projected.new_zeros((0, projected.shape[-1]))],
                        dim=0,
                    )
                    t_flat = torch.cat(
                        [null_bank[: lens[i]].detach() for i in null_rows if lens[i] > 0]
                        or [projected.new_zeros((0, projected.shape[-1]))],
                        dim=0,
                    )
                    if w_flat.numel() > 0:
                        loss_null = torch.nn.functional.mse_loss(w_flat, t_flat)
                final_rows = []
                for i in range(projected.shape[0]):
                    if has_spatial[i] or not use_null.get(i, False):
                        final_rows.append(projected[i, : lens[i]])  # encoder fill
                    else:
                        final_rows.append(null_bank[: lens[i]])
                merged = torch.cat([r for r in final_rows if r.shape[0] > 0], dim=0)
            else:
                merged = torch.cat(
                    [projected[i, : lens[i]] for i in range(projected.shape[0]) if lens[i] > 0],
                    dim=0,
                )

            # --- base embeddings with official audio injection -------------
            inputs_embeds = self.model.embed_tokens_extend(
                input_ids=input_ids,
                input_embeds=None,
                input_image_embeds=None,
                input_audio_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                audio_projection_mode=audio_projection_mode,
                wte=self.model.embed_tokens,
            )

            # --- spatial index_put (real embeddings, not token rows) -------
            positions = torch.nonzero(input_ids == self.spatial_token_id, as_tuple=True)
            if merged.shape[0] != positions[0].shape[0]:
                raise RuntimeError(
                    f"spatial injection mismatch: {merged.shape[0]} embeddings vs "
                    f"{positions[0].shape[0]} placeholder positions"
                )
            with torch.autocast(device_type=inputs_embeds.device.type, enabled=False):
                inputs_embeds = inputs_embeds.index_put(
                    positions, merged.to(dtype=inputs_embeds.dtype)
                )

            # --- decoder + LM head + loss (mirrors official tail) ----------
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                **{k: v for k, v in kwargs.items() if k in (
                    "attention_mask", "position_ids", "past_key_values",
                    "output_attentions", "output_hidden_states", "return_dict",
                    "cache_position", "use_cache",
                )},
            )
            hidden_states = outputs[0]
            keep = num_logits_to_keep or 0
            logits = self.lm_head(hidden_states[:, -keep:, :])
            loss = None
            if labels is not None:
                loss = self.loss_function(logits, labels, self.vocab_size)
            if loss_null is not None:
                null_term = self.null_alignment_weight * loss_null.to(logits.device)
                loss = null_term if loss is None else loss + null_term
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

    SoPhi4MMForCausalLM.__name__ = "SoPhi4MMForCausalLM"
    _SO_CLASS_CACHE[model_dir] = SoPhi4MMForCausalLM
    return SoPhi4MMForCausalLM
