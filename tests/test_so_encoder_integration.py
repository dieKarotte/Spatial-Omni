#!/usr/bin/env python3
"""Integration tests for the SO-Encoder (10 Hz native → 2.5 Hz after pixel
shuffle).

Usage:
    PYTHONPATH=. python tests/test_so_encoder_integration.py

The encoder-forward checks (single-clip / variable-batch / end-to-end) require
a pretrained SO-Encoder checkpoint. Set ``SO_ENCODER_CKPT`` to the path of a
``best.pt`` produced by :mod:`spatial_omni.encoders.beats.train_so_pretrain`,
or place one at ``$SO_BEATS_REPO/checkpoints/so_encoder/best.pt``. When no
checkpoint is found those tests are SKIPPED (not failed) so the projector /
mask / flatten checks still run as a bare-install smoke.
"""

import os
import sys
import traceback

# --- Setup paths ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEATS_REPO = os.environ.get(
    "SO_BEATS_REPO",
    os.path.normpath(os.path.join(REPO_ROOT, "..", "unilm", "beats")),
)

sys.path.insert(0, REPO_ROOT)
if os.path.isdir(BEATS_REPO):
    sys.path.insert(0, BEATS_REPO)

import torch

# Resolve the encoder checkpoint. Priority: env var SO_ENCODER_CKPT,
# then a conventional path under the BEATs repo. When neither exists the
# encoder-forward tests are skipped (not failed).
CHECKPOINT_PATH = os.environ.get("SO_ENCODER_CKPT", "")
if not CHECKPOINT_PATH:
    CHECKPOINT_PATH = os.path.join(BEATS_REPO, "checkpoints", "so_encoder", "best.pt")

SAMPLE_RATE = 16000
LLM_HIDDEN_SIZE = 3584
ENCODER_DIM = 768
PROJECTOR_HIDDEN_DIM = 768
ENCODER_TOKEN_RATE = 10.0   # SO-Encoder native
LLM_TOKEN_RATE = 2.5        # after pixel-shuffle k=4
SHUFFLE_FACTOR = 4

results = []


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    msg = f"[{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def test_encoder_single_clip():
    """Single 10s FOA clip → ``round(10 * encoder_token_rate)`` tokens.

    The encoder's native frame rate is read from the loaded checkpoint
    (``cfg.target_token_rate``) so this test works for any preset (e.g. 10 Hz
    native + projector pixel-shuffle, or 2.5 Hz native with no shuffle).
    """
    from spatial_omni.modules.so_encoder import (
        SOEncoder, SOEncoderOutput,
    )
    wrapper = SOEncoder(
        checkpoint_path=CHECKPOINT_PATH, beats_repo_path=BEATS_REPO,
        freeze_backbone=True, max_audio_seconds=20.0,
        encoder_token_rate=ENCODER_TOKEN_RATE,
    )
    wrapper._build_model()
    # The actual emitted rate is the loaded checkpoint's ``target_token_rate``
    # (the constructor arg is just a hint). Read it back so this test works
    # for any preset (10 Hz native, 2.5 Hz native, ...).
    cfg = getattr(wrapper, "_beats_cfg", None)
    native_rate = float(getattr(cfg, "target_token_rate", None) or
                        getattr(wrapper, "encoder_token_rate", ENCODER_TOKEN_RATE))
    B, T_audio = 1, SAMPLE_RATE * 10
    spatial_audio = torch.randn(B, T_audio, 4)
    with torch.no_grad():
        out = wrapper(spatial_audio=spatial_audio)
    assert isinstance(out, SOEncoderOutput)
    expected_tokens = int(round(10 * native_rate))
    assert out.spatial_tokens.shape == (1, expected_tokens, ENCODER_DIM), \
        f"Expected [1, {expected_tokens}, {ENCODER_DIM}], got {list(out.spatial_tokens.shape)}"
    assert out.spatial_token_lengths[0].item() == expected_tokens
    report(f"encoder_single_clip_{native_rate:g}hz", True,
           f"shape={list(out.spatial_tokens.shape)}, length={out.spatial_token_lengths.tolist()}")
    return wrapper


def test_encoder_variable_batch(wrapper):
    """Variable-length FOA → per-sample encoder token lengths."""
    cfg = getattr(wrapper, "_beats_cfg", None)
    native_rate = float(getattr(cfg, "target_token_rate", None) or
                        getattr(wrapper, "encoder_token_rate", ENCODER_TOKEN_RATE))
    B, T_max = 2, SAMPLE_RATE * 10
    spatial_audio = torch.randn(B, T_max, 4)
    lengths = torch.tensor([SAMPLE_RATE * 10, SAMPLE_RATE * 8], dtype=torch.long)
    with torch.no_grad():
        out = wrapper(spatial_audio=spatial_audio, spatial_audio_lengths=lengths)
    expected_lengths = [
        int(round(10 * native_rate)),
        int(round(8 * native_rate)),
    ]
    actual_lengths = out.spatial_token_lengths.tolist()
    assert out.spatial_tokens.shape[0] == B
    assert out.spatial_tokens.shape[1] == int(round(10 * native_rate))
    assert out.spatial_tokens.shape[2] == ENCODER_DIM
    assert actual_lengths == expected_lengths, \
        f"Expected {expected_lengths}, got {actual_lengths}"
    report(f"encoder_variable_batch_2_{native_rate:g}hz", True,
           f"shape={list(out.spatial_tokens.shape)}, lengths={actual_lengths}")


def test_projector_pixel_shuffle_k4():
    """pixel_shuffle(k=4) collapses 10 Hz → 2.5 Hz and maps 768 → 3584."""
    from spatial_omni.modules.so_token_projector import (
        build_so_token_projector,
    )
    proj = build_so_token_projector(
        projector_type="pixel_shuffle",
        input_dim=ENCODER_DIM,
        hidden_dim=PROJECTOR_HIDDEN_DIM,
        output_dim=LLM_HIDDEN_SIZE,
        shuffle_factor=SHUFFLE_FACTOR,
    )
    B, T_s_enc = 2, int(round(20 * ENCODER_TOKEN_RATE))  # 200
    tokens = torch.randn(B, T_s_enc, ENCODER_DIM)
    out = proj(tokens)
    expected_T = T_s_enc // SHUFFLE_FACTOR  # 50
    assert out.shape == (B, expected_T, LLM_HIDDEN_SIZE), \
        f"Expected [{B}, {expected_T}, {LLM_HIDDEN_SIZE}], got {list(out.shape)}"
    report("projector_pixel_shuffle_k4", True,
           f"in={list(tokens.shape)} → out={list(out.shape)}")


def test_encoder_plus_projector_end_to_end(wrapper):
    """Wire encoder → pixel_shuffle(k=4) → [B, native_rate*10/4, 3584] for a 10s clip.

    For 10 Hz native the projected length is 25; for 2.5 Hz
    native, no shuffle (k=1) is used so the shape passes through.
    """
    from spatial_omni.modules.so_token_projector import (
        build_so_token_projector,
    )
    native_rate = float(getattr(getattr(wrapper, "_beats_cfg", None), "target_token_rate", None) or
                        getattr(wrapper, "encoder_token_rate", ENCODER_TOKEN_RATE))
    # Pick a shuffle factor that lands on the LLM's 2.5 Hz rate.
    shuffle = max(1, int(round(native_rate / LLM_TOKEN_RATE)))
    proj = build_so_token_projector(
        projector_type="pixel_shuffle",
        input_dim=ENCODER_DIM,
        hidden_dim=PROJECTOR_HIDDEN_DIM,
        output_dim=LLM_HIDDEN_SIZE,
        shuffle_factor=shuffle,
    )
    B, T_audio = 1, SAMPLE_RATE * 10
    spatial_audio = torch.randn(B, T_audio, 4)
    with torch.no_grad():
        enc_out = wrapper(spatial_audio=spatial_audio)
        projected = proj(enc_out.spatial_tokens)
    expected_T = int(round(10 * native_rate)) // shuffle
    assert projected.shape == (B, expected_T, LLM_HIDDEN_SIZE), \
        f"Expected [1, {expected_T}, {LLM_HIDDEN_SIZE}], got {list(projected.shape)}"
    report("encoder_to_projector_end_to_end", True,
           f"10s clip → encoder {list(enc_out.spatial_tokens.shape)} → projector {list(projected.shape)}")


def test_spatial_mask():
    SPATIAL_TOKEN_ID = 999999
    B, T_text, T_s = 2, 80, 50  # 20s clip at 2.5 Hz = 50 placeholders
    input_ids = torch.randint(0, 1000, (B, T_text))
    for b in range(B):
        input_ids[b, 10:10 + T_s] = SPATIAL_TOKEN_ID
    mask = (input_ids == SPATIAL_TOKEN_ID)
    count = mask.sum().item()
    expected = B * T_s
    assert count == expected, f"Expected {expected} spatial positions, got {count}"
    report("spatial_mask_building", True, f"mask_count={count}, expected={expected}")


def test_flatten():
    B, T_max = 3, 50
    lengths = torch.tensor([50, 40, 25], dtype=torch.long)
    projected = torch.randn(B, T_max, LLM_HIDDEN_SIZE)
    valid_rows = [projected[i, :l] for i, l in enumerate(lengths.tolist())]
    flat = torch.cat(valid_rows, dim=0)
    expected_total = lengths.sum().item()
    assert flat.shape == (expected_total, LLM_HIDDEN_SIZE), \
        f"Expected [{expected_total}, {LLM_HIDDEN_SIZE}], got {list(flat.shape)}"
    report("flatten_variable_lengths", True,
           f"flat_shape={list(flat.shape)}, total_tokens={expected_total}")


def test_no_interference():
    SPATIAL_TOKEN_ID = 999999
    B, T_text, T_s = 1, 80, 25
    input_ids = torch.randint(0, 1000, (B, T_text))
    input_ids[0, 5:5 + T_s] = SPATIAL_TOKEN_ID
    inputs_embeds = torch.randn(B, T_text, LLM_HIDDEN_SIZE)
    original_embeds = inputs_embeds.clone()
    projected_flat = torch.randn(T_s, LLM_HIDDEN_SIZE)
    spatial_mask = (input_ids == SPATIAL_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
    result = inputs_embeds.masked_scatter(spatial_mask, projected_flat)
    non_spatial_mask = ~spatial_mask
    assert torch.allclose(
        result[non_spatial_mask], original_embeds[non_spatial_mask]
    ), "Non-spatial positions were modified!"
    assert not torch.allclose(
        result[spatial_mask].view(T_s, LLM_HIDDEN_SIZE),
        original_embeds[0, 5:5 + T_s],
    ), "Spatial positions were NOT replaced!"
    report("no_interference_masked_scatter", True, "non-spatial positions preserved")


if __name__ == "__main__":
    print("=" * 70)
    print("SO-Encoder Integration Tests (encoder 10 Hz, LLM 2.5 Hz)")
    print("=" * 70)
    print()

    skipped = []
    if not os.path.exists(CHECKPOINT_PATH):
        msg = (f"no SO-Encoder checkpoint at {CHECKPOINT_PATH} — set "
               f"SO_ENCODER_CKPT to skip this notice")
        skipped.extend([
            ("encoder_single_clip_10hz", msg),
            ("encoder_variable_batch_2_10hz", msg),
            ("encoder_to_projector_end_to_end", msg),
        ])
        wrapper = None
        for name, m in skipped:
            print(f"[SKIP] {name} -- {m}")
    else:
        try:
            wrapper = test_encoder_single_clip()
        except Exception as e:
            report("encoder_single_clip_10hz", False, f"{e}")
            traceback.print_exc()
            wrapper = None

    if wrapper is not None:
        for test_fn in [test_encoder_variable_batch, test_encoder_plus_projector_end_to_end]:
            try:
                test_fn(wrapper)
            except Exception as e:
                report(test_fn.__name__.replace("test_", ""), False, f"{e}")
                traceback.print_exc()

    for test_fn in [test_projector_pixel_shuffle_k4, test_spatial_mask, test_flatten, test_no_interference]:
        try:
            test_fn()
        except Exception as e:
            report(test_fn.__name__.replace("test_", ""), False, f"{e}")
            traceback.print_exc()

    print()
    print("=" * 70)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Results: {passed}/{total} passed, {len(skipped)} skipped")
    if passed < total:
        for name, p in results:
            if not p:
                print(f"  FAILED: {name}")
    sys.exit(0 if passed == total else 1)
    print("=" * 70)
