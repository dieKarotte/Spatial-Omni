"""诊断测试：验证 run_validation_generation 中左填充 batch 的 decode 偏移 bug。

Bug 位置：train_so_qa.py 的 run_validation_generation()
  pls = gi["attention_mask"].sum(1).tolist()
  ...
  pt = processor.tokenizer.decode(gen[i, int(pl):], ...)   ← 错误！

正确逻辑：左填充 batch 中，generate() 输出的新 token 从 ml = gi["input_ids"].shape[1] 开始，
          对 batch 内所有样本统一，与各样本 prefix 长度 pl_i 无关。

运行方式：
    cd ${SO_REPO}
    python tests/test_generation_decode_offset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


# =========================================================================
# Part 1: 纯逻辑验证（不需要加载模型）
# =========================================================================

def test_left_padded_decode_offset():
    """验证左填充 batch 中新 token 起点是 ml，而非各自的 pl_i。

    模拟场景：
        batch_size = 3
        前缀长度：[600, 650, 700]（最长 ml = 700）
        左填充后所有序列长 700
        generate() 假设各生成 3 个新 token：
            sample 0: token [11, 12, 13]
            sample 1: token [21, 22, 23]
            sample 2: token [31, 32, 33]
    """
    pad_id = 0
    pl_list = [600, 650, 700]
    ml = max(pl_list)
    fake_prefix_text_start = 550  # AUDIO(500) + spatial(50) 之后就是文本 token
    AUDIO_ID = 151646  # 假设
    SPATIAL_ID = 151665

    # 构造 gi（左填充后的 input_ids）
    B = len(pl_list)
    gi = torch.full((B, ml), fill_value=pad_id, dtype=torch.long)
    gm = torch.zeros(B, ml, dtype=torch.long)
    for i, pl in enumerate(pl_list):
        s = ml - pl
        # 填充前缀 token：500个AUDIO + 50个spatial + 文本
        prefix = (
            [AUDIO_ID] * 500
            + [SPATIAL_ID] * 50
            + list(range(1000 + i * 100, 1000 + i * 100 + pl - 550))  # 文本 token
        )
        assert len(prefix) == pl, f"prefix len mismatch: {len(prefix)} vs {pl}"
        gi[i, s:] = torch.tensor(prefix, dtype=torch.long)
        gm[i, s:] = 1

    # 模拟 generate() 输出：[B, ml + max_new_tokens]
    max_new_tokens = 3
    new_token_ids = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
    gen = torch.cat([gi, torch.zeros(B, max_new_tokens, dtype=torch.long)], dim=1)
    for i, new_toks in enumerate(new_token_ids):
        gen[i, ml: ml + max_new_tokens] = torch.tensor(new_toks, dtype=torch.long)

    pls = gm.sum(1).tolist()  # = [600, 650, 700]

    print("=" * 60)
    print("测试：左填充 batch 的 decode 起点")
    print("=" * 60)
    print(f"ml = {ml}, pl_list = {pl_list}")
    print()

    print("【当前错误的截取方式：gen[i, pl_i:]】")
    for i, pl in enumerate(pls):
        segment = gen[i, int(pl):]
        # 过滤 AUDIO、SPATIAL token（模拟 skip_special_tokens）
        filtered = [t.item() for t in segment if t.item() not in (AUDIO_ID, SPATIAL_ID, pad_id)]
        new_only = new_token_ids[i]
        # 找前缀文本 token 范围
        s = ml - int(pl)
        prefix_text_tokens_included = gen[i, int(pl): ml].tolist()
        print(f"  sample {i}: pl={pl}, s={s}, gen[i,{pl}:{ml}]={prefix_text_tokens_included[:10]}... "
              f"(来自前缀，不应该出现!) + new_tokens={gen[i,ml:].tolist()}")
        if s > 0:
            print(f"    ❌ 错误：包含了前缀文本 token（s={s}个），被错误地 decode 为 echo！")
        else:
            print(f"    ✓ 正确（该样本是 batch 内最长，s=0）")
    print()

    print("【正确的截取方式：gen[i, ml:]】")
    for i, pl in enumerate(pls):
        segment = gen[i, ml:].tolist()
        assert segment == new_token_ids[i], f"sample {i}: expected {new_token_ids[i]}, got {segment}"
        print(f"  sample {i}: gen[i,{ml}:]={segment} ✓ (仅新生成 token)")

    print()
    print("结论：应将 gen[i, int(pl):] 改为 gen[i, ml:]，")
    print("      其中 ml = gi['input_ids'].shape[1]")
    print()


# =========================================================================
# Part 2: 用实际 tokenizer 验证文本 token 的 decode 效果
# =========================================================================

def test_echo_decode_with_tokenizer(tokenizer_path=None):
    """验证含文本 token 的前缀 echo 在 decode 后确实产生重复文本。"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers 未安装，跳过 tokenizer 测试")
        return

    if tokenizer_path is None:
        # 尝试从环境变量或 HF Hub 默认路径加载
        candidates = [
            os.environ.get("SO_BASE_MODEL", ""),
            "Qwen/Qwen2.5-Omni-7B",
            "Qwen/Qwen2.5-Omni-3B",
        ]
        for c in candidates:
            if os.path.exists(c):
                tokenizer_path = c
                break
    if tokenizer_path is None or not os.path.exists(tokenizer_path):
        print("未找到 tokenizer，跳过 Part 2")
        return

    print("=" * 60)
    print("测试（Part 2）：实际 tokenizer 验证 echo decode")
    print("=" * 60)

    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"加载 tokenizer 失败: {e}")
        return

    prompt_text = "Is there a sound source positioned at the front?\n"
    answer_text = "yes"
    eos_token = tok.eos_token or ""
    full_text_prefix = "<|AUDIO|>" + "<|spatial|>" + f"\n{prompt_text}\n"
    ans_sfx = answer_text + eos_token

    # 获取 spatial token id
    vocab = tok.get_vocab()
    spatial_tok = "<|spatial|>"
    if spatial_tok not in vocab:
        tok.add_special_tokens({"additional_special_tokens": [spatial_tok]})
    spatial_id = tok.convert_tokens_to_ids(spatial_tok)
    audio_id = tok.convert_tokens_to_ids("<|AUDIO|>")

    # Tokenize prefix
    full_seq = full_text_prefix
    prefix_ids = tok.encode(full_seq, add_special_tokens=False)
    print(f"prefix length: {len(prefix_ids)} tokens")
    print(f"  (expected: ~550 + text tokens for AUDIO+spatial+text)")
    print(f"  first 5 token ids: {prefix_ids[:5]}")
    print(f"  last 10 token ids: {prefix_ids[-10:]}")

    # 模拟左填充 batch（该样本 + 一个更长的样本）
    longer_extra = 80  # 更长样本比此样本多 80 token
    ml = len(prefix_ids) + longer_extra

    # 模拟 generate 输出（echo prefix 的后半段 + 真实答案 token）
    answer_ids = tok.encode(answer_text, add_special_tokens=False)
    pl_i = len(prefix_ids)
    s_i = ml - pl_i  # = longer_extra = 80

    # gen[i] 布局：[pad * s_i | prefix_ids | answer_ids | ...]
    gen_i = (
        [tok.pad_token_id or 0] * s_i
        + prefix_ids
        + answer_ids
        + [tok.pad_token_id or 0] * (48 - len(answer_ids))  # 填充到 max_new_tokens
    )
    gen_tensor = torch.tensor(gen_i, dtype=torch.long)

    # 模拟错误的截取
    wrong_decode = tok.decode(gen_tensor[pl_i:], skip_special_tokens=True).strip()
    # 模拟正确的截取
    correct_decode = tok.decode(gen_tensor[ml:], skip_special_tokens=True).strip()

    print()
    print(f"pl_i = {pl_i}, ml = {ml}, s_i = s_i = {s_i}")
    print(f"  错误截取 gen[i, {pl_i}:]  → decode = '{wrong_decode[:80]}...' (可能含 echo)")
    print(f"  正确截取 gen[i, {ml}:]   → decode = '{correct_decode}' (仅答案)")
    print()

    # 关键断言
    if prompt_text.strip()[:10] in wrong_decode:
        print("  ❌ 确认：错误截取导致 prompt 文本出现在 decode 输出中！")
    else:
        print("  (错误截取 decode 未包含明确 prompt 文本，可能 special token 过滤了大部分)")
    if correct_decode.strip() == answer_text:
        print("  ✓ 正确截取仅含答案 token")
    print()


# =========================================================================
# Part 3: 验证 build_left_padded_batch + mask 逻辑
# =========================================================================

def test_build_left_padded_batch():
    """验证 build_left_padded_batch 正确性（mask 与 input_ids 对齐）。"""
    print("=" * 60)
    print("测试（Part 3）：build_left_padded_batch 与 attention_mask 正确性")
    print("=" * 60)

    # 构造右填充的 input_ids（模拟 training batch）
    pad_id = 0
    # 3 个样本，不同长度
    seqs = [
        list(range(1, 11)),     # 长度 10
        list(range(1, 8)),      # 长度 7
        list(range(1, 16)),     # 长度 15
    ]
    max_len = max(len(s) for s in seqs)
    B = len(seqs)

    # 右填充
    input_ids_right = torch.zeros(B, max_len, dtype=torch.long)
    attn_right = torch.zeros(B, max_len, dtype=torch.long)
    pl_list = []
    for i, s in enumerate(seqs):
        input_ids_right[i, :len(s)] = torch.tensor(s)
        attn_right[i, :len(s)] = 1
        pl_list.append(len(s))
    pl = torch.tensor(pl_list, dtype=torch.long)

    # 模拟 build_left_padded_batch
    ml = int(pl.max()); B2 = input_ids_right.shape[0]
    gi = torch.full((B2, ml), fill_value=pad_id, dtype=input_ids_right.dtype)
    gm = torch.zeros((B2, ml), dtype=attn_right.dtype)
    for i, p in enumerate(pl.tolist()):
        s = ml - p
        gi[i, s:] = input_ids_right[i, :p]
        gm[i, s:] = 1

    # 验证 1：gm.sum(1) == pl
    assert (gm.sum(1) == pl).all(), "attention_mask sum 与 prefix_lengths 不一致！"
    print("  ✓ gm.sum(1) == pl（attention_mask 正确反映 prefix 长度）")

    # 验证 2：gi 中非 padding 区域 == 原始前缀 tokens
    for i, p in enumerate(pl.tolist()):
        s = ml - p
        expected = input_ids_right[i, :p]
        got = gi[i, s:]
        assert (expected == got).all(), f"sample {i}: 左填充 token 不对"
    print("  ✓ 左填充 token 内容正确（gi[i, s:] == original prefix）")

    # 验证 3：新生成 token 起点是 ml，不是 pl_i
    print(f"  ml = {ml}, pl_list = {pl_list}")
    print(f"  对 batch 中非最长样本（pl < ml），新 token 起点 = ml，而非 pl")
    for i, p in enumerate(pl.tolist()):
        if p < ml:
            print(f"    sample {i}: pl={p}, s={ml-p}, 正确截取起点={ml}（差 {ml-p} 个 prefix token）")
    print()

    # 关键结论
    print("  结论：decode 应用 gen[i, ml:] 而非 gen[i, pl_i:]")
    print()


# =========================================================================
# Part 4: 检查 spatial placeholder 的正确性
# =========================================================================

def test_spatial_placeholder_alignment():
    """验证 spatial placeholder 在左填充 batch 中的正确对齐。

    spatial placeholder 是前缀的一部分，在左填充后位于 [s_i, s_i+500+50] 位置区间内。
    forward() 里用 input_ids == spatial_token_id 来定位 placeholder，与填充无关。
    这部分验证在左填充 input_ids 中 spatial token 仍然存在且位置正确。
    """
    print("=" * 60)
    print("测试（Part 4）：spatial placeholder 在左填充 input_ids 中的对齐")
    print("=" * 60)

    AUDIO_ID = 151646
    SPATIAL_ID = 151665
    PAD_ID = 0
    N_AUDIO = 500
    N_SPATIAL_SHORT = 40   # 短样本（8s 音频）
    N_SPATIAL_LONG = 50    # 长样本（20s 音频）
    TEXT_SHORT = 20        # 短文本 token 数
    TEXT_LONG = 70         # 长文本 token 数

    # 构造两个样本的前缀
    prefix_short = [AUDIO_ID] * N_AUDIO + [SPATIAL_ID] * N_SPATIAL_SHORT + list(range(2000, 2000 + TEXT_SHORT))
    prefix_long  = [AUDIO_ID] * N_AUDIO + [SPATIAL_ID] * N_SPATIAL_LONG  + list(range(3000, 3000 + TEXT_LONG))
    pl_short = len(prefix_short)   # 500 + 40 + 20 = 560
    pl_long  = len(prefix_long)    # 500 + 50 + 70 = 620
    ml = max(pl_short, pl_long)

    # 右填充训练 input_ids
    B = 2
    input_ids = torch.zeros(B, ml, dtype=torch.long)
    input_ids[0, :pl_short] = torch.tensor(prefix_short)
    input_ids[1, :pl_long]  = torch.tensor(prefix_long)
    attn = (input_ids != PAD_ID).long()
    pl = torch.tensor([pl_short, pl_long], dtype=torch.long)

    # 构造左填充 gen input_ids
    gi = torch.full((B, ml), fill_value=PAD_ID, dtype=torch.long)
    for i, p in enumerate(pl.tolist()):
        s = ml - p
        gi[i, s:] = input_ids[i, :p]

    # 验证 spatial token 仍在 gi 中
    for i in range(B):
        spatial_positions = (gi[i] == SPATIAL_ID).nonzero(as_tuple=True)[0].tolist()
        expected_n = [N_SPATIAL_SHORT, N_SPATIAL_LONG][i]
        print(f"  sample {i}: {len(spatial_positions)} spatial tokens (预期 {expected_n})", end="")
        if len(spatial_positions) == expected_n:
            print(" ✓")
        else:
            print(" ❌ 数量不对！")
        if spatial_positions:
            print(f"    位置范围: [{spatial_positions[0]}, {spatial_positions[-1]}]"
                  f"（padding 占据 [0, {ml - [pl_short, pl_long][i] - 1}]）")

    print()
    print("  spatial placeholder 在左填充 input_ids 中仍然存在，")
    print("  forward() 的 masked_scatter 可以正确找到并替换它们。")
    print("  → spatial 注入逻辑本身没有问题。")
    print()

    # 重要：gen[i, pl_i:] 截取时会截到哪些 spatial token？
    print("  echo 分析：gen[i, pl_short:]（短样本，错误截取）包含：")
    wrong_start = pl_short
    content_in_wrong = gi[0, wrong_start:].tolist()
    audio_cnt  = sum(1 for t in content_in_wrong if t == AUDIO_ID)
    spatial_cnt = sum(1 for t in content_in_wrong if t == SPATIAL_ID)
    text_cnt   = sum(1 for t in content_in_wrong if t not in (AUDIO_ID, SPATIAL_ID, PAD_ID))
    pad_cnt    = sum(1 for t in content_in_wrong if t == PAD_ID)
    print(f"    padding={pad_cnt}, AUDIO={audio_cnt}, spatial={spatial_cnt}, text={text_cnt}")
    print(f"    → 其中 text={text_cnt} 个 token 在 skip_special_tokens=True 时不被过滤，")
    print(f"      被 decode 出来就是问题文本的 echo！")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("诊断测试：run_validation_generation 中左填充 decode 偏移 bug")
    print("=" * 70 + "\n")

    test_left_padded_decode_offset()
    test_build_left_padded_batch()
    test_spatial_placeholder_alignment()
    test_echo_decode_with_tokenizer()

    print("=" * 70)
    print("【修复方案】")
    print("=" * 70)
    print("""
在 train_so_qa.py 的 run_validation_generation() 中：

  OLD（有 bug）:
    pls = gi["attention_mask"].sum(1).tolist(); gen = gen.detach().cpu()
    for i, pl in enumerate(pls):
        pt = processor.tokenizer.decode(
            gen[i, int(pl):], skip_special_tokens=True).strip()

  FIXED:
    ml = gi["input_ids"].shape[1]   # 左填充后 batch 的统一序列长度
    gen = gen.detach().cpu()
    for i in range(len(batch["meta"])):
        pt = processor.tokenizer.decode(
            gen[i, ml:], skip_special_tokens=True).strip()

原因：左填充 batch 中 generate() 输出的新 token 从位置 ml 开始（对所有样本统一），
      而非各自的 pl_i（pl_i 仅是该样本的实际 prefix 长度）。
      当 pl_i < ml 时，gen[i, pl_i:ml] = 前缀末尾的文本 token，被错误 decode 为 echo。
""")
