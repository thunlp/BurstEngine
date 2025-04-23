import sys
sys.path.append("../baselines/InternEvo/internlm/model/ops/ring_flash_attn/zigzag_ring_flash_attn_with_sliding_window.py")
from 
def zigzag_ring_flash_attn_qkvsplited_func_with_sliding_window(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    context_group=None,
    inter_window_group=None,
    intra_window_group=None,
    dkv_inter_window_group=None,
    dkv_intra_window_group=None,
    layer_idx=0,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        context_group,
        inter_window_group,
        intra_window_group,
        dkv_inter_window_group,
        dkv_intra_window_group,
        layer_idx,
    )
