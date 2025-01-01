def num_floating_point_operations(
    seq_length,
    num_layers,
    hidden_size,
    ffn_hidden_size,
    vocab_size,
    num_attention_heads,
    kv_channels=None,
    num_query_groups=None,
    causal_tflops=False,
    swiglu=True,
    num_experts=None,
    moe_router_topk=1,
    group_query_attention=False,
    attn_only=False,
):
    # Attention projection size.
    kv_channels = (
        hidden_size // num_attention_heads if kv_channels is None else kv_channels
    )
    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
    # Group Query Attention.
    if not group_query_attention:
        num_query_groups = num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if num_experts is None else moe_router_topk
    gated_linear_multiplier = 3 / 2 if swiglu else 1

    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2
    if attn_only:
        return (
            expansion_factor
            * seq_length
            * num_layers
            * hidden_size
            * hidden_size
            * (+(seq_length / hidden_size) * (0.5 if causal_tflops else 1))
            * query_projection_to_hidden_size_ratio
        )
    return (
        expansion_factor
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            # Attention.
            (
                (
                    1
                    + (num_query_groups / num_attention_heads)
                    + (seq_length / hidden_size) * (0.5 if causal_tflops else 1)
                )
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (ffn_hidden_size / hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Logit.
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )
