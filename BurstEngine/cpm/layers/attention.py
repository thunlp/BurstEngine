# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional
from typing import Tuple

from .flash_triton import FlashAttnFunc,flash_cuda_forward
from burst_attn import burst_attn_func, burst_attn_func_striped
from .ring_attn import ring_attn
from bmtrain import nccl
import inspect

import bmtrain as bmt
import torch
from ring_flash_attn import zigzag_ring_flash_attn_func
from cpm.arguments import get_args
from .linear import Linear, ColumnParallelLinear
from .position_embedding import apply_chatglm_rotary_pos_emb
from torch import Tensor
from functools import lru_cache
_mask = None
def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = nccl.commCount(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    # output = torch.empty_like(input_t)
    output  = bmt.distributed.all_to_all(input_t, comm=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()

class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx, *grad_output: Tensor):
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)

class Attention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.half,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        add_qkv_bias: bool = False,
        use_flash_attn: bool = False,
        tp: int = 0,
        sp: str = 'none',
        tp_sp: bool = False,
        layer_idx:int=-1,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads
        self.dim_head = dim_head

        self.project_q = Linear(
            self.dim_model, self.num_heads * self.dim_head, bias=add_qkv_bias, dtype=dtype, scale=scale, tp = tp,
        )
        self.project_k = Linear(
            self.dim_model, self.num_kv_heads * self.dim_head, bias=add_qkv_bias, dtype=dtype, scale=scale, tp = tp,
        )
        self.project_v = Linear(
            self.dim_model, self.num_kv_heads * self.dim_head, bias=add_qkv_bias, dtype=dtype, scale=scale, tp = tp,
        )
        if tp:
            self.attention_out = Linear(
                self.num_heads * self.dim_head, self.dim_model, dtype=dtype, scale=scale, tp = tp * 2, all_reduce_output = not tp_sp
            )
        else:
            self.attention_out = Linear(
                self.num_heads * self.dim_head, self.dim_model, dtype=dtype, scale=scale, tp = tp * 2
            )

        self.softmax = torch.nn.Softmax(dim=-1)

        from flash_attn.flash_attn_interface import (
            _flash_attn_backward as _flash_attn_backward_cuda,
        )
        self.bwd_opt = not "bwd_opt" in get_args().ablation

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
            self.dropout_p = dropout_p
        else:
            self.dropout = None

        self.use_flash_attn = use_flash_attn
        self.tp = tp
        self.sp = sp
        self.tp_sp = tp_sp
        self.layer_idx = layer_idx
        self._layer_dict = {}

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[torch.Tensor] = None,
        attention_mask_bias: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """This model inherits from bmt.DistributedModule.
        Args:
            hidden_q (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.size(0)

        if self.tp:
            assert hidden_q.data_ptr() == hidden_kv.data_ptr()
            if self.project_q.scale and self.project_q.scale_before:
                hidden_q = hidden_q / math.sqrt(self.project_q.dim_in)
            tp_sp = self.tp_sp
            hidden_q = bmt.nn.OpParallelLinear.apply(
                hidden_q,
                torch.cat([self.project_q.weight, self.project_k.weight, self.project_v.weight], dim=0),
                torch.cat([self.project_q.bias, self.project_k.bias, self.project_v.bias], dim=0) if self.project_q.bias is not None else None,
                tp_sp, False,
                False, None,
                1
            )
            if self.project_q.scale and not self.project_q.scale_before:
                hidden_q = hidden_q / math.sqrt(self.project_q.dim_in)
            hidden_q = hidden_q.view(batch_size, -1, hidden_q.shape[-1])

            block_size = hidden_q.shape[-1]//(self.head_groups+1+1)
            h_q = hidden_q[..., :block_size*self.head_groups]
            h_k = hidden_q[..., block_size*self.head_groups:block_size*(self.head_groups+1)]
            h_v = hidden_q[..., block_size*(self.head_groups+1):]
        else:
            h_q = self.project_q(hidden_q)
            h_k = self.project_k(hidden_kv)
            h_v = self.project_v(hidden_kv)
        
        len_q = h_q.size(1)
        len_k = h_k.size(1)

        assert pos_bias_type == "rotary"
        if self.sp == 'none':
            if not self.use_flash_attn:
                h_q = h_q / math.sqrt(math.sqrt(self.dim_head))
                h_k = h_k / math.sqrt(math.sqrt(self.dim_head))

                h_q = h_q.view(batch_size, len_q, -1, self.dim_head).permute(0, 2, 1, 3)
                h_k = h_k.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3)
                h_v = h_v.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3)

                h_q, h_k = position_bias(h_q, h_k, -2, offset=past_kv[0].size(-2) if past_kv is not None else 0)

                if past_kv is not None:
                    h_k = torch.cat([past_kv[0], h_k], dim=-2)
                    h_v = torch.cat([past_kv[1], h_v], dim=-2)
                    len_k = h_k.size(-2)

                # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
                # (b, n_kv_h, n_h_groups*len_q, d_h) @ (b, n_kv_h, d_h, len_k) -> (b, n_kv_h, n_h_groups*len_q, len_k) -> (b, n_h, len_q, len_k)
                if self.head_groups == 1:
                    score = torch.matmul(h_q, h_k.transpose(-1, -2))  # / math.sqrt(self.dim_head) moved to line 75~76
                else:
                    score = torch.matmul(
                        h_q.reshape(batch_size, -1, self.head_groups * len_q, self.dim_head),
                        h_k.transpose(-1, -2),
                    ).view(
                        batch_size, -1, len_q, len_k
                    )  # / math.sqrt(self.dim_head) moved to line 75~76
                if attention_mask is not None:
                    score = torch.masked_fill(
                        score,
                        attention_mask.view(batch_size, 1, len_q, len_k) == False,
                        torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
                    )

                score = self.softmax(score)

                if attention_mask is not None:
                    score = torch.masked_fill(
                        score,
                        attention_mask.view(batch_size, 1, len_q, len_k) == False,
                        torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
                    )

                if self.dropout is not None:
                    score = self.dropout(score)

                # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
                # (b, n_kv_h, n_h_groups*len_q, len_k) @ (b, n_kv_h, len_k, d_h) -> (b, n_kv_h, n_h_groups*len_q, d_h) -> (b, n_h, len_q, d_h)
                score = torch.matmul(
                    score.view(batch_size, -1, self.head_groups * len_q, len_k),
                    h_v
                ).view(batch_size, -1, len_q, self.dim_head)

                score = score.view(batch_size, -1, len_q, self.dim_head).permute(0, 2, 1, 3)
                score = score.contiguous().view(batch_size, len_q, -1)

            else:
                assert pos_bias_type == "rotary"
                h_q = h_q.view(batch_size, len_q, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_k = h_k.view(batch_size, len_k, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_v = h_v.view(batch_size, len_k, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_q, h_k = position_bias(h_q, h_k, -3)
                h_k = h_k.repeat(1, 1, self.head_groups, 1).contiguous()
                h_v = h_v.repeat(1, 1, self.head_groups, 1).contiguous()
                
                if attention_mask_bias is not None:
                    attention_mask_bias = attention_mask_bias.contiguous()
                if get_args().flash == "cuda":
                    score = flash_cuda_forward(h_q, h_k, h_v, True, None)
                    score = score.view(batch_size, len_q, -1)
                elif get_args().flash == "triton":
                    score = FlashAttnFunc.apply(h_q, h_k, h_v, None, True, None)
                    score = score.view(batch_size, len_q, -1)
        else:
            if self.sp.startswith('burst'):
                sl_ckpt = "sl_ckpt" in self.sp
                whole_ckpt = "whole_ckpt" in self.sp
                if whole_ckpt:
                    ckpt_method = "full"
                elif sl_ckpt:
                    ckpt_method = "seq-wise"
                else:
                    ckpt_method = "none"
                is_sparse = "sparse" in self.sp
                if is_sparse:
                    from burst_attn.sparse_utils import create_block_mask
                    @lru_cache(maxsize=2)
                    def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
                        block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
                        return block_mask
                    global _mask
                    if _mask is None:
                        from burst_attn.sparse_utils import create_block_mask
                        def half_mod(b, h, q_idx, kv_idx):
                            windowed_mask = (
                                q_idx - kv_idx %2 == 0
                            )  # We dont need to check the right side of the sliding window since we are applying the causal mask
                            # windowed_mask = torch.randint_like(q_idx<kv_idx, 0, 2)
                            # windowed_mask = torch.randint(0, 2, (1,))
                            return windowed_mask
                        _mask = create_block_mask_cached(half_mod, batch_size, self.num_heads, len_q, len_k)

                    
                _burst_func = burst_attn_func if not sl_ckpt else burst_attn_func_striped
                if self.use_flash_attn:
                    h_q = h_q.view(batch_size, len_q, -1, self.dim_head)
                    h_k = h_k.view(batch_size, len_k, -1, self.dim_head)
                    h_v = h_v.view(batch_size, len_k, -1, self.dim_head)
                    h_q, h_k = position_bias(h_q, h_k, -3, offset=bmt.config['tp_rank']*h_q.shape[-3], max_length=bmt.config['tp_size']*h_q.shape[-3])
                    h_k = h_k.repeat(1, 1, self.head_groups, 1).contiguous()
                    h_v = h_v.repeat(1, 1, self.head_groups, 1).contiguous()
                    h_q = h_q.contiguous()
                    if bmt.config['tp_size'] <= bmt.config['local_size'] or "double" in get_args().ablation:
                        double_comm = (None, None)
                        intra_comm = None
                        inter_comm = None
                        assert "ulysses" not in self.sp, "hybrid does not support tp_size < local_world_size"
                    else:
                        if bmt.config['tp_size'] < bmt.config['world_size']:
                            double_comm = (bmt.config['local_comm'], bmt.config['tp_local_idx_comm'])
                            intra_comm = bmt.config['local_comm']
                            inter_comm = bmt.config['tp_local_idx_comm']
                        else:
                            double_comm = (bmt.config['local_comm'], bmt.config['local_idx_comm'])
                            intra_comm = bmt.config['local_comm']
                            inter_comm = bmt.config['local_idx_comm']
                    causal = not "causal" in get_args().ablation
                    if "ulysses" in self.sp:
                        h_q = _SeqAllToAll.apply(intra_comm, h_q, 2, 1)
                        h_k = _SeqAllToAll.apply(intra_comm, h_k, 2, 1)
                        h_v = _SeqAllToAll.apply(intra_comm, h_v, 2, 1)
                        if is_sparse:
                            from burst_attn.burst_attn_interface  import burst_attn_func_sparse
                            score = burst_attn_func_sparse(
                                h_q,
                                h_k,
                                h_v,
                                _mask,
                                1.0 / math.sqrt(self.dim_head),
                                True,
                                inter_comm,
                                [None, None],
                                False,
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx
                            )
                        elif "sliding" in self.sp:
                            score = burst_attn_func_striped(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                causal,
                                self.bwd_opt,
                                False,
                                inter_comm,
                                [None, None],
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx,
                                sliding_window = (4096, 0 if causal else 4096)
                            )
                        elif "doc_mask" in self.sp:
                            cu_seqlen = torch.arange(0, h_q.shape[1]+1, 4096, dtype=torch.float32, device="cuda")
                            score = burst_attn_func_striped(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                causal,
                                self.bwd_opt,
                                False,
                                inter_comm,
                                [None, None],
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx,
                                cu_seqlens = cu_seqlen
                            )
                        else:
                            score = _burst_func(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                causal,
                                self.bwd_opt,
                                False,
                                inter_comm,
                                [None, None],
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx
                            )
                        score = _SeqAllToAll.apply(intra_comm, score, 1, 2)
                    else:
                        if "sparse" in self.sp:
                            from burst_attn.burst_attn_interface import burst_attn_func_sparse
                            score = burst_attn_func_sparse(
                                h_q,
                                h_k,
                                h_v,
                                _mask,
                                1.0 / math.sqrt(self.dim_head),
                                True,
                                bmt.config["tp_comm"],
                                double_comm,
                                False,
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx
                            )
                        elif "full" in self.sp:
                            score = burst_attn_func(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                False,
                                self.bwd_opt,
                                False,
                                bmt.config["tp_comm"],
                                double_comm,
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx,
                            )
                        elif "sliding" in self.sp:
                            score = burst_attn_func_striped(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                causal,
                                self.bwd_opt,
                                False,
                                bmt.config["tp_comm"],
                                double_comm,
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx,
                                sliding_window = (4096, 0 if causal else 4096)
                            )
                        elif "doc_mask" in self.sp:
                            cu_seqlen = torch.arange(0, h_q.shape[1]+1, 4096, dtype=torch.kloat32, device="cuda")
                            score = burst_attn_func_striped(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                causal,
                                self.bwd_opt,
                                False,
                                bmt.config["tp_comm"],
                                double_comm,
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx,
                                cu_seqlen = cu_seqlen
                            )
                        else:
                            score = _burst_func(
                                h_q,
                                h_k,
                                h_v,
                                1.0 / math.sqrt(self.dim_head),
                                get_args().flash,
                                causal,
                                self.bwd_opt,
                                False,
                                bmt.config["tp_comm"],
                                double_comm,
                                grad_ckpt=ckpt_method,
                                layer_idx=self.layer_idx
                            )
                else:
                    h_q = h_q.view(batch_size, len_q, -1, self.dim_head).permute(0, 2, 1, 3).contiguous()
                    h_k = h_k.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3).contiguous()
                    h_v = h_v.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3).contiguous()
                    h_q, h_k = position_bias(h_q, h_k, -2, offset=bmt.config['tp_rank']*h_q.shape[-2], max_length=bmt.config['tp_size']*h_q.shape[-2])
                    h_k = h_k.repeat(1, self.head_groups, 1, 1).contiguous()
                    h_v = h_v.repeat(1, self.head_groups, 1, 1).contiguous()
                    h_q = h_q.contiguous()
                    if attention_mask is None:
                        mask = None
                    else:
                        mask = attention_mask.unsqueeze(dim=2)
                    double_comm = (bmt.config['local_comm'], bmt.config['local_idx_comm'])
                    score = burst_attn_func(h_q,h_k,h_v, 1.0/math.sqrt(self.dim_head), get_args().flash, True, True, False, bmt.config['tp_comm'], double_comm)
            elif self.sp == 'ulysses':
                h_q = h_q.view(batch_size, len_q, -1, self.dim_head)
                h_k = h_k.view(batch_size, len_k, -1, self.dim_head)
                h_v = h_v.view(batch_size, len_k, -1, self.dim_head)
                h_q = _SeqAllToAll.apply(bmt.config['tp_comm'], h_q, 2, 1)
                h_k = _SeqAllToAll.apply(bmt.config['tp_comm'], h_k, 2, 1)
                h_v = _SeqAllToAll.apply(bmt.config['tp_comm'], h_v, 2, 1)
                score = flash_cuda_forward(h_q, h_k, h_v, True, None)
                score = _SeqAllToAll.apply(bmt.config['tp_comm'], score, 1, 2)
            elif self.sp == 'ring':

                if self.use_flash_attn:
                    h_q = h_q.view(batch_size, len_q, -1, self.dim_head)
                    h_k = h_k.view(batch_size, len_k, -1, self.dim_head)
                    h_v = h_v.view(batch_size, len_k, -1, self.dim_head)
                    h_q, h_k = position_bias(h_q, h_k, -3, offset=bmt.config['tp_rank']*h_q.shape[-3], max_length=bmt.config['tp_size']*h_q.shape[-3])
                    h_k = h_k.repeat(1, 1, self.head_groups, 1).contiguous()
                    h_v = h_v.repeat(1, 1, self.head_groups, 1).contiguous()
                    h_q = h_q.contiguous()
                    double_comm = (bmt.config['local_comm'], bmt.config['local_idx_comm'])
                    score = zigzag_ring_flash_attn_func(h_q,h_k,h_v,0.0, 1.0/math.sqrt(self.dim_head),True,(-1,-1),None,False, False, double_comm)
                else:
                    h_q = h_q.view(batch_size, len_q, -1, self.dim_head).permute(0, 2, 1, 3).contiguous()
                    h_k = h_k.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3).contiguous()
                    h_v = h_v.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3).contiguous()
                    h_q, h_k = position_bias(h_q, h_k, -2, offset=bmt.config['tp_rank']*h_q.shape[-2], max_length=bmt.config['tp_size']*h_q.shape[-2])
                    h_k = h_k.repeat(1, self.head_groups, 1, 1).contiguous()
                    h_v = h_v.repeat(1, self.head_groups, 1, 1).contiguous()
                    score = ring_attn(h_q,h_k,h_v,1.0/math.sqrt(self.dim_head),attention_mask_bias)

            score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, len_q, -1)

        score = self.attention_out(score)

        if use_cache:
            return score, (h_k, h_v)
        else:
            return score
