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

from typing import List
from typing import Optional
from typing import Tuple

import bmtrain as bmt
import torch
import torch.nn.functional as F
from typing_extensions import TypedDict

from ...arguments import get_args
from ...layers import Embedding
from ...layers import Encoder
from ...layers import RotaryEmbeddingESM
from ...utils import Config
from ...utils import gradient_shrink


class LlamaInferenceState(TypedDict):
    buffer_context: torch.Tensor
    buffer_sample_ids: torch.Tensor
    buffer: List[Tuple[torch.Tensor, torch.Tensor]]


class LlamaConfig(Config):
    def __init__(
        self,
        vocab_size=32000,
        dim_model=4096,
        num_heads=32,
        num_kv_heads=32,
        dim_head=128,
        dim_ff=11008,
        num_layers=32,
        dropout_p=0.0,
        activate_fn="silu",
        scale=True,
        eps=1e-5,
        half: bool = True,
        bf16: bool = False,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        use_flash_attn: bool = True,
        flash_attn_mask_shape="1d",
        flash_impl="cuda",
        base=10000,
        tp=0,
        sp='none',
        disabled_checkpoint=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.activate_fn = activate_fn
        self.scale = scale
        self.eps = eps
        if half:
            if bf16:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.flash_impl = flash_impl
        self.mask_modules = mask_modules
        self.use_flash_attn = use_flash_attn
        self.flash_attn_mask_shape = flash_attn_mask_shape
        self.base = base
        self.tp = tp
        self.sp = sp
        self.disabled_checkpoint = disabled_checkpoint


class Llama(bmt.DistributedModule):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dim_head=config.dim_head,
            activate_fn=config.activate_fn,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            scale=config.scale,
            mask_modules=config.mask_modules,
            use_flash_attn=config.use_flash_attn,
            tp=config.tp,
            tp_sp=config.tp_sp,
            sp=config.sp,
            disabled_checkpoint=config.disabled_checkpoint,
            ckpt=config.ckpt
        )

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            scale=config.scale,
            dtype=config.dtype,
            init_std=0.02,
            tp=config.tp,
            sp=config.sp!='none',
            tp_sp=config.tp_sp,
        )

        self.position_bias = RotaryEmbeddingESM(
            dim=config.dim_head, dtype=config.dtype, base=config.base, persistent=False, mixed_precision=True
        )

        self.lm_head = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            scale=config.scale,
            dtype=config.dtype,
            init_std=0.02,
            tp=config.tp,
            tp_sp=config.tp_sp,
            sp=config.sp!='none',
        )

        self.flash_impl = config.flash_impl
        self.use_flash_attn = config.use_flash_attn
        self.flash_attn_mask_shape = config.flash_attn_mask_shape
        self.config = config
        self.with_mask = config.casual_mask

    def forward(
        self,
        input: torch.Tensor,  # (batch, seqlen) int32
        length: torch.Tensor = None,  # (batch) int32
        context: torch.Tensor = None,  # (batch, seqlen) bool
        span: torch.Tensor = None,  # (batch, seqlen) int32
        cu_seqlens: torch.Tensor = None,  # (real_batch+2) int32
        max_seqlen: int = None,
        position_ids: torch.Tensor = None,  # (batch, seqlen) int32
    ):
        batch = context.size(0)
        seqlen = context.size(1)
        device = context.device

        hidden_states = self.input_embedding(input)

        if length is not None and length.dim() == 1:
            length = torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]

        # processing masks and position bias bucket
        if self.with_mask: 
            if not self.use_flash_attn or (self.flash_attn_mask_shape == "2d" and self.flash_impl == "triton"):
                if self.config.sp == 'burst':
                    sub_seqlen = hidden_states.shape[1]
                    with torch.no_grad():
                        mask = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)[bmt.rank()*sub_seqlen:(bmt.rank()+1)*sub_seqlen, :]
                        mask = length.view(batch, seqlen, 1)[:, bmt.rank()*sub_seqlen:(bmt.rank()+1)*sub_seqlen, :] & length.view(batch, 1, seqlen) & mask.unsqueeze(0)
                        attention_mask = torch.cat(mask.chunk(bmt.world_size(),dim=2), dim=1)
                        mask = torch.arange(seqlen, device=device)[bmt.rank()*sub_seqlen:(bmt.rank()+1)*sub_seqlen] <= torch.arange(seqlen, device=device).view(-1, 1)
                        mask = length.view(batch, seqlen, 1) & length.view(batch, 1, seqlen)[:, :, bmt.rank()*sub_seqlen:(bmt.rank()+1)*sub_seqlen] & mask.unsqueeze(0)
                        attention_mask = torch.stack([attention_mask, mask], dim=0)
                        mask = None
                else:
                    with torch.no_grad():
                        attention_mask = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
                        attention_mask = length.view(batch, seqlen, 1) & length.view(batch, 1, seqlen) & attention_mask.unsqueeze(0)
            else:
                attention_mask = None

            if self.config.sp == 'ring' or self.use_flash_attn:
                with torch.no_grad():
                    attention_mask = attention_mask.unsqueeze(dim=1)
                    attention_mask_bias = torch.zeros_like(attention_mask, device="cuda", dtype=torch.float16)
                    attention_mask_bias[attention_mask == False] = -10000
                attention_mask = None
            else:
                attention_mask_bias = None
        else:
            attention_mask = None
            attention_mask_bias = None
        if self.use_flash_attn:
            if self.flash_attn_mask_shape == "1d":
                hidden_states = self.encoder(
                    hidden_states,
                    attention_mask=None,
                    position_bias=self.position_bias,
                    pos_bias_type="rotary",
                    length_mask=length,
                )
            else:
                hidden_states = self.encoder(
                    hidden_states,
                    attention_mask=None,
                    position_bias=self.position_bias,
                    pos_bias_type="rotary",
                    length_mask=None,
                    attention_mask_bias=attention_mask_bias,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                )
        else:
            hidden_states = self.encoder(
                hidden_states, attention_mask=attention_mask, position_bias=self.position_bias, pos_bias_type="rotary", attention_mask_bias=attention_mask_bias
            )
        if get_args().fused_sl_loss:
            logits = hidden_states
        else:
            logits = self.lm_head.projection(hidden_states)

        return logits, hidden_states
