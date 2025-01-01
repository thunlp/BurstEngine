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

from typing import Optional
import math
import bmtrain as bmt
import torch

from .linear import Linear, LastLinear


class DenseGatedACT(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        activate_fn: str = "gelu",
        scale: bool = True,
        dtype=torch.half,
        tp: int = 0,
        tp_sp: bool = False,
    ):
        super().__init__()
        self.tp_sp = tp_sp
        self.scale = scale
        if tp:
            self.w_0 = Linear(
                dim_in=dim_in,
                dim_out=dim_ff,
                dtype=dtype,
                scale=scale,
                scale_before=False,
                tp=tp,
                gather_input=tp_sp
            )
        else:
            self.w_0 = Linear(
                dim_in=dim_in,
                dim_out=dim_ff,
                dtype=dtype,
                scale=scale,
                scale_before=False,
                tp=tp,
            )
        if activate_fn == "silu":
            if tp:
                self.w_1 = Linear(
                    dim_in=dim_in,
                    dim_out=dim_ff,
                    dtype=dtype,
                    scale=scale,
                    scale_before=False,
                    tp=tp,
                    gather_input=tp_sp
                )
            else:
                self.w_1 = Linear(
                    dim_in=dim_in,
                    dim_out=dim_ff,
                    dtype=dtype,
                    scale=scale,
                    scale_before=False,
                    tp=tp,
                )

        self._gated = activate_fn == "silu"
        if activate_fn == "gelu":
            self.act = torch.nn.GELU()
        elif activate_fn == "silu":
            self.act = torch.nn.functional.silu
        else:
            raise NotImplementedError(f"{activate_fn} is not supported")

    def forward(self, x: torch.Tensor):
        """This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        if self._gated:
            if self.tp_sp:
                x = bmt.nn.OpParallelLinear.apply(
                    x,
                    torch.cat([self.w_0.weight, self.w_1.weight], dim=0),
                    torch.cat([self.w_0.bias, self.w_1.bias], dim=0) if self.w_0.bias is not None else None,
                    self.tp_sp, False,
                    False, None,
                    1
                )
            else:
                x = bmt.nn.OpLinear.apply(x,
                    torch.cat([self.w_0.weight, self.w_1.weight], dim=0),
                    torch.cat([self.w_0.bias, self.w_1.bias], dim=0) if self.w_0.bias is not None else None,
                )
            if self.scale and not self.scale_before:
                x = x / math.sqrt(self.dim_in)
            x_0, x_1 = torch.split(x, x.shape[-1] // 2, dim=-1)
            gate_score = self.act(x_0)

            x = gate_score * x_1
            return x
        else:
            if self.tp_sp:
                x = bmt.nn.OpParallelLinear.apply(
                    x,
                    self.w_0.weight,
                    self.w_0.bias,
                    self.tp_sp, False,
                    False, None,
                    1
                )
            else:
                x = bmt.nn.OpLinear.apply(x, self.w_0.weight, self.w_0.bias)
            if self.scale and not self.scale_before:
                x = x / math.sqrt(self.dim_in)
            x = self.act(x)
            return x


class FeedForward(bmt.DistributedModule):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str = "gelu",
        dtype=torch.half,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        tp: int = 0,
        tp_sp: bool = False,
    ):
        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            scale=scale,
            tp=tp,
            tp_sp=tp_sp
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        if tp:
            self.w_out = Linear(
                dim_in=dim_ff,
                dim_out=dim_model,
                dtype=dtype,
                scale=scale,
                scale_before=False,
                tp=tp*2,
                all_reduce_output=not tp_sp
            )
        else:
            self.w_out = Linear(
                dim_in=dim_ff,
                dim_out=dim_model,
                dtype=dtype,
                scale=scale,
                scale_before=False,
                tp=tp*2,
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        x = self.w_in(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)
        return x
