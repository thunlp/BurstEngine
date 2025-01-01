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

import inspect
import math

import bmtrain as bmt
import torch
import torch.nn.functional as F

def Linear(*args, **kwargs):
    tp = kwargs.pop('tp', 0)
    if tp == 0:
        return NormalLinear(*args, **kwargs)
    if tp == 1:
        return ColumnParallelLinear(*args, **kwargs)
    if tp == 2:
        return RowParallelLinear(*args, **kwargs)

class OpLastLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, record, x, weight, bias=None):
        ctx.self = self
        if not record and "r" in self._layer_dict:
            ctx.save_for_backward(x, weight, bias)
            self._layer_dict.pop("r")
            return torch.zeros((*x.shape[:-1], self.out_features), device=x.device, dtype=x.dtype)
        else:
            ctx.save_for_backward(x, weight, bias)
            if record:
                self._layer_dict["r"] = True
            return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if x.requires_grad:
            grad_x = grad_output.matmul(weight)
        if weight.requires_grad:
            grad_weight = grad_output.reshape(-1,
                grad_output.shape[-1]).t().matmul(x.reshape(-1, x.shape[-1]))
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return None, None, grad_x, grad_weight, grad_bias

class LastLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
        tp: int = 0,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before

        if not scale:
            init_std = 1 / ((dim_in+dim_out)**0.5)

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )
        self.bias = bmt.DistributedParameter(
            torch.empty(dim_out, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        ) if bias else None
        self._layer_dict = {}

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        x = OpLastLinear.apply(self, not torch.is_grad_enabled(), x, self.weight, self.bias)
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x

class NormalLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 0.02,
        scale: bool = True,
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before
        if not scale:
            init_std = 1 / ((dim_in+dim_out)**0.5)

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )
        self.bias = bmt.DistributedParameter(
            torch.empty(dim_out, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        ) if bias else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        if "tp_size" in inspect.signature(bmt.init_distributed).parameters:
            x = bmt.nn.OpLinear.apply(x, self.weight, self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x

class ColumnParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
        gather_output=False,
        gather_input=True,
    ):
        super().__init__()
        assert dim_out % bmt.config['tp_size'] == 0
        if not scale:
            init_std = 1 / ((dim_in+dim_out)**0.5)
        dim_out = dim_out // bmt.config['tp_size']
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before
        self.gather_input = gather_input
        self.gather_output = gather_output

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=0, tp_mode=True,
        )
        self.bias = bmt.DistributedParameter(
            torch.empty(dim_out, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=0, tp_mode=True,
        ) if bias else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, None,
            self.gather_input, self.gather_output,
            False, None,
            1
        )
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x

class RowParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = True,
        scale_before: bool = False,
        split_input=False,
        all_reduce_output=False,
    ):
        super().__init__()
        assert dim_in % bmt.config['tp_size'] == 0
        if not scale:
            init_std = 1 / ((dim_in+dim_out)**0.5)
        dim_in = dim_in // bmt.config['tp_size']
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale = scale
        self.scale_before = scale_before
        self.split_input = split_input
        self.all_reduce_output = all_reduce_output

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=1, tp_mode=True,
        )
        self.bias = bmt.DistributedParameter(
            torch.empty(dim_out, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=-1, tp_mode=True,
        ) if bias else None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if not self.all_reduce_output:
            x = x.view(x.shape[0] * bmt.config["tp_size"], -1, x.shape[-1])
        if self.scale and self.scale_before:
            x = x / math.sqrt(self.dim_in)
        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, None,
            self.split_input, False,
            self.split_input, 1 if self.all_reduce_output else 2,
            1
        )
        if self.bias is not None:
            x = x + self.bias
        if self.scale and not self.scale_before:
            x = x / math.sqrt(self.dim_in)
        return x
