#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from torch import nn

from internlm.core.context import global_context as gpc
from internlm.model.ops.cross_entropy import new_cross_entropy
from internlm.utils.logger import get_logger


logger = get_logger(__file__)


class FlashGPTLMLoss(nn.Module):
    """
    Loss function for flash GPT Language Model.
    """

    def __init__(self, parallel_output=True, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None:
            if label_smoothing != 0:
                if gpc.is_rank_for_log():
                    print(f"use label_smoothing: {label_smoothing}")
        else:
            label_smoothing = 0

        self.label_smoothing = label_smoothing
        self.loss_fn = new_cross_entropy(
            reduction="mean",
            label_smoothing=self.label_smoothing,
            parallel_output=parallel_output,
            inplace_backward=True,
        )

    def forward(self, *args):
        print("*************************88")
        print(len(args), gpc.config.cce_loss)
        if len(args) == 3:
            # residual is to match prenorm
            if gpc.config.cce_loss:
                logits, lm_head, labels = args
                # tp_size = gpc.config.parallel.tensor.size
                # tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
                # label = label.chunk(tp_size)[tp_rank]
                from cut_cross_entropy import linear_cross_entropy
                ret = linear_cross_entropy(
                    logits,
                    lm_head,
                    labels[0])

                return ret
            else:
                logits, _, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(
            shift_logits, shift_labels
        )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
        # calculated through the calculation range, and -100 must be outside this range, so there is no problem

        return loss
