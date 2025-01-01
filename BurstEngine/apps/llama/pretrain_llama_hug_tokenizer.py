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
import json
import math
import os
import re
import sys
import time
from einops import rearrange, repeat
from typing import Any
from typing import Dict
from typing import List
from typing import Union
import numpy as np

import bmtrain as bmt
from bmtrain import nccl
import torch

sys.path.insert(0, "../../")
from transformers import LlamaTokenizer

from cpm.arguments import get_args
from cpm.llama.models import Llama
from cpm.llama.models import LlamaConfig
from cpm.llama.training_tasks import MixedDataset
from cpm.utils import allgather_objects
from cpm.utils import logger
from cpm.utils import LogManager
from flops import get_tflops
def get_parameters_in_billions(config):
    total_params = 0
    total_params += config.vocab_size * config.dim_model
    for _ in range(config.num_layers):
        total_params += 4 * (config.dim_model ** 2)
        # gated feedforward
        total_params += 3 * config.dim_model * config.dim_ff

    total_params += config.dim_model * config.vocab_size

    return total_params / 1000**3


def get_flops(args, model, elapsed_time_per_iter ):
    gpus_per_model = bmt.config["tp_size"]
    batch_size = args.batch_size
    seq_len = args.max_length
    config = model.config
    approx_parameters_in_billions = get_parameters_in_billions(model.config)
    toks = batch_size * seq_len / elapsed_time_per_iter
    ffn_hidden_size = model.config.dim_ff

    hidden_size = model.config.dim_model
    num_layers = config.num_layers
    vocab_size = config.vocab_size
    num_heads = config.num_heads
    tflops = get_tflops(num_layers, hidden_size, ffn_hidden_size, num_heads, vocab_size, seq_len) 
    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 3 if not args.ckpt else 4
    # if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
    #     checkpoint_activations_factor = 4
    # if hasattr(args, 'recompute_granularity') and args.recompute_granularity == 'selective':
    #     checkpoint_activations_factor = 4
    # flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    # tflops = flops_per_iteration / (elapsed_time_per_iter * gpus_per_model * (10**12))
    tflops = batch_size * tflops * checkpoint_activations_factor / (elapsed_time_per_iter * gpus_per_model )
    return toks, tflops, approx_parameters_in_billions

def get_tokenizer(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    return tokenizer


def get_model(args):
    config = LlamaConfig.from_json_file(args.model_config)
    config.tp = 1 if (args.sp == 'none' and args.tp != 1) else 0
    config.sp = args.sp
    config.tp_sp = args.tp_sp
    config.casual_mask = args.mask
    config.ckpt = args.ckpt
    args._config = config
    if args.flash == "none":
        config.use_flash_attn = False
    else:
        config.use_flash_attn = True
        if args.flash == "1d":
            config.flash_attn_mask_shape = "1d"
        else:
            config.flash_attn_mask_shape = "2d"
            if args.flash == "triton":
                config.flash_impl = "triton"
            elif args.flash == "cuda":
                config.flash_impl = "cuda"
    model = Llama(config)
    if args.load is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints" + args.load)
        bmt.load(model, args.load)
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    else:
        optimizer = bmt.optim.AdamOptimizer(model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay)
    return optimizer


class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        return max(self.start_lr * 0.1, self.start_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    # lr_scheduler = bmt.lr_scheduler.Noam(
    lr_scheduler = Cosine(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):
    start = time.time()
    model = get_model(args)
    logger.info("load model in {:.2f}s".format(time.time() - start))

    start = time.time()
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    logger.info("load tokenizer in {:.2f}s".format(time.time() - start))

    start = time.time()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, lr_scheduler


def initialize():
    args = get_args(pretrain=True)
    args.fused_sl_loss = "sl_loss" in args.sp
    bmt.init_distributed(seed=args.seed, tp_size=args.tp)
    #import burst_attn
    #burst_attn.comm.init_comm_config(backend="bmt")
    bmt.config['sp_stream'] = torch.cuda.Stream(-1)
    bmt.config['sp_stream2'] = torch.cuda.Stream(-1)
    if args.sp != 'none' and not args.spzero:
        bmt.config['zero_comm'] = nccl.commInitRank(nccl.getUniqueId(), 1, 0)
    if args.load is not None:
        if args.start_step == 0:
            args.start_step = (int)(re.search("(\d+).pt", args.load)[1])
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    bmt.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage

def pretrain(
    args,
    tokenizer: LlamaTokenizer,
    model: Llama,
    optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    if args.fused_sl_loss:
        from burst_attn import SLFusedProjLoss
        sl_loss_func = SLFusedProjLoss(model.lm_head.weight, ignore_index=-100, reduction="mean", proj_scale=model.lm_head.scale)
    else:
        sl_loss_func = None

    optim_manager = bmt.optim.OptimManager(
        loss_scale= None if args.bf16 else args.loss_scale,
        loss_scale_steps=args.loss_scale_steps,
        loss_scale_factor=2,
        max_loss_scale=args.max_loss_scale,
        min_loss_scale=args.min_loss_scale,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    from dataloader import DataLoader
    dataloader = MixedDataset(args.dataset, args.batch_size, args.max_length, tokenizer, unpad=False, rank=0, world_size=1)
    # def forward_hook(module, input, output):
    #     bmt.print_rank(f"Module forwarding: {module.__class__.__name__}")
    #
    # for name, module in model.named_modules():
    #     module.register_forward_hook(forward_hook)

    dataloader.start()
    dataloader = dataloader
    bmt.print_rank("finish dataset start")
    mean_loss = 0.
    mean_ppl = 0.
    mean_forward = []
    mean_time = []
    max_memory_forward = None
    max_memory = 0.
    max_memory_allocated = 0.
    try:
        for it, data in enumerate(dataloader):
            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            targets.masked_fill_(targets == -100, 123)
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]
            # if args.flash == "cuda":
            #     cu_seqlens = torch.from_numpy(data["cu_seqlens"]).cuda().to(torch.int32)
            #     max_seqlen = data["max_seqlen"]
            #     position_ids = torch.from_numpy(data["position_ids"]).cuda().to(torch.int32)
            # else:
            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_context = torch.zeros_like(input_ids).cuda().bool()
            input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)

            # ===========
            # if it % args.grad_accum == 0:
            #     optim_manager.zero_grad()
            torch.cuda.empty_cache()
            mem_usage = {}
            tim_usage = {}

            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # ===========
            if args.eval_only:
                with torch.no_grad():
                    # if args.flash == "cuda":
                    #     logits, _ = model(
                    #         input_ids,
                    #         cu_seqlens=cu_seqlens,
                    #         max_seqlen=max_seqlen,
                    #         position_ids=position_ids,
                    #     )
                    # else:
                    logits, _ = model(
                        input_ids,
                        input_length,
                        input_context,
                        input_span,
                    )
                mem_usage, tim_usage = add_mem_time("forward_1", mem_usage, tim_usage)
            if not args.eval_only:
                # if args.flash == "cuda":
                #     logits, _ = model(
                #         input_ids,
                #         cu_seqlens=cu_seqlens,
                #         max_seqlen=max_seqlen,
                #         position_ids=position_ids,
                #     )
                # else:
                logits, _ = model(
                    input_ids,
                    input_length,
                    input_context,
                    input_span,
                )

                mem_usage, tim_usage = add_mem_time("forward_2", mem_usage, tim_usage)
                # targets = targets.view(-1).chunk(bmt.config['tp_size'])[bmt.config['tp_rank']].view(-1)
                if args.sp != "none" or args.tp_sp:
                    targets = targets.chunk(bmt.config['tp_size'], dim=1)[bmt.config['tp_rank']]
                if args.fused_sl_loss:
                    logits = logits.flatten(0, 1)
                    targets = targets.flatten(0, 1)
                    loss = sl_loss_func(logits, targets, 64)
                else:
                    loss = loss_func(logits.flatten(0, 1), targets.flatten(0, 1))
                logits, _ = None, None
                global_loss = bmt.sum_loss(loss).item()
                mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

                # ===========
                optim_manager.backward(loss)
                mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)
                # ===========
                grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
                optim_manager.step()
                mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)

                # ==========

            if args.eval_only:
                with torch.no_grad():
                    task_num = len(task_names)
                    targets_tmp = targets.expand(task_num, -1, -1)
                    task = torch.arange(task_num, dtype=torch.int32, device="cuda")[:, None, None]
                    targets_tmp = torch.where(
                        task_ids == task,
                        targets_tmp,
                        torch.scalar_tensor(-100, dtype=torch.int32, device="cuda"),
                    )

                    task_loss_map: Dict[str, float] = {}
                    task_loss_tot: Dict[str, float] = {}
                    for i in range(task_num):
                        task_loss_map[task_names[i]] = loss_func(logits.view(-1, logits.size(-1)), targets_tmp[i, :].view(-1)).item()
                        task_loss_tot[task_names[i]] = (targets_tmp[i, :].view(-1) >= 0).sum().float().item()
                    gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)
                    gatherd_task_loss_tot: List[Dict[str, float]] = allgather_objects(task_loss_tot)

                    global_task_loss_map: Dict[str, Union[List[float], float]] = {}
                    global_task_loss_tot: Dict[str, Union[List[float], float]] = {}

                    for idx, local_task_loss_map in enumerate(gatherd_task_loss_map):
                        for task_name, task_loss in local_task_loss_map.items():
                            if task_name not in global_task_loss_map:
                                global_task_loss_map[task_name] = []
                            global_task_loss_map[task_name].append(task_loss)
                        for task_name, task_tot in gatherd_task_loss_tot[idx].items():
                            if task_name not in global_task_loss_tot:
                                global_task_loss_tot[task_name] = []
                            global_task_loss_tot[task_name].append(task_tot)

                    task_loss_map = {}
                    for task_name in sorted(list(global_task_loss_map.keys())):
                        avg_loss = 0.0
                        sum_token = sum(global_task_loss_tot[task_name])
                        for loss, token in zip(global_task_loss_map[task_name], global_task_loss_tot[task_name]):
                            avg_loss += loss * token / sum_token
                        task_loss_map[task_name] = avg_loss
                    mean_forward.append(tim_usage['forward_1'] - tim_usage['init'])
                    if max_memory_forward is None:
                        max_memory_forward = mem_usage['forward_1'][1]
                logits = None
                # mean_loss += task_loss_map['wikipedia']
                # mean_ppl += torch.exp(torch.tensor(task_loss_map['wikipedia'])).item()
                mean_loss += task_loss_map['c4']
                mean_ppl += torch.exp(torch.tensor(task_loss_map['c4'])).item()

                bmt.print_rank(task_loss_map['c4'])
                bmt.print_rank(mem_usage)

                if args.ppl and it == 99:
                    bmt.print_rank(
                        (
                            "| loss: {:.4f} | ppl: {:.4f} |"
                        ).format(
                            mean_loss / (it+1), mean_ppl / (it+1),
                        )
                    )
                    break
                elif not args.ppl and it == 3:
                    mean_forward = sorted(mean_forward)
                    elapsed_time_per_iter = sum(mean_forward[1:-1])/(len(mean_forward)-2)
                    stderr = np.std(mean_forward[1:-1], ddof=1)
                    toks, tflops, approx_parameters_in_billions = get_flops(args, model, elapsed_time_per_iter)
                    bmt.print_rank(
                        (
                            "| mem_infer: {:.4f} | infer: {:.4f} | toks: {:.2f} | params: {:.2f} | std: {:.2f} |"
                        ).format(
                            max_memory_forward, sum(mean_forward[1:-1])/(len(mean_forward)-2), toks / bmt.world_size(), approx_parameters_in_billions, stderr
                        )
                    )
                    break
            else:
                mean_time.append(tim_usage['optim'] - tim_usage['init'])
                bmt.print_rank("total_time: ",tim_usage['backward'] - tim_usage['init'])
                max_memory = max(max_memory, max(mem_usage['forward_2'][1], mem_usage['backward'][1]))
                max_memory_allocated = max(max_memory_allocated, max(mem_usage['forward_2'][0], mem_usage['backward'][0]))
                bmt.print_rank(f"loss {global_loss:.4f} grad_norm {grad_norm:.4f}")
                bmt.print_rank("forward_time: ",tim_usage['forward'] - tim_usage['init'])
                bmt.print_rank("backward_time: ",tim_usage['backward'] - tim_usage['forward'])
                bmt.print_rank("optim_time: ",tim_usage['optim'] - tim_usage['backward'])
                elapsed_time_per_iter = mean_time[-1]
                bmt.print_rank(elapsed_time_per_iter )
                stderr = 0
                if bmt.rank() == 0 and it == 3:
                    torch.cuda.cudart().cudaProfilerStart()
                if it == 3:
                    if bmt.rank() == 0:
                        torch.cuda.cudart().cudaProfilerStop()
                    def mean_std(data):
                        mean = sum(data) / len(data)
                        std = np.std(data, ddof=1)
                        return mean, std
                    mean_time = mean_time[3:]
                    stats = [get_flops(args, model, t) for t in mean_time]
                    tok_mean, tok_std  = mean_std([t / bmt.config['tp_size'] for t, _, _ in stats])
                    time_mean, time_std = mean_std(mean_time)
                    if os.environ.get("LOG_FILE", None) is not None:
                        method = args.sp
                        sp_size = args.tp
                        world_size = bmt.world_size()
                        if bmt.rank() == 0:
                            with open(os.environ["LOG_FILE"], "a") as f:
                                f.write("******************\n")
                                f.write(f"method: {method}\n")
                                f.write(f"sp_size: {sp_size}\n")
                                f.write(f"seqlen: {args.max_length}\n")
                                f.write(f"world_size: {world_size}\n")
                                f.write("max_memory_reserved: {:.4f}\n".format(max_memory))
                                f.write("max_memory_allocated: {:.4f}\n".format(max_memory_allocated))
                                f.write(f"mean_time: {time_mean:.2f}+{time_std:.2f}\n")
                                f.write(f"toks: {tok_mean:.2f}+{tok_std:.2f}\n")
                    else:
                        print("LOG_FILE not set")
                    # toks, tflops, approx_parameters_in_billions = get_flops(args, model, mean_time)

                    break
                toks, tflops, approx_parameters_in_billions = get_flops(args, model, elapsed_time_per_iter)
                bmt.print_rank(
                    (
                        "|mem_train_allocated: {:.4f} |mem_train_reserved: {:.4f} | train: {:.4f} | tflops: {:.2f} | toks: {:.2f} | params: {:.2f} | std: {:.2f} |"
                    ).format(
                        max_memory_allocated, max_memory, elapsed_time_per_iter, tflops, toks / bmt.config['tp_size'], approx_parameters_in_billions, stderr
                    )
                )

    finally:
        # dataloader.close()
        pass



def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    bmt.print_rank("finish loading")
    pretrain(args, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
