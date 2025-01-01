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

from collections import OrderedDict

import torch


def convert7b():
    ckpt = torch.load("/mnt/data/user/tc_agi/user/zhaoweilin/cpmlive-llama-2-7b/pytorch_model.pt")
    new_ckpt = OrderedDict()

    def init(nv, v):
        mean = v.float().mean().half()
        std = v.float().std().half()
        print(mean, std)
        torch.nn.init.normal_(nv, mean=0, std=std)
        # pass

    for k, v in ckpt.items():
        if k in ["input_embedding.weight", "lm_head.weight"]:
            print(k)
            nv = torch.zeros((121472, 4096), dtype=torch.half)
            init(nv, v)
            nv[:32000, :] = v
            new_ckpt[k] = nv
        else:
            new_ckpt[k] = v

    torch.save(new_ckpt, "new-vocab-llama-2-7b.pt")


def convert70b():
    ckpt = torch.load("/mnt/data/user/tc_agi/user/zhaoweilin/cpmlive-llama-2-70b/pytorch_model.pt")
    new_ckpt = OrderedDict()

    def init(nv, v):
        mean = v.float().mean().half()
        std = v.float().std().half()
        print(mean, std)
        torch.nn.init.normal_(nv, mean=0, std=std)
        # pass

    for k, v in ckpt.items():
        if k in ["input_embedding.weight", "lm_head.weight"]:
            print(k)
            nv = torch.zeros((121472, 8192), dtype=torch.half)
            init(nv, v)
            nv[:32000, :] = v
            new_ckpt[k] = nv
        else:
            new_ckpt[k] = v

    torch.save(new_ckpt, "new-vocab-llama-2-70b.pt")


if __name__ == "__main__":
    convert7b()
    convert70b()
