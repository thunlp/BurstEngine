# Copyright (c) InternLM. All rights reserved.

from internlm.model.modeling_internlm import InternLM1
from internlm.model.modeling_llama import Llama2
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

LOAD_FUNC_DICT = {
    "llama": Llama2.load_llama_pretrained_weights,
    "internlm_test": InternLM1.load_internlm_with_dynamic_parallel_size,
}
