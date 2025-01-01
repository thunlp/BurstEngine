from enum import Enum

class ModelSize(Enum):
    CONFIG_7B = {
        "VOCAB_SIZE": 32000,
        "HIDDEN_SIZE": 4096,
        "MLP_RATIO": 2.6875,
        "NUM_ATTENTION_HEAD": 32,
        "NUM_KV_ATTENTION_HEAD": 32,
        "NUM_LAYER": 32,
    }

    CONFIG_70B = {
        "VOCAB_SIZE": 32000,
        "HIDDEN_SIZE": 8192,
        "NUM_ATTENTION_HEAD": 64,
        "NUM_KV_ATTENTION_HEAD": 64,
        "MLP_RATIO": 3.5,
        "NUM_LAYER": 80,
    }

    CONFIG_13B = {
        "VOCAB_SIZE": 32000,
        "HIDDEN_SIZE": 5120,
        "NUM_ATTENTION_HEAD": 40,
        "NUM_KV_ATTENTION_HEAD": 40,
        "MLP_RATIO": 2.7,
        "NUM_LAYER": 40,
    }

    CONFIG_30B = {
        "VOCAB_SIZE": 32000,
        "HIDDEN_SIZE": 6144,
        "NUM_ATTENTION_HEAD": 64,
        "NUM_KV_ATTENTION_HEAD": 64,
        "MLP_RATIO": 2.7,
        "NUM_LAYER": 64,
    }
name_map ={
    "7b": ModelSize.CONFIG_7B,
    "70b": ModelSize.CONFIG_70B,
    "13b": ModelSize.CONFIG_13B,
    "30b": ModelSize.CONFIG_30B,
}

def get_config(size, type="llama"):
    assert size in name_map, f"size {size} not found"
    if type == "llama":
        return name_map[size].value
    elif type == "gpt2":
        res = name_map[size].value
        res["MLP_RATIO"] = 2
        return res
    else:
        raise ValueError(f"type {type} not supported")
