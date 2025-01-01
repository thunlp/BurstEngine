
methods_order = [
    "Megatron TP",
    "Megatron CP",
    "Deepspeed-Ulysses",
    "LoongTrain-DoubleRing",
    "LoongTrain-USP",
    "BurstEngine w. Ulysses(intra node)",
    "BurstEngine",
    "BurstEngine valina",
]
methods = {
    "Burst-Ring": "BurstEngine",
    # "Burst-USP": "BurstEngine w. Ulysses(intra node)",
    "megatron-cp": "Megatron CP",
    "Megatron": "Megatron CP",
    "megatron-tp": "Megatron TP",
    "ds-ulysses": "Deepspeed-Ulysses",
    "Deepspeed-Ulysses": "Deepspeed-Ulysses",
    "burst": "BurstEngine valina",
    "LoongTrain-Ring": "LoongTrain-DoubleRing",
    "LoongTrain-USP": "LoongTrain-USP",
}

whole_color_mapping = {
    "BurstEngine": "#0077B6",
    "BurstEngine valina": "#0077B6",
    "BurstEngine w. Ulysses(intra node)": "#E07A5F",
    "Deepspeed-Ulysses": "#4CAF50",
    "Megatron CP": "#673AB7",
    "Megatron TP": "#FF4500",
    "LoongTrain-DoubleRing": "#FFD700",
    "LoongTrain-USP": "#FF4500",
}

whole_line_mapping = {
    "BurstEngine": "xx",
    "BurstEngine valina": "\\",
    "BurstEngine w. Ulysses(intra node)": "\\",
    "Deepspeed-Ulysses": "",
    "Megatron CP": "//",
    "Megatron TP": "o",
    "LoongTrain-DoubleRing": ".",
    "LoongTrain-USP": "o",
}
