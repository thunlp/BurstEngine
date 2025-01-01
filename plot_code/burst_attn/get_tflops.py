from flops import num_floating_point_operations

# Define model parameters
model_7b_params = (32, 4096, 11008, 32000, 32)
model_13b_params = (40, 5120, 13824, 32000, 40)
model_14b_params = (40, 5120, 13824, 120000, 40)
model_30b_params = (64, 6144, 16588, 32000, 64)
model_70b_params = (80, 8192, 28672, 32000, 80)
params_dict = {
    "7b": model_7b_params,
    "13b": model_13b_params,
    "30b": model_30b_params,
    "70b": model_70b_params,
    "14b": model_14b_params
}
def flops_func(seq, c, **kwargs):
    return num_floating_point_operations(seq, *params_dict[c], **kwargs)

def calc_flops_per_sec(toks, seq, c):
    flops  = flops_func(seq, c)
    return flops / (seq/toks)

model = "7b"
seq = 2000
# toks = input("Enter TGS: ")
for toks in [6000]:
    toks = float(toks)
    flops_per_sec = calc_flops_per_sec(toks, seq, model)
    print(f"{float(flops_per_sec) / 1e12 :.2f}")


