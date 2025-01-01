import numpy as np
from flops import num_floating_point_operations
model_7b_params = (32, 4096, 4096*4, 32000, 32)
model_13b_params = (40, 5120, 5120*4, 32000, 40)
model_30b_params = (64, 6144, 6144*4, 32000, 64)

flops_func_7b = lambda seq: num_floating_point_operations(seq, *model_7b_params, swiglu=False)
flops_func_13b = lambda seq: num_floating_point_operations(seq, *model_13b_params, swiglu=False)
flops_func_30b = lambda seq: num_floating_point_operations(seq, *model_30b_params, swiglu=False)

# Calculate FLOPs for the given settings
res = [826.44, 987.5, 532.69, "OOM", 772.29, 868.75, 524.96, "OOM", 346.45, 418.75, 232.08, "OOM"]
flops_7b_128k = flops_func_7b(131072) / 131072
flops_7b_256k = flops_func_7b(262144) / 262144 
flops_13b_64k = flops_func_13b(65536) / 65536
flops_13b_128k = flops_func_13b(131072) / 131072
flops_30b_64k = flops_func_30b(65536) / 65536
flops_30b_128k = flops_func_30b(131072) / 131072

order = [flops_7b_128k, flops_7b_256k, flops_13b_64k, flops_13b_128k, flops_30b_64k, flops_30b_128k]
two_order = [ o for o in order for i in range(2)  ]
from IPython import embed;embed()
for i in range(len(res)):
    if res[i] == "OOM":
        res[i] = np.nan
    mfu = two_order[i] * res[i] / 312  / 10**12
    print(mfu)


