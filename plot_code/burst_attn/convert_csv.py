import pandas as pd
import numpy as np
from flops import num_floating_point_operations

# num_layers, hidden_size, ffn_hidden, vocab, attn_heads
model_7b_params = (32, 4096, 11008, 32000, 32)
model_13b_params = (40, 5120, 13824, 32000, 40)
model_14b_params = (40, 5120, 13824, 120000, 40)
params_dict = {
    "7b": model_7b_params,
    "13b": model_13b_params,
    "14b": model_14b_params,
}
flops_func = lambda seq, c: num_floating_point_operations(seq, *params_dict[c])
def parse_tokens(x, tok_col="Tokens/s"):
    if isinstance(x[tok_col], str):
        if "+" in x[tok_col]:
            val = x[tok_col].split("+")[0]
        elif "±" in x[tok_col]:
            val = x[tok_col].split("±")[0]
        elif "OOM" in x[tok_col]: 
            val = np.nan
        else:
            val = x[tok_col]
    else:
        if x[tok_col] == 0:
            val = np.nan
        else:
            val = x[tok_col]
    val = float(val)
    return val

def flops_throughput(x, config):
    if isinstance(x['Seqlen(k)'], str):
        x['Seqlen(k)'] = float(x['Seqlen(k)'].strip("k"))
    if x['toks'] > 0:
        time = x['Seqlen(k)'] / x['toks']
    else:
        time = np.nan
    import re
    
    model_size = re.search(r'\d+', config).group(0)
    res = flops_func(x['Seqlen(k)'] * 1024, model_size+"b") / (time * 1000)
    x['Throughput(TFLOPS/s)'] = res / (10 ** 12)
    return x

def read_file(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name)
    return data

def select_data(x):
    if float(x['Seqlen(k)'].strip("k")) > 512:
        return False
    
    return True

def add_flops(df, size, tok_col="Tokens/s"):
    df['toks'] = df.apply(lambda x: parse_tokens(x, tok_col=tok_col),axis=1)
    df = df.apply(lambda x: flops_throughput(x, size), axis=1)
    df['Tokens/s'] = df['toks']
    df = df.drop('toks', axis=1)
    return df
# filename = "burst_exp.xlsx"
# df = read_file(filename, "7b model 4 nodes")
# df = read_file(filename, "7b model 1 nodes")
# df['toks'] = df.apply(parse_tokens,axis=1)
# size=""
# df = df.apply(lambda x: flops_throughput(x, size), axis=1)
# df['Tokens/s'] = df['toks']
# df = df.drop('toks', axis=1)
# df.to_csv("single.csv",index=False)
