
import matplotlib.pyplot as plt
import math
import seaborn as sns
from flops import num_floating_point_operations
import pandas as pd

# Define model parameters
model_7b_params = (32, 4096, 11008, 32000, 32)
model_13b_params = (40, 5120, 13824, 32000, 40)
model_30b_params = (64, 6144, 16588, 32000, 64)
model_70b_params = (80, 8192, 28672, 32000, 80)
qwen_25_params = (40, 5120, 13824, 152064, 40)
llama3_8b = (32, 4096, 14336, 128256, 32)
params_dict = {
    "7b": model_7b_params,
    "13b": model_13b_params,
    "30b": model_30b_params,
    "70b": model_70b_params,
    "8b_llama3": llama3_8b,
}
def get_mem(seq, c, is_lightseq=False):
    num_layers, hidden, ffn, vocab_size, num_heads = params_dict[c]
    # _, hidden, _, vocab_size, _ = params_dict[c]
    return num_layers * seq * hidden * 2


# Set up the plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(4, 4))

# Prepare data for seaborn
data = []
x_var = [128, 32*8, 64*8, 128*8]
labels = ["Selective Checkpointing++", "BurstEngine"]
for i, label in enumerate(labels):
    mems = [get_mem(seq * 1024 // (2 if i > 0 else 1), "7b", False) / 1024**3 for seq in x_var]
    for seq, mem in zip(x_var, mems):
        data.append({"Sequence Length": seq, "Memory (GB)": mem, "Method": label})

df = pd.DataFrame(data)

# Plot using seaborn
colors = ['#FFF1C3', '#C5E0B4']
hatches = {
    "Selective Checkpointing++": "--",
    "BurstEngine": "\\",
}

methods = df['Method'].unique()
hatches = ['', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'][:len(methods)]  # Example hatches, can be customized

bar_width = 0.35
r = range(len(df['Sequence Length'].unique()))

for i, method in enumerate(methods):
    subset = df[df['Method'] == method]
    plt.bar([pos + i*bar_width for pos in r], subset['Memory (GB)'], width=bar_width, label=method, color=colors[i], edgecolor='black', hatch=hatches[i])

# Applying hatches based on method

# Customize the plot
# plt.xticks(x_var, [f"{t}k" for t in x_var], fontsize=14, fontname="Arial")
ranger_scale = range(0, 320, 50)
plt.yticks(ranger_scale, [f"{g} GB" for g in ranger_scale], fontsize=14)
plt.ylim(0, 350)
# plot_scale("13b", x_var)
# plot_scale("30b", x_var)

# Add title and labels
# ticks = [1, 256, 512, 1024]
# plt.xscale('log', base=2)
# plt.xticks(x_var)
plt.xticks(range(len(x_var)), [f"{t}k" for t in x_var], fontsize=14, fontname="Arial")
# plt.yticks(range(0, max(), 20), [f"{g} GB" for g in range(0, 140, 20)], fontsize=14)
plt.title("Memory (GB)", fontsize=18, weight='bold', y=1)
plt.xlabel("Sequence Length", fontsize=18, weight='bold')
# plt.ylabel("Attention Computation Cost (%)", fontsize=14)
fontsize=12
plt.legend(fontsize=fontsize, loc="upper left", prop={'weight': 'bold', "size":fontsize}, bbox_to_anchor=(-0.02, 1.02))

# Add grid and show the plot
ax = plt.gca()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
plt.tight_layout()
plt.savefig("/Users/tachicoma/Docs/tex/BurstEngine/sc_2025/samples/exp_images/recompute_mem.pdf", bbox_inches='tight', dpi=500, transparent=True)
