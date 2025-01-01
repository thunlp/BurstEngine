import matplotlib.pyplot as plt
import math
import seaborn as sns
from flops import num_floating_point_operations

# Define model parameters
model_7b_params = (32, 4096, 11008, 32000, 32)
model_13b_params = (40, 5120, 13824, 32000, 40)
model_30b_params = (64, 6144, 16588, 32000, 64)
model_70b_params = (80, 8192, 28672, 32000, 80)
params_dict = {
    "7b": model_7b_params,
    "13b": model_13b_params,
    "30b": model_30b_params,
    "70b": model_70b_params,
}
def flops_func(seq, c, **kwargs):
    return num_floating_point_operations(seq, *params_dict[c], **kwargs)

def plot_scale(msz, seqs):
    """Plot the scaling of Attention computation percentage with sequence length for a given model size."""
    tflops = [flops_func(seq * 1024, msz) / 1e12 for seq in seqs]
    attn_tflops = [flops_func(seq * 1024, msz, attn_only=True) / 1e12 for seq in seqs]
    attn_percentage = [attn / total * 100 for attn, total in zip(attn_tflops, tflops)]
    plt.plot(seqs, attn_percentage, label=f"{msz} model", marker='o', linestyle='-', linewidth=2, color='#8B0000', markeredgecolor='black', markeredgewidth=1)

# Set up the plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(3.3, 4))

# Plot the scaling for both model sizes
x_var = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
plot_scale("7b", x_var)
# plot_scale("13b", x_var)
# plot_scale("30b", x_var)

# Add title and labels
ticks = [1, 256, 512, 1024]
# plt.xscale('log', base=2)
# plt.xticks(ticks)
plt.xticks(ticks, [f"{t}k" for t in ticks], fontsize=14)
plt.yticks(range(0, 120, 20), ['0%', '20%', '40%', '60%', '80%', "100%"], fontsize=14)
plt.title("Attention Operation\n Time (%)", fontsize=18, weight='bold')
plt.xlabel("Sequence Length", fontsize=18, weight='bold')
# plt.ylabel("Attention Computation Cost (%)", fontsize=14)
# plt.legend(fontsize=12)

# Add grid and show the plot
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
plt.tight_layout()
plt.savefig("/Users/tachicoma/Docs/tex/BurstEngine/atc/images/attn_scale.pdf", bbox_inches='tight', dpi=500, transparent=True)
plt.show()
