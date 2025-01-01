
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from convert_csv import add_flops
import matplotlib.patheffects as path_effects

import logging
import sys
from utils import methods_order, methods

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_logger.addHandler(_handler)

label_size = 13

title_size = 10

legend_size = 7


def clean_data(df):
    # df = df[df.apply(lambda x: x["GPUs"] in methods, axis=1)]
    # df["GPUs"] = df["GPUs"].apply(lambda x: methods[x])
    df["MFU"] = df["Throughput(TFLOPS/s)"] / 312
    return df


def normal_barplot(data, x, y, hue, ax, palette=None, bar_width=1, hatches={}):
    # Calculate the width and offset for each group
    group_width = 2.0
    n_categories = len(data[hue].unique())
    bar_width = group_width / n_categories

    # Set chart labels and legend
    ax.set_ylabel("", fontsize=12, fontweight="bold")
    ax.set_xlabel("", fontsize=12, fontweight="bold")

    # Plot bars with custom colors and hatches
    data[y] = data[y].apply(lambda x: 0 if np.isnan(x) else x)
    abc_str = "1248"
    labels = [f"({abc_str[i]}) {h}" for i, h in enumerate(data[hue])]
    bars = ax.bar(
        range(len(data[y])),
        data[y],
        color=[palette.get(x, None) for x in data[hue]],
        hatch=[hatches.get(x, None) for x in data[hue]],
        label=labels,
        width=bar_width,  # Adjust the width to make bars closer
        align="edge",
        edgecolor="black",
        linewidth=0.5,
    )

    # Remove x-ticks
    ax.set_xticks([])

    # Add X for NaN values with improved styling
    for i, bar in enumerate(bars):
        y_value = bar.get_height()
        x_value = bar.get_x() + bar.get_width() / 2
        print(bar.get_label()[3:])
        print(palette)
        if y_value == 0:
            ax.annotate(
                "X",
                (x_value, 0),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=palette.get(bar.get_label()[4:], None),
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )

    # Add labels 'abcdefg' at the bottom of each bar
    offset = -0.05 if y == "MFU" else -10
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            offset,
            "1248"[i % 4],
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # Add grid lines for better readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Set background color for the plot area
    ax.set_facecolor("#f0f0f0")

    # Add padding around the plot
    # ax.margins(x=0.05, y=0.1)


def barplot(data, x, y, hue, ax, palette=None, bar_width=1, hatches={}):
    group_width = 2.0
    space = 0.5
    groups = data[x].unique()
    categories = data[hue].unique()
    disable_groupby = len(groups) == 1
    return normal_barplot(data, x, y, hue, ax, palette, bar_width, hatches)


def plot_func(ax, df, y_label, msz, nodes) -> None:

    plt.subplots_adjust(hspace=0.7, wspace=0.5)  # 调整子图之间的
    barplot(
        df,
        "Seqlen(k)",
        y_label,
        "GPUs",
        ax,
    )
    tick_size = 10
    ax.tick_params(axis="x", labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    if y_label == "Tokens/s":
        ax.set_yticks(np.arange(0, 140, 20))
    else:
        ax.set_yticks(np.arange(0, 0.6, 0.1))
    return ax, tick_size


def set_up_df(df) -> None:
    # df = df.sort_values(by="GPUs", key=lambda x: x.map(methods_order.index))

    # Set the style for the plot
    sns.set(style="whitegrid", font_scale=1.2)
    return df


def plot_throughput_and_tokens(df, label, ax, msz, nodes):
    # Convert 'Seqlen(k)' to integer for better grouping
    df = set_up_df(df)

    ax,  tick_size = plot_func(ax, df, label, msz, nodes)




# Example usage
def plot_xlsx(is_multi_node=False):
    row = 1.5
    column = 2
    filename = "burst_exp.xlsx"
    sheet = "scaling 14b"
    df = pd.read_excel(filename, sheet)
    df = add_flops(df, "14b", "Tokens/s")
    df = clean_data(df)
    fig, axs = plt.subplots(
        1, 2,
        figsize=(5, 2)
    )
    df = df.rename(columns={'Mem': 'Memory (GB)'})
    df = df.rename(columns={'MFU': 'MFU (%)'})
    df['TFLOPs/s'] = df['Throughput(TFLOPS/s)'] * df['GPUs']
    # if not is_multi_node:
    #     df.plot(x="GPUs", y="MFU (%)", ax=axs[0], kind="bar", color="#0077B6", edgecolor="black", linewidth=0.5, hatch="xx", legend=False)
    # else:
    #     df['GPUs'] = df['GPUs'].apply(lambda x: x//8)
    #     df.plot(x="GPUs", y="MFU (%)", ax=axs[0], kind="bar", color="#0077B6", edgecolor="black", linewidth=0.5, hatch="xx", legend=False)
    df1 = df[~df['Offload']] if is_multi_node else df[df['Offload']]
    df2 = df[df['Offload']] if is_multi_node else df[~df['Offload']]
    from IPython import embed;embed()
    df1.plot(x="GPUs", y="TFLOPs/s", ax=axs[0], kind="line", color="#0077B6",  linewidth=0.5,  legend=False)
    df2.plot(x="GPUs", y="TFLOPs/s", ax=axs[1], kind="line", color="#0077B6",  linewidth=0.5,  legend=False)
    # axs[0].set_xlabel("Node Size", fontweight='bold', fontsize=12)
    # axs[1].set_xlabel("Node Size", fontweight='bold', fontsize=12)
    # axs[0].set_ylim(0, 0.6)
    # axs[0].set_yticks(np.arange(0, 0.7, 0.1))
    #
    # axs[0].tick_params(axis='x', rotation=0)
    # axs[1].tick_params(axis='x', rotation=0)
    # axs[0].grid(axis="y", linestyle="--", alpha=0.7)
    # axs[0].set_facecolor("#f0f0f0")
    # axs[1].grid(axis="y", linestyle="--", alpha=0.7)
    # axs[1].set_facecolor("#f0f0f0")
    # axs[0].set_title("MFU (%)", fontsize=title_size, fontweight="bold", y=1.05)
    # axs[1].set_title("Memory (GB)", fontsize=title_size, fontweight="bold", y=1.05)

    plt.subplots_adjust(hspace=0.7, wspace=0.5)  # 调整子图之间的
    plt.show()
    handles, labels = plt.gca().get_legend_handles_labels()
    # multi_label = "multi" if is_multi_node else "single"
    # df.to_latex(f"./{multi_label}.tex")
    # file_name = f"/Users/tachicoma/Notes/markdowns/Research/Burst/working_dir/exp_images/scale_plot_{multi_label}.pdf"
    # plt.savefig(file_name, bbox_inches="tight", dpi=500)

    # plt.show()

plot_xlsx(True)
