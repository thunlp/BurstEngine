import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from convert_csv import add_flops
import matplotlib.patheffects as path_effects

import logging
import sys
from utils import methods_order, methods, whole_color_mapping, whole_line_mapping

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_logger.addHandler(_handler)

label_size = 13

toverlapping itle_size = 10

legend_size = 7


def clean_data(df):
    df = df[df.apply(lambda x: x["Method"] in methods, axis=1)]
    df["Method"] = df["Method"].apply(lambda x: methods[x])
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
    abc_str = "ABCDEFG"
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
            "ABCDEFG"[i % 7],
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
    color_mapping = {}
    for m in df["Method"].unique():
        color_mapping[m] = whole_color_mapping[m]
    line_mapping = {}
    for m in df["Method"].unique():
        line_mapping[m] = whole_line_mapping[m]

    plt.subplots_adjust(hspace=0.7, wspace=0.5)  # 调整子图之间的
    barplot(
        df,
        "Seqlen(k)",
        y_label,
        "Method",
        ax,
        palette=color_mapping,
        hatches=line_mapping,
    )
    if nodes == 4:
        seqlen = "2M" if msz == "7b" else "1M"
    else:
        seqlen = "4M" if msz == "7b" else "2M"
    # ax.set_title(f'{y_label} \nfor {msz} {seqlen}\n on {nodes*8}xA100', fontsize=title_size, fontweight='bold', y=1.)
    ax.set_title(
        f"{y_label} \n {seqlen} Sequence\n", fontsize=title_size, fontweight="bold", y=0.84
    )
    # ax.set_xlabel('Seqlen (k)', fontsize=label_size)
    # ax.set_ylabel(y_label, fontsize=label_size)
    tick_size = 10
    ax.tick_params(axis="x", labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    if y_label == "Tokens/s":
        ax.set_yticks(np.arange(0, 140, 20))
    else:
        ax.set_yticks(np.arange(0, 0.6, 0.1))
    return ax, color_mapping, line_mapping, tick_size


def set_up_df(df) -> None:
    df["Seqlen(k)"] = df["Seqlen(k)"].astype(int)
    df.sort_values(by="Seqlen(k)", inplace=True)
    df = df.sort_values(by="Method", key=lambda x: x.map(methods_order.index))

    # Set the style for the plot
    sns.set(style="whitegrid", font_scale=1.2)
    return df


def plot_throughput_and_tokens(df, label, ax, msz, nodes):
    # Convert 'Seqlen(k)' to integer for better grouping
    df = set_up_df(df)

    ax, color_mapping, line_mapping, tick_size = plot_func(ax, df, label, msz, nodes)


def clean_df(model_size, nodes) -> None:
    filename = "burst_exp.xlsx"
    sheet = f"{model_size} model {nodes} nodes"
    logging.info(f"Plotting {sheet} from {filename}")
    # try:
    df = pd.read_excel(filename, sheet)
    df = add_flops(df, model_size, "Tokens/s")
    df = clean_data(df)
    # df = df[df['Seqlen(k)'] <= 512]
    # except Exception as e:
    #     logging.error(f"Failed to read {sheet} from {filename}")
    #     logging.error(e)
    # return
    return df, filename


# Example usage
def plot_xlsx(model_size, nodes_num, labels):
    row = 1.5
    column = 2
    fig, axs = plt.subplots(
        len(nodes_num),
        len(labels) * len(model_size),
        figsize=(row * len(labels) * len(model_size), column * len(nodes_num)),
    )
    print(len(axs))
    for _n, nodes in enumerate(nodes_num):
        for _j, m in enumerate(model_size):
            df, filename = clean_df(m, nodes)
            for _i, label in enumerate(labels):
                i = len(nodes_num) * _j + _i
                j = _j
                if len(model_size) > 1 and len(labels) > 1:
                    print(axs.shape)
                    ax = axs[_n, i]
                else:
                    ax = axs[i] if len(model_size) == 1 else axs[j]
                plot_throughput_and_tokens(df, label, ax, m, nodes)
                if _i == 0:
                    ax.text(
                        1.3,
                        1.3,
                        f"{m.upper()} {nodes*8}xA800",
                        fontsize=12,
                        fontweight="bold",
                        transform=ax.transAxes,
                        va="bottom",
                        ha="center",
                    )
    filename = f"{nodes_num}x{model_size}.pdf"
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_size = 9
    plt.figlegend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=len(labels) // 2,
        fontsize=legend_size,
        columnspacing=0.5,
        prop={"size": legend_size, "weight": "bold"},
        frameon=True,
        edgecolor="black",
    )
    plt.savefig(filename, bbox_inches="tight", dpi=500)
    file_name = "/Users/tachicoma/Notes/markdowns/Research/Burst/working_dir/exp_images/main_exp.pdf"
    plt.savefig(file_name, bbox_inches="tight", dpi=500)
    # plt.show()


plot_xlsx(["7b", "14b"], [4, 8], ["MFU", "Tokens/s"])
# plot_xlsx(["13b"], 8, ["MFU", "Tokens/s"])
# plot_xlsx(["7b", "13b"], 4, ["MFU", "Tokens/s"])
