
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from convert_csv import add_flops
import logging
import sys

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
_logger.addHandler(_handler)
model_order = [
    "7b",
    "13b",
]

methods_order = [
    "Megatron TP",
    "Megatron CP",
    "Deepspeed Ulysses",
    "LoongTrain CP",
    "BurstAttention w. DoubleRing",
    "LoongTrain w. Ulysses(intra node)",
    # "BurstAttention w. Ulysses(intra node)",
    "BurstAttention",
    "BurstAttention w. Sparse",
]
methods = {
    "Burst-Sparse":"BurstAttention w. Sparse",
    "Burst-Ring": "BurstAttention w. DoubleRing",
    "BurstAttention": "BurstAttention",
    "Burst-USP": "BurstAttention w. Ulysses(intra node)",
    "megatron-cp": "Megatron CP",
    "Megatron": "Megatron CP",
    "megatron-tp": "Megatron TP",
    "ds-ulysses": "Deepspeed Ulysses",
    "Deepspeed-Ulysses": "Deepspeed Ulysses",
    "burst": "BurstAttention",
    "LoongTrain-Ring": "LoongTrain CP",
    "LoongTrain-USP": "LoongTrain w. Ulysses(intra node)",
}

whole_color_mapping = {
    "BurstAttention w. Sparse": "#E07A5F",
    'BurstAttention w. DoubleRing': '#0077B6',
    'BurstAttention': '#0077B6',
    'BurstAttention w. Ulysses(intra node)': '#E07A5F',
    'Deepspeed Ulysses': '#4CAF50',
    'Megatron CP': '#673AB7',
    'Megatron TP': '#FF4500',
    "LoongTrain CP": '#FFD700',
    "LoongTrain w. Ulysses(intra node)": '#FF4500',
}

whole_line_mapping = {
    "BurstAttention w. Sparse": "o",
    'BurstAttention w. DoubleRing': "xx",
    'BurstAttention': "\\",
    'BurstAttention w. Ulysses(intra node)': "\\",
    'Deepspeed Ulysses': "",
    'Megatron CP': "//",
    'Megatron TP': 'o',
    "LoongTrain CP": '.',
    "LoongTrain w. Ulysses(intra node)": 'o',
}

label_size = 13

title_size = 10

legend_size = 7

def clean_data(df):
    df = df[df.apply(lambda x: x['Method'] in methods, axis=1)]
    df['Method'] = df['Method'].apply(lambda x: methods[x])
    df['MFU'] = df['Throughput(TFLOPS/s)'] / 312
    return df

def barplot(data, x, y, hue, ax, palette=None, bar_width=1, hatches={}):
    group_width = 1.5
    space = 0.5
    groups = data[x].unique()
    categories = data[hue].unique()
    categories = sorted(categories, key=lambda x: methods_order.index(x))
    n_groups = len(groups)
    n_categories = len(categories)

    # 计算每个组的宽度和偏移量
    bar_width = group_width / n_categories
    group_pos = np.arange(n_groups) * (group_width + space)  # 为每个组添加间隔
    # 为每个类别画条形图
    for i, category in enumerate(categories):
        # 筛选出当前类别的数据
        category_data = data[data[hue] == category]
        # 通过分组平均值来绘图
        means = category_data.groupby(x)[y].mean().reset_index()
        means = means.sort_values(by=x, key=lambda x: x.map(model_order.index))[y]
        # 使用偏移量来调整条形位置
        offsets = group_pos + (i * bar_width)
        edge_color = 'black'
        ax.barh( offsets, means, height=bar_width, label=category, align='edge', hatch=hatches.get(category, None), color=palette.get(category, None), edgecolor=edge_color)

    # 设置图表的标签和图例
    ax.set_ylabel(x)
    ax.set_xlabel(y)
    ax.set_yticks(group_pos + (group_width) / 2)
    ax.set_yticklabels(groups)
    for i, bar in enumerate(ax.patches):
        y_value = bar.get_height()
        x_value = bar.get_x() + bar.get_width() / 2
        if np.isnan(y_value):
            ax.annotate('X', (x_value, 0), textcoords="offset points",
                        xytext=(0,1), ha='center', fontsize=label_size+1, fontweight='bold')


def plot_func(ax, df, y_label, msz, nodes) -> None:
    color_mapping = {}
    for m in df['Method'].unique():
        color_mapping[m] = whole_color_mapping[m]
    line_mapping = {}
    for m in df['Method'].unique(): 
        line_mapping[m] = whole_line_mapping[m]

    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # 调整子图之间的
    barplot(df, 'Model', y_label, 'Method', ax, palette=color_mapping, hatches=line_mapping)
    if nodes == 4:
        seqlen = "1024k" if msz == "13b" else "2048k"
    else:
        seqlen = "2048k" if msz == "13b" else "4096k"
    tick_size = 10
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    if y_label == 'Tokens/s':
        ax.set_xticks(np.arange(0, 2000, 400))
    else:
        ax.set_xticks(np.arange(0, 0.6, 0.1))
    plt.legend(df['Method'].unique(), loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 1.3), ncol=4)
    return ax, color_mapping, line_mapping, tick_size


def set_up_df(df) -> None:
    df['Seqlen(k)'] = df['Seqlen(k)'].astype(int)
    df.sort_values(by='Model', inplace=True, key=lambda x: x.map(model_order.index))
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
def plot_xlsx(model_size, nodes, labels):
    row = 4
    column = 1.5
    fig, axs = plt.subplots(len(model_size), len(labels), figsize=(row*len(labels), column*len(model_size)))
    for j,m in enumerate(model_size):
        df, filename = clean_df(m, nodes)
        for i, label in enumerate(labels):
            if len(model_size)> 1 and len(labels) > 1:
                ax = axs[i, j]
            else:
                if len(model_size) == 1 and len(labels) == 1:
                    ax = axs
                else:
                    ax = axs[i] if len(model_size) == 1 else axs[j]
            plot_throughput_and_tokens(df, label, ax, m, nodes)
    filename = f"{8 * nodes}x{model_size}.pdf" 
    plt.savefig(filename, bbox_inches='tight', dpi=500)
    plt.show()

# plot_xlsx(["7b", "13b"], 4, ["MFU", "Tokens/s"])
plot_xlsx(["7b_sparse"], 1, ["Tokens/s"])
# plot_xlsx(["7b", "13b"], 4, ["MFU", "Tokens/s"])

