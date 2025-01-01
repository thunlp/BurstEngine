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

methods_order = [
    "Megatron TP",
    "Megatron CP",
    "Deepspeed Ulysses",
    "LoongTrain CP",
    "BurstAttention w. DoubleRing",
    "LoongTrain w. Ulysses(intra node)",
    "BurstAttention w. Ulysses(intra node)",
    "BurstAttention",
]
methods = {
    "Burst-Ring": "BurstAttention w. DoubleRing",
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

def normal_barplot(data, x, y, hue, ax, palette=None, bar_width=1, hatches={}):
    # Calculate the width and offset for each group
    group_width = 2.0
    n_categories = len(data[hue].unique())
    bar_width = group_width / n_categories
    
    # Set chart labels and legend
    ax.set_ylabel(y, fontsize=12, fontweight='bold')
    ax.set_xlabel('', fontsize=12, fontweight='bold')
    
    # Plot bars with custom colors and hatches
    data[y] = data[y].apply(lambda x: 0 if np.isnan(x) else x)
    bars = ax.bar(range(len(data[y])), data[y], 
                  color=[palette.get(x, None) for x in data[hue]], 
                  hatch=[hatches.get(x, None) for x in data[hue]], 
                  label=data[hue], 
                  width=bar_width,  # Adjust the width to make bars closer
                  align='edge', 
                  edgecolor='black', 
                  linewidth=0.5)
    
    # Remove x-ticks
    ax.set_xticks([])
    
    # Add X for NaN values with improved styling
    for i, bar in enumerate(bars):
        y_value = bar.get_height()
        x_value = bar.get_x() + bar.get_width() / 2
        if y_value == 0:
            ax.annotate('X', (x_value, 0), 
                        textcoords="offset points",
                        xytext=(0, 2), 
                        ha='center', 
                        fontsize=12, 
                        fontweight='bold', 
                        color=palette.get(bar.get_label(), None))
    
    # Add labels 'abcdefg' at the bottom of each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, -0.05, 
                'ABCDEFG'[i % 7], 
                ha='center', 
                va='top', 
                fontsize=10, 
                fontweight='bold')
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Set background color for the plot area
    ax.set_facecolor('#f0f0f0')
    
    # Add padding around the plot
    ax.margins(x=0.05, y=0.1)

def barplot(data, x, y, hue, ax, palette=None, bar_width=1, hatches={}):
    group_width = 2.0
    space = 0.5
    groups = data[x].unique()
    categories = data[hue].unique()
    disable_groupby=len(groups) == 1
    if disable_groupby:
        return normal_barplot(data,x, y, hue, ax, palette, bar_width, hatches)
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
        if disable_groupby:
            means = category_data[y]
        else:
            means = category_data.groupby(x)[y].mean()
        # 使用偏移量来调整条形位置
        offsets = group_pos + (i * bar_width)
        edge_color = 'black'
        ax.bar(offsets, means, width=bar_width, label=category, align='edge', hatch=hatches.get(category, None), color=palette.get(category, None), edgecolor=edge_color)

    # 设置图表的标签和图例
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticks(group_pos + (group_width) / 2)
    ax.set_xticklabels(groups)
    for i, bar in enumerate(ax.patches):
        y_value = bar.get_height()
        x_value = bar.get_x() + bar.get_width() / 2
        if np.isnan(y_value):
            ax.annotate('X', (x_value, 0), textcoords="offset points",
                        xytext=(0,1), ha='center', fontsize=label_size+1, fontweight='bold')
    # ax.legend(title=hue)

def plot_throughput_and_tokens(df):
    # Convert 'Seqlen(k)' to integer for better grouping
    df['Seqlen(k)'] = df['Seqlen(k)'].astype(int)
    df.sort_values(by='Seqlen(k)', inplace=True)
    df = df.sort_values(by="Method", key=lambda x: x.map(methods_order.index))

    # Set the style for the plot
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axs = plt.subplots(2, 1, figsize=(3, 4))

    # Define color and hatch mappings

    # Plot Throughput
    color_mapping = {}
    for m in df['Method'].unique():
        color_mapping[m] = whole_color_mapping[m]
    line_mapping = {}
    for m in df['Method'].unique(): 
        line_mapping[m] = whole_line_mapping[m]

    ax = axs[0]
    plt.subplots_adjust(hspace=0.5)  # 调整子图之间的
    barplot(df, 'Seqlen(k)', 'MFU', 'Method', ax, palette=color_mapping, hatches=line_mapping)
    ax.set_title('MFU for {}-32xA100 1024k Training', fontsize=title_size, fontweight='bold', y=1.)
    # ax.set_xlabel('Seqlen (k)', fontsize=label_size)
    ax.set_ylabel('MFU', fontsize=label_size)
    tick_size = 10
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_yticks(np.arange(0, 0.6, 0.1))

    # Plot Tokens/s
    ax = axs[1]
    barplot(df, 'Seqlen(k)', 'Tokens/s', 'Method', ax, palette=color_mapping, hatches=line_mapping)
    ax.set_title('Tokens/s for 7B-32xA100 1024k Training', fontsize=title_size, fontweight='bold', y=1.)
    # ax.set_xlabel('Seqlen (k)', fontsize=label_size)
    ax.set_ylabel('Tokens/s', fontsize=label_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels,  loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(labels) // 3, fontsize=legend_size, columnspacing=0.5)

    # Adjust layout
    # plt.tight_layout()

# Example usage
def plot_xlsx(model_size, nodes):
    filename = "burst_exp.xlsx"
    sheet = f"{model_size} model {nodes} nodes" 
    logging.info(f"Plotting {sheet} from {filename}")
    try:
        df = pd.read_excel(filename, sheet)
        df = add_flops(df, model_size, "Tokens/s")
        df = clean_data(df)
        # df = df[df['Seqlen(k)'] <= 512]
    except Exception as e:
        logging.error(f"Failed to read {sheet} from {filename}")
        logging.error(e)
        return
    plot_throughput_and_tokens(df)
    filename = f"{8 * nodes}x{model_size}.pdf" 
    plt.savefig(filename, bbox_inches='tight', dpi=500)
    plt.show()

# plot_xlsx("7b", 1)
plot_xlsx("13b", 8)
# plot_xlsx("7b", 4)

