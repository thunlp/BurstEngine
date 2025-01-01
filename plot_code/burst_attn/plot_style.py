import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def barplot(data, x, y, hue, ax, palette=None, bar_width=1, hatches={}):
    group_width = 2.0
    space = 0.5

    groups = data[x].unique()
    categories = data[hue].unique()
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
        means = category_data.groupby(x)[y].mean()
        # 使用偏移量来调整条形位置
        offsets = group_pos + (i * bar_width)
        edge_color = 'black'
        ax.bar(offsets, means, width=bar_width, label=category, align='edge', hatch=hatches.get(category, None), color=palette.get(category, None), edgecolor=edge_color)

    # 设置图表的标签和图例
    plt.subplots_adjust(wspace=0.1)  # 调整子图之间的
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticks(group_pos + (group_width) / 2)
    ax.set_xticklabels(groups)
    ax.legend(title=hue)

def plot_throughput_and_tokens(df):
    # Convert 'Seqlen(k)' to integer for better grouping
    df['Seqlen(k)'] = df['Seqlen(k)'].astype(int)

    # Set the style for the plot
    sns.set(style="whitegrid", font_scale=1.2)
    fig, axs = plt.subplots(2, 1, figsize=(14, 16))

    # Define color and hatch mappings
    color_mapping = {
        'Method1': '#FFFFFF',
        'Method2': '#B0A8B9',
        'Method3': '#4B4453',
    }
    line_mapping = {
        'Method1': "//",
        'Method2': "\\",
        'Method3': "",
    }

    # Plot Throughput
    ax = axs[0]
    barplot(df, 'Seqlen(k)', 'MFU', 'Method', ax, palette=color_mapping, hatches=line_mapping)
    ax.set_title('MFU by Seqlen and Method', fontsize=24)
    ax.set_xlabel('Seqlen (k)', fontsize=20)
    ax.set_ylabel('MFU', fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Plot Tokens/s
    ax = axs[1]
    barplot(df, 'Seqlen(k)', 'Tokens/s', 'Method', ax, palette=color_mapping, hatches=line_mapping)
    ax.set_title('Tokens/s by Seqlen and Method', fontsize=24)
    ax.set_xlabel('Seqlen (k)', fontsize=20)
    ax.set_ylabel('Tokens/s', fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Adjust layout
    plt.tight_layout()
    plt.show()

