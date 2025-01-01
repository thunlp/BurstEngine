import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
import os
import logging
import sys
import matplotlib

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(levelname)s - %(message)s",
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 设置默认参数
TITLE_SIZE = 11
LABEL_SIZE = 10
LEGEND_SIZE = 7
TICK_SIZE = 8

# 设置图表风格参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def format_seq_length(length):
    """将序列长度格式化为更易读的形式（如32768 -> 32K）"""
    if length >= 1024:
        return f"{length//1024}K"
    else:
        return str(length)

def plot_benchmark_data(csv_file="exp_data.csv", save_path="plots", 
                       show_plot=True, paper_ready=True, output_format="pdf"):
    """
    绘制benchmark数据的分组柱状图
    
    Args:
        csv_file: CSV数据文件路径
        save_path: 保存图片的路径
        show_plot: 是否显示图片
        paper_ready: 是否使用适合论文的高质量设置
        output_format: 输出文件格式（pdf或png）
    """
    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    
    # 读取CSV文件
    logger.info(f"读取数据文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 分离单节点和多节点数据（如果有相关列）
    if 'Multi-node' in df.columns:
        single_node_df = df[df['Multi-node'] == False].copy()
        multi_node_df = df[df['Multi-node'] == True].copy()
    else:
        single_node_df = df.copy()
        multi_node_df = pd.DataFrame()
    
    # 绘制单节点数据
    if not single_node_df.empty:
        file_name = os.path.join(save_path, f"single_node_benchmark.{output_format}")
        logger.info(f"绘制单节点数据图表，保存至: {file_name}")
        plot_grouped_bar(single_node_df, "Distributed Attention Performance on 32xA800", 
                         file_name, show_plot, paper_ready)
    
    # 绘制多节点数据
    if not multi_node_df.empty:
        file_name = os.path.join(save_path, f"kernel_benchmark.{output_format}")
        logger.info(f"绘制多节点数据图表，保存至: {file_name}")
        plot_grouped_bar(multi_node_df, "Distributed Attention \non 32xA800", 
                         file_name, show_plot, paper_ready)

def plot_grouped_bar(df, title, save_path, show_plot=True, paper_ready=True):
    """
    按序列长度分组绘制柱状图
    
    Args:
        df: 包含数据的DataFrame
        title: 图表标题
        save_path: 保存路径
        show_plot: 是否显示图像
        paper_ready: 是否使用适合论文的高质量设置
    """
    # 获取唯一的序列长度和实现方式
    seq_lengths = sorted(df['Seq Len'].unique(), reverse=True)
    implementations = sorted(df['Implementation'].unique())
    score_map = ["Mega","Double", "USP", "Burst"]
    def get_score(name):
        for i, s in enumerate(score_map):
            if s in name:
                return i
        return -1
    implementations = sorted(implementations, key=get_score, reverse=False)  # 按得分顺序排序
    
    # 配置图表大小和风格
    sns.set_style("whitegrid")
    fig_width = 4 if paper_ready else 12
    fig_height = 4.4 if paper_ready else 9
    plt.figure(figsize=(fig_height, fig_width), dpi=300 if paper_ready else 100)  # Swapped dimensions
    
    # 定义柱状图布局参数
    group_width = 0.8
    bar_width = group_width / len(implementations)
    group_positions = np.arange(len(seq_lengths))
    
    # 设置图表底色
    ax = plt.gca()
    ax.set_facecolor('#f8f8f8')
    
    # 创建颜色映射 - 高对比度且具有视觉吸引力的颜色
    colors = ['#ff9da7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', 
              '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']
    hatches = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '***', '---', '|||']
    
    # 确保颜色和实现方式一一对应
    color_map = {impl: colors[i % len(colors)] for i, impl in enumerate(implementations)}
    hatch_map = {impl: hatches[i % len(hatches)] for i, impl in enumerate(implementations)}
    
    # 用于标记字母的偏移量
    offset = -3
    abc_str = ''.join(reversed("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(implementations)]))
    
    # 为每个实现方式创建一个柱状图组
    for _i, impl in enumerate(implementations):
        i = len(implementations) - _i - 1  # 反向索引以保持顺序
        print(i)

        # 计算当前实现方式的柱子位置
        positions = group_positions - (group_width/2) + (i + 0.5) * bar_width
        
        # 提取性能数据
        perf_data = []
        for seq in seq_lengths:
            subset = df[(df['Seq Len'] == seq) & (df['Implementation'] == impl)]
            if not subset.empty:
                perf_data.append(subset['TFLOPs/s'].values[0])
            else:
                # 对于缺失的数据，填充0
                perf_data.append(0)
        
        # 绘制柱状图（水平方向）
        print(impl,color_map[impl])
        bars = plt.barh(positions, perf_data, height=bar_width, 
                      label=f"({abc_str[i]}) {impl}", 
                      color=color_map[impl], 
                      hatch=hatch_map[impl],
                      edgecolor='black', 
                      linewidth=0.8)
        
        # 在每个柱子上添加数值标签
        for j, v in enumerate(perf_data):
            if v > 0:  # 只为非零值添加标签
                # 根据数值大小调整标签位置
                if v < max(perf_data) * 0.15:  # 较小值放在柱子右侧
                    plt.text(v + max(perf_data) * 0.02, positions[j], 
                            f"{v:.1f}", ha='left', va='center', 
                            fontsize=TICK_SIZE - 1 if paper_ready else TICK_SIZE,
                            fontweight='bold')
                else:  # 较大值放在柱子内部
                    text = plt.text(v * 0.5, positions[j], 
                                   f"{v:.1f}", ha='center', va='center',
                                   fontsize=TICK_SIZE - 1 if paper_ready else TICK_SIZE,
                                   fontweight='bold', color='white')
                    # 添加文字轮廓以增强可读性
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=2, foreground='black'),
                        path_effects.Normal()
                    ])
        
        # 在柱子左侧添加标记字母
        for j in range(len(seq_lengths)):
            plt.text(offset, positions[j], abc_str[i], 
                    ha='center', va='center', 
                    fontsize=TICK_SIZE, fontweight='bold')
    
    # 添加标记 X 表示缺失或无法运行的情况
    for i, seq in enumerate(seq_lengths):
        for j, impl in enumerate(implementations):
            subset = df[(df['Seq Len'] == seq) & (df['Implementation'] == impl)]
            if subset.empty or (not subset.empty and subset['TFLOPs/s'].values[0] == 0):

                _j = len(implementations) - j - 1  # 反向索引以保持顺序
                pos = group_positions[i] - (group_width/2) + (_j + 0.5) * bar_width
                plt.text(4, pos, "X", ha='center', va='center', 
                        fontsize=12, fontweight='bold', color=color_map[impl],
                        path_effects=[
                            path_effects.Stroke(linewidth=2, foreground='black'),
                            path_effects.Normal()
                        ])
    
    # 配置Y轴（原来的X轴）
    plt.yticks(group_positions, [format_seq_length(seq) for seq in seq_lengths], 
              fontsize=TICK_SIZE, fontweight='bold')
    plt.ylabel('Sequence Length', fontsize=LABEL_SIZE, fontweight='bold', labelpad=5)
    
    # 配置X轴（原来的Y轴）
    plt.xlabel('Performance (TFLOPs/s)', fontsize=LABEL_SIZE, fontweight='bold', labelpad=5, rotation=0)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE, rotation=45)
    
    # 自动计算X轴上限以确保有足够的空间
    x_max = df['TFLOPs/s'].max() * 1.15
    plt.xlim(0, x_max)
    
    # 添加标题
    plt.title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=5)
    
    # 添加图例
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), 
              ncol=min(2, len(implementations)), 
              fontsize=LEGEND_SIZE, frameon=True,
              facecolor='white', edgecolor='lightgray', 
              framealpha=0.9)
    
    # 添加网格线
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 确保布局紧凑且美观
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"图表已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    import argparse
    csv_path = "./kernel_perf.csv"
    output = "/Users/tachicoma/Docs/tex/BurstEngine/sc_2025/samples/exp_images/"
    no_show = True
    paper_ready = True
    format = "pdf"
    plot_benchmark_data(csv_path, output, not no_show, paper_ready, format)

if __name__ == "__main__":
    main()

