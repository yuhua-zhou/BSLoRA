import json

import matplotlib.pyplot as plt
import numpy as np


def plot_radar():
    plt.rcParams.update({'font.size': 15})  # 设置默认字体大小为12
    # Data and categories
    data = [
        [45.40, 48.46, 77.35, 76.73, 80.03, 70.17, 78.50, 47.01],
        [45.20, 49.08, 77.36, 76.98, 80.20, 69.93, 78.20, 46.16],
        [45.60, 48.31, 77.29, 76.81, 80.09, 69.93, 77.89, 47.44],
        [44.60, 48.31, 77.34, 76.85, 79.98, 70.56, 77.80, 46.84]
    ]

    # data = [
    #     [45.00, 48.46, 77.59, 77.02, 79.89, 70.01, 77.89, 45.05],
    #     [45.20, 47.54, 77.22, 76.64, 80.20, 70.32, 75.44, 47.53],
    #     [43.80, 46.32, 77.96, 75.97, 79.16, 70.32, 75.32, 46.16],
    # ]

    categories = ['openbookqa', 'siqa', 'hellaswag', 'arc_easy', 'piqa', 'winogrande', 'boolq', 'arc_challenge']

    min_values = [44, 46, 76, 75, 79, 69, 75, 44]
    max_values = [48, 50, 78, 78, 81, 71, 79, 48]

    # legend_names = ['r=8,0,0', 'r=0,8,0', 'r=0,0,8']
    legend_names = ['r_1', 'r_2', '1_r', '2_r']
    num_vars = len(categories)

    # Normalize the data to a [0, 1] range based on custom min/max values
    normalized_data = (np.array(data) - min_values) / (np.array(max_values) - np.array(min_values))

    # Create a radar chart
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Add data for each row
    for i, row in enumerate(normalized_data):
        row_data = np.append(row, row[0])  # Append first value to end to close the loop
        ax.plot(angles, row_data, linewidth=2, linestyle='solid', label=legend_names[i])
        ax.fill(angles, row_data, alpha=0.25)

    # Add category labels
    ax.set_yticklabels([])  # Remove radial labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=18)

    # Adjust label distance from the center
    ax.tick_params(pad=10)  # Increase the distance of axis labels from the plot

    # Add custom legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15))

    # 导出为 SVG
    plt.savefig('radar.svg', format='svg')

    # Show the plot
    plt.show()


def plot_bar():
    plt.rcParams.update({'font.size': 25})  # 设置默认字体大小为12

    # Data for the groups
    categories = ['SS', 'GT', 'KE']
    groups = ["△Wq", "△Wk", "△Wv", "△Wup", "△Wdown"]
    data = [
        [22, 4, 21.1875],
        [22, 4, 21.09375],
        [22, 4, 21.21875],
        [22, 4, 21.4375],
        [22, 4, 21.34375]
    ]

    colors = ["#DEECF9", "#F7CCAD", "#A3D4D5", "#D2E1D4", "#9DC3E6"]

    # Bar width
    bar_width = 0.18

    # The x positions for the bars
    index = np.arange(len(categories))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(len(data)):
        bar = ax.bar(index + (i + 1) * bar_width, data[i], bar_width, label=groups[i], color=colors[i],
                     edgecolor='black', linewidth=1)

    # Add labels and title
    ax.set_xlabel('Shape Transformation Methods', fontsize=25, labelpad=10)
    ax.set_ylabel('Rank Values', fontsize=25)
    ax.set_xticks(index + bar_width * 3)
    ax.set_xticklabels(categories, rotation=0, fontsize=20)
    # 调整 x 轴标签和坐标轴的距离
    ax.tick_params(axis='x', which='major', pad=10)  # pad 是标签和轴线之间的距离

    # Add a legend
    ax.legend()

    # 导出为 SVG
    plt.savefig('bar.svg', format='svg')

    # Display the chart
    plt.show()


def plot_bar2():
    plt.rcParams.update({'font.size': 20})  # 设置默认字体大小为12

    labels = ['r_1', 'r_2', '1_r', '2_r']
    data = [65.39, 65.46, 65.28, 65.42]
    colors = ["#F7CCAD", "#F7CCAD", "#D2E1D4", "#D2E1D4"]

    x = np.arange(len(labels))  # 标签位置
    width = 0.60

    plt.figure(figsize=(8, 5))

    for i in range(len(labels)):
        plt.bar(x[i], data[i], width, color=colors[i], edgecolor='black', linewidth=1, zorder=2)

    plt.axvline(x=1.5, color='grey', linewidth=2, linestyle='--', zorder=0)

    plt.xticks(x, labels)
    plt.ylabel('Average Performance')
    plt.ylim(65, 65.5)
    plt.grid(alpha=0.3)

    # # 导出为 SVG
    plt.savefig('bar2.svg', format='svg')

    plt.show()


def plot_memory():
    plt.rcParams.update({'font.size': 15})  # 设置默认字体大小为12

    # Simple data for the line chart
    x = range(100, 1600, 100)
    lora = [
        18264.150390625,
        23614.150390625,
        28964.150390625,
        34314.150390625,
        39664.150390625,
        45014.150390625,
        50364.150390625,
        55714.150390625,
        61064.150390625,
        66414.150390625,
        71764.150390625,
        77114.150390625,
        81920.000000000,
        81920.000000000,
        81920.000000000,
    ]

    share = [
        14618.353515625,
        16362.103515625,
        18105.853515625,
        19849.603515625,
        21593.353515625,
        23337.103515625,
        25080.853515625,
        26824.603515625,
        28568.353515625,
        30312.103515625,
        32055.853515625,
        33799.603515625,
        35543.353515625,
        37287.103515625,
        39030.853515625,
    ]

    # Plot the line chart
    plt.figure(figsize=(8, 5))
    # oom
    plt.scatter([1300, 1400, 1500], [81920, 81920, 81920], color='red', marker='x', s=200, label='Out Of Memory',
                linewidths=2.5, zorder=2)

    plt.plot(x, lora, marker='o', label='LoRA', linestyle='-', color='blue', zorder=1)
    plt.plot(x, share, marker='s', label='Bi-Share LoRA', linestyle='--', color='orange', zorder=1)

    plt.xlabel("Number of Serving LoRA")
    plt.ylabel("Memory (MB)")
    plt.title("Memory Usage In Multi-lora Serving")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.ylim(0, 83920)  # Set the y-axis limit to 81920
    plt.yticks(range(0, 83921, 10000))  # Set y-axis ticks with an interval of 10000
    plt.tight_layout()

    # 导出为 SVG
    plt.savefig('serving.pdf', format='pdf')

    # Display the plot
    plt.show()


def plot_memory_32g():
    plt.rcParams.update({'font.size': 15})  # 设置默认字体大小为12

    loras = [
        {
            "weight_path": "/mnt/user-576454879397019648/vol-7898655345270267904/zyh/ShareLoRA/rebuttal/tune_log/qkvupdown/standard_lora/8/result.json",
            "label": "LoRA",
            "marker": "o",
            "line_style": "-"
        },
        {
            "weight_path": "/mnt/user-576454879397019648/vol-7898655345270267904/zyh/ShareLoRA/rebuttal/tune_log/qkvupdown/share_kron/2_4_8/result.json",
            "label": "BSLoRA(KE)",
            "marker": "s",
            "line_style": "--"
        }
    ]

    max_len = 11  # 1095

    # Simple data for the line chart
    x = range(0, max_len * 100 + 5, 100)

    # Plot the line chart
    plt.figure(figsize=(8, 5))

    # oom
    plt.scatter([400, 500, 600, 700, 800, 900, 1000, 1100], [32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768],
                color='red',
                marker='x', s=200, label='Out Of Memory',
                linewidths=2.5, zorder=2)

    for item in loras:
        memory_list = json.load(open(item["weight_path"], "r"))
        memory_list = list(memory_list["stage_memory"].values())
        memory_list = [memory_list[i] for i in range(len(memory_list)) if i % 10 == 0]

        memory_list = memory_list + [32768] * (max_len - len(memory_list) + 1)

        plt.plot(x, memory_list, marker=item["marker"], label=item["label"], linestyle=item["line_style"], zorder=1)

    plt.xlabel("Number of Serving LoRA")
    plt.ylabel("Memory (MB)")
    plt.title("Memory Usage In Multi-lora Serving")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.ylim(10000, 32768 + 1000)  # Set the y-axis limit to 32768
    plt.yticks(range(10000, 32768 + 1000, 3000))  # Set y-axis ticks with an interval of 10000
    plt.tight_layout()

    # 导出为 SVG
    plt.savefig('serving.pdf', format='pdf')

    # Display the plot
    plt.show()


def plot_speed():
    plt.rcParams.update({'font.size': 25})  # 设置默认字体大小为12

    # Data for the groups
    categories = ['LoRA', 'VeRA', 'VB-LoRA', 'Bi-Share LoRA']
    data = [25, 30, 40, 15]

    # Plot the vertical bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(categories, data, color='lightcoral', edgecolor='black', linewidth=2)
    plt.xlabel("Categories")
    plt.ylabel("Time Cost")
    plt.title("Simple Vertical Bar Chart")
    plt.grid(linestyle='--', alpha=0.7)

    # plt.grid()
    plt.tight_layout()

    # 导出为 SVG
    plt.savefig('speed.svg', format='svg')

    # Display the plot
    plt.show()


# plot_bar()
plot_bar2()
# plot_radar()
# plot_speed()
# plot_memory()
# plot_memory_32g()
