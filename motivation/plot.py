import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置全局默认字体大小
plt.rcParams['font.size'] = 16

def plot_layer_sim():
    # bar_data = np.load('layer_similarity_static.npy')
    bar_data = np.load('layer_similarity_active.npy')

    vmin = 0.60  # 0.95
    vmax = 1.0

    indices = [f"{i}" for i in range(len(bar_data))]
    plt.figure(figsize=(16, 8))  # 设置图像大小
    # plt.bar(indices, bar_data, color="#9DC3E6")
    # plt.bar(indices, bar_data)
    plt.bar(indices, bar_data, color="green",edgecolor='black', linewidth=1)
    plt.title('Average Similarity Within Layers', fontsize=25)
    plt.xlabel('Layer Index', fontsize=20)
    plt.ylabel('Layer Similarity', fontsize=20)
    plt.ylim(ymin=vmin, ymax=vmax)
    plt.xlim([-0.5, 32 - 0.5])
    plt.savefig("layer_similarity_active.svg")
    # plt.savefig("layer_similarity_static.svg")
    plt.show()


def plot_heatmap():
    # heatmap_data = np.load('heatmap_static.npy')
    heatmap_data = np.load('heatmap_active.npy')

    vmin = 0.20  # 0.95
    vmax = 1.0

    # 绘制heatmap
    plt.figure(figsize=(10, 8))  # 设置图像大小
    # heatmap = sns.heatmap(heatmap_data, cmap='viridis', vmin=vmin, vmax=vmax, cbar=True)  # cbar=True表示添加颜色标尺
    heatmap = sns.heatmap(heatmap_data, cmap='Blues', vmin=vmin, vmax=vmax, cbar=True)  # cbar=True表示添加颜色标尺

    plt.title("Similarity Across Layers", fontsize=25)
    plt.xlabel("Layer Index", fontsize=20)
    plt.ylabel("Layer Index", fontsize=20)
    plt.savefig("heatmap_active.svg")
    # plt.savefig("heatmap_static.svg")
    plt.show()


def plot_radar(data, legend_names, name):
    categories = ['openbookqa', 'arc_challenge', 'hellaswag', 'arc_easy', 'piqa', 'winogrande', 'boolq', 'siqa']

    min_values = [44, 44, 76, 75, 79, 69, 75, 46]
    max_values = [48, 48, 78, 78, 81, 71, 79, 50]

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
    ax.set_xticklabels(categories, fontsize=12)

    # Adjust label distance from the center
    ax.tick_params(pad=10)  # Increase the distance of axis labels from the plot

    # Add custom legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig(name)
    # Show the plot
    plt.show()


def plot_contribution():
    data = [
        [45.00, 45.05, 77.59, 77.02, 79.89, 70.01, 77.89, 48.46],
        [45.20, 47.53, 77.22, 76.64, 80.20, 70.32, 75.44, 47.54],
        [43.80, 46.16, 77.96, 75.97, 79.16, 70.32, 75.32, 46.32],
    ]
    legend_names = ['r=8,0,0', 'r=0,8,0', 'r=0,0,8']
    name = "contribution.svg"
    plot_radar(data, legend_names, name)


def plot_extension():
    data = [
        [45.40, 47.01, 77.35, 76.73, 80.03, 70.17, 78.50, 48.46],
        [45.20, 46.16, 77.36, 76.98, 80.20, 69.93, 78.20, 49.08],
        [45.60, 47.44, 77.29, 76.81, 80.09, 69.93, 77.89, 48.31],
        [44.60, 46.84, 77.34, 76.85, 79.98, 70.56, 77.80, 48.31]
    ]
    legend_names = ['r_1', 'r_2', '1_r', '2_r']
    name = "extension.svg"
    plot_radar(data, legend_names, name)


def plot_rank_bar():
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
    fig, ax = plt.subplots(figsize=(8, 6))


    for i in range(len(data)):
        bar = ax.bar(index + (i + 1) * bar_width, data[i], bar_width, label=groups[i], color=colors[i],
                     edgecolor='black', linewidth=1)

    # Add labels and title
    ax.set_xlabel('Shape Transformation Methods', fontsize=20)
    ax.set_ylabel('Rank Values', fontsize=20)
    ax.set_xticks(index + bar_width * 3)
    ax.set_xticklabels(categories, fontsize=15)

    # Add a legend
    ax.legend()

    plt.savefig("rank.svg")

    # Display the chart
    plt.show()


plot_rank_bar()
# plot_contribution()
# plot_extension()
# plot_layer_sim()
# plot_heatmap()
