import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
import math
from utils import compute_matrix_entropy, entropy_similarity

# 读取lora矩阵
lora_dict = torch.load("../tune_log/qkvupdown/standard_lora_r64/adapter_model.bin", map_location='cpu')
# print(lora_dict.keys())

target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj", "mlp.down_proj"]

# 构建成参数矩阵
parameters = [{module: 0 for module in target_modules} for i in range(32)]
for i in range(32):
    for module in target_modules:
        lora_A_key = f"base_model.model.model.layers.{i}.{module}.lora_A.weight"
        lora_B_key = f"base_model.model.model.layers.{i}.{module}.lora_B.weight"
        lora_A = lora_dict[lora_A_key]
        lora_B = lora_dict[lora_B_key]

        delta_W = lora_B @ lora_A
        parameters[i][module] = delta_W


# print(parameters)

def compute_intra_similarity(parameters):
    layer_similarity = []
    # 计算每一层的相似度
    for layer in tqdm(parameters):
        # for module in target_modules:
        #     entropy = compute_matrix_entropy(layer[module].numpy())
        #     print(module, entropy)

        count = 0
        sum = 0
        for i in range(len(target_modules)):
            for j in range(i + 1, len(target_modules)):
                similarity = entropy_similarity(
                    layer[target_modules[i]].numpy(),
                    layer[target_modules[j]].numpy(),
                )
                # print(f"{target_modules[i]},{target_modules[j]}", similarity)
                count += 1
                sum += similarity

        # print(f"layer average similarity = {sum / count}")
        layer_similarity.append(sum / count)

    layer_similarity = np.array(layer_similarity)
    np.save("layer_similarity_static.npy", layer_similarity)
    return layer_similarity


def compute_inter_similarity(parameters):
    layer_matrix = []
    # 计算每一层的相似度
    for layer in parameters:
        layer = np.concatenate([
            value.numpy().flatten() for value in layer.values()
        ])
        layer_matrix.append(layer)

    heatmap = np.ones((32, 32))
    for i in tqdm(range(len(layer_matrix))):
        for j in range(i + 1, len(layer_matrix)):
            similarity = entropy_similarity(
                layer_matrix[i],
                layer_matrix[j]
            )
            heatmap[i, j] = similarity
            heatmap[j, i] = similarity

    np.save("heatmap_static.npy", heatmap)
    return heatmap


# layer_similarity = compute_intra_similarity(parameters)
# heatmap = compute_inter_similarity(parameters)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

vmin = 0.95
vmax = 1.0

data = np.load("layer_similarity_static.npy")
layer_indices = [f"{i}" for i in range(data.shape[0])]

plt.figure(figsize=(16, 8))
plt.title("Average Similarity within Layers", fontsize=40)
plt.bar(layer_indices, data)
plt.ylim(ymin=vmin, ymax=vmax)
plt.xlabel("Layer Index", fontsize=32)
plt.ylabel("Layer Similarity", fontsize=32)
plt.tight_layout()  # 自动调整布局
# 调整x轴范围，移除空白
plt.xlim([-0.5, len(layer_indices) - 0.5])
plt.savefig("../figures/layer_similarity_static.svg")
plt.show()

data = np.load("heatmap_static.npy")  # 33x33 matrix to represent 33 transformer layers

# Create a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(data, cmap="plasma", vmin=vmin, vmax=vmax, square=True, cbar=True)

# Set axis labels and title
plt.title("Similarity across Layers", fontsize=20)
plt.xlabel("Layer Index", fontsize=16)
plt.ylabel("Layer Index", fontsize=16)

# Show the plot
plt.tight_layout()  # 自动调整布局
plt.savefig("../figures/heatmap_static.svg")
plt.show()
