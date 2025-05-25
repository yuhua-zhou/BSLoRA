import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
import math
from utils import compute_matrix_entropy, entropy_similarity
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM

data = load_dataset("yahma/alpaca-cleaned")
prompt = data["train"][0]["instruction"]
model = LlamaForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
tokenizer = LlamaTokenizer.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
inputs = tokenizer.encode(prompt, return_tensors="pt")
dummy_input = model.get_input_embeddings()(inputs)
dummy_input = dummy_input.squeeze(0)
dummy_input = dummy_input.detach().numpy()

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


def layer_forward(layer, x):
    q = np.dot(x, layer["self_attn.q_proj"])
    k = np.dot(x, layer["self_attn.k_proj"])
    v = np.dot(x, layer["self_attn.v_proj"])

    attention_scores = np.dot(q, k.T) / np.sqrt(4096)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)

    x = np.dot(attention_weights, v)

    down = np.dot(x, layer["mlp.down_proj"])
    up = np.dot(down, layer["mlp.up_proj"])

    modules = [q, k, v, up, down]
    count = 0
    sum = 0
    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            similarity = entropy_similarity(modules[i], modules[j])
            count += 1
            sum += similarity
    similarity = sum / count

    # print(f"layer average similarity = {sum / count}")
    return similarity, up


layer_similarity = []
layer_matrix = []
for i in (range(len(parameters))):
    similarity, dummy_input = layer_forward(parameters[i], dummy_input)
    layer_similarity.append(similarity)
    layer_matrix.append(dummy_input)

    print(f"layer {i}: similarity={similarity}")

layer_similarity = np.array(layer_similarity)
np.save("layer_similarity_active.npy", layer_similarity)

heatmap = np.ones((32, 32))
for i in (range(len(layer_matrix))):
    for j in range(i + 1, len(layer_matrix)):
        similarity = entropy_similarity(
            layer_matrix[i],
            layer_matrix[j]
        )
        heatmap[i, j] = similarity
        heatmap[j, i] = similarity
        print(i, j, similarity)

np.save("heatmap_active.npy", heatmap)
