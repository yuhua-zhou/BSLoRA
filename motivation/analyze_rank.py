from safetensors.torch import load_file
import numpy as np
import torch
from tqdm import tqdm

lora_dict = load_file(
    "../tune_log/qkvupdown/ablation/kaiming_init_share_lora/2_4_16/checkpoint-800/adapter_model.safetensors")

# print(lora_dict.keys())
#
# for key in lora_dict.keys():
#     _, _, _, _, layer_id, _, module, lora, share, _ = key.split(".")
#     print(layer_id, module, lora, share)

target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj", "mlp.down_proj"]
target_ranks = {module: [] for module in target_modules}
weight_names = [
    "sharelora_lora_A.local",
    "sharelora_lora_A.intra",
    "sharelora_lora_A.inter",
    "sharelora_lora_B.local",
    "sharelora_lora_B.intra",
    "sharelora_lora_B.inter",
    "sharelora_sampler.intra_sampler_A",
    "sharelora_sampler.intra_sampler_B",
    "sharelora_sampler.inter_sampler_A",
    "sharelora_sampler.inter_sampler_B"
]

for i in range(32):
    for module in tqdm(target_modules):
        weights = [
            lora_dict[f"base_model.model.model.layers.{i}.{module}.{name}.weight"] for name in weight_names
        ]
        # for name in weight_names:
        #     key = f"base_model.model.model.layers.{i}.{module}.{name}.weight"
        #     weight = lora_dict[key].numpy()
        #     print(key, weight.shape)

        delta_W_local = weights[0].T @ weights[3].T
        delta_W_intra = torch.kron(weights[1], weights[6]).T @ torch.kron(weights[4], weights[7]).T
        delta_W_inter = torch.kron(weights[2], weights[8]).T @ torch.kron(weights[5], weights[9]).T

        # print(delta_W_local.shape, delta_W_intra.shape, delta_W_inter.shape)

        delta_W = delta_W_local + delta_W_intra + delta_W_inter

        rank = np.linalg.matrix_rank(delta_W.numpy())
        # print(i, module, delta_W.shape, rank)

        target_ranks[module].append(rank)

print(target_ranks)