# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import math
import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

class ShareLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("sharelora_lora_A", "sharelora_lora_B", "sharelora_sampler", "sharelora_gate")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, share_mode: str, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.share_mode = share_mode

        self.r = {}

        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})

        # share lora AB modules
        self.sharelora_lora_A = nn.ModuleDict({})
        self.sharelora_lora_B = nn.ModuleDict({})

        # ---------- kron mode ---------
        self.sharelora_sampler = nn.ModuleDict({})

        # ---------- gate mode ---------
        self.sharelora_gate = nn.ModuleDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        # currently we only consider the nn.Linear module
        base_layer = self.get_base_layer()
        in_features, out_features = base_layer.in_features, base_layer.out_features

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r_local, r_intra, r_inter, lora_alpha, lora_dropout, init_lora_weights,
                     intra_module, inter_module):
        # This code works for linear layers, override for other layer types
        if (r_local + r_intra + r_inter) <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r_local}")

        self.r[adapter_name] = {"local": r_local, "intra": r_intra, "inter": r_inter}
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # ------------ init lora module ------------

        # Actual trainable parameters
        self.sharelora_lora_A[adapter_name] = nn.ModuleDict({
            "local": nn.Linear(self.in_features, r_local, bias=False),
            "intra": intra_module["lora_A"],
            "inter": inter_module["lora_A"]
        })
        self.sharelora_lora_B[adapter_name] = nn.ModuleDict({
            "local": nn.Linear(r_local, self.out_features, bias=False),
            "intra": intra_module["lora_B"],
            "inter": inter_module["lora_B"]
        })

        # ------------ init kron ------------
        if self.share_mode == "kron":
            intra_kernel = (
                self.weight.shape[1] // intra_module["lora_A"].weight.shape[1],
                self.weight.shape[0] // intra_module["lora_B"].weight.shape[0],
            )
            inter_kernel = (
                self.weight.shape[1] // inter_module["lora_A"].weight.shape[1],
                self.weight.shape[0] // inter_module["lora_B"].weight.shape[0],
            )
            self.sharelora_sampler[adapter_name] = nn.ModuleDict({
                "intra_sampler_A": nn.Linear(intra_kernel[0], 1, bias=False),  # (256, r) -> (4096, 2*r)
                "intra_sampler_B": nn.Linear(1, intra_kernel[1], bias=False),  # (r, 256) -> (2*r, 4096)
                "inter_sampler_A": nn.Linear(inter_kernel[0], 1, bias=False),  # (256, r) -> (4096, 2*r)
                "inter_sampler_B": nn.Linear(1, inter_kernel[1], bias=False),  # (r, 256) -> (2*r, 4096)
            })

            # kaiming uniform init
            for key in self.sharelora_sampler[adapter_name].keys():
                nn.init.kaiming_uniform_(self.sharelora_sampler[adapter_name][key].weight, a=math.sqrt(5))

        # ------------ init gate ------------
        if self.share_mode == "gate":
            self.sharelora_gate[adapter_name] = nn.ModuleDict({
                # intra gate
                "intra_input_gate_down": nn.Linear(self.in_features, 1, bias=False),
                "intra_input_gate_up": nn.Linear(1, intra_module["lora_A"].in_features, bias=False),
                "intra_output_gate_down": nn.Linear(intra_module["lora_B"].out_features, 1, bias=False),
                "intra_output_gate_up": nn.Linear(1, self.out_features, bias=False),
                # inter gate
                "inter_input_gate_down": nn.Linear(self.in_features, 1, bias=False),
                "inter_input_gate_up": nn.Linear(1, intra_module["lora_A"].in_features, bias=False),
                "inter_output_gate_down": nn.Linear(intra_module["lora_B"].out_features, 1, bias=False),
                "inter_output_gate_up": nn.Linear(1, self.out_features, bias=False),
            })

            # kaiming uniform init
            for key in self.sharelora_gate[adapter_name].keys():
                nn.init.kaiming_uniform_(self.sharelora_gate[adapter_name][key].weight, a=math.sqrt(5))

        self.scaling[adapter_name] = 2  # lora_alpha / r_local

        self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.sharelora_lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                for key in self.sharelora_lora_A[adapter_name].keys():
                    nn.init.kaiming_uniform_(self.sharelora_lora_A[adapter_name][key].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                for key in self.sharelora_lora_A[adapter_name].keys():
                    nn.init.normal_(self.sharelora_lora_A[adapter_name][key].weight,
                                    std=1 / self.r[adapter_name]["local"])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")

            # zero initialization
            for key in self.sharelora_lora_B[adapter_name].keys():
                nn.init.zeros_(self.sharelora_lora_B[adapter_name][key].weight)

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]["local"]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.sharelora_lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.sharelora_lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]["local"]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.sharelora_lora_A.keys():
                continue

            lora_A = self.sharelora_lora_A[active_adapter]
            lora_B = self.sharelora_lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result

class Linear(nn.Module, ShareLoraLayer):
    # Lora implemented in a dense layer
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def __init__(
            self,
            base_layer,
            adapter_name: str,
            share_mode: str,
            r_local: int = 0,
            r_intra: int = 0,
            r_inter: int = 0,
            inter_module=None,
            intra_module=None,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            init_lora_weights: Union[bool, str] = True,
            **kwargs,
    ) -> None:
        super().__init__()
        ShareLoraLayer.__init__(self, base_layer, share_mode, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r_local=r_local,
            r_intra=r_intra,
            r_inter=r_inter,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            intra_module=intra_module,
            inter_module=inter_module
        )

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if self.share_mode == "kron":
                    self.kron_forward(active_adapter, x, result)
                elif self.share_mode == "gate":
                    self.gate_forward(active_adapter, x, result)
                elif self.share_mode == "slice":
                    self.slice_forward(active_adapter, x, result)

            result = result.to(torch_result_dtype)

        return result

    def kron_forward(self, active_adapter, x, result):

        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        sharelora_lora_A = self.sharelora_lora_A[active_adapter]
        sharelora_lora_B = self.sharelora_lora_B[active_adapter]

        sharelora_sampler = self.sharelora_sampler[active_adapter]

        for key in sharelora_lora_A.keys():
            lora_A = sharelora_lora_A[key]
            lora_B = sharelora_lora_B[key]

            sampler_name_A = f"{key}_sampler_A"
            sampler_name_B = f"{key}_sampler_B"

            x = x.to(lora_A.weight.dtype)

            if sampler_name_A not in sharelora_sampler.keys():
                result += lora_B(lora_A(dropout(x))) * scaling
            else:
                sampler_A = sharelora_sampler[sampler_name_A]
                sampler_B = sharelora_sampler[sampler_name_B]

                sampled_lora_A = torch.kron(sampler_A.weight, lora_A.weight)
                sampled_lora_B = torch.kron(sampler_B.weight, lora_B.weight)

                result += x @ sampled_lora_A.T @ sampled_lora_B.T * scaling

    def gate_forward(self, active_adapter, x, result):
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        sharelora_lora_A = self.sharelora_lora_A[active_adapter]
        sharelora_lora_B = self.sharelora_lora_B[active_adapter]
        gate = self.sharelora_gate[active_adapter]

        # ----------------- gate mode -----------------
        for key in sharelora_lora_A.keys():

            lora_A = sharelora_lora_A[key]
            lora_B = sharelora_lora_B[key]

            x = x.to(lora_A.weight.dtype)

            if f"{key}_input_gate_down" in gate.keys():
                input_gate_down = gate[f"{key}_input_gate_down"]
                input_gate_up = gate[f"{key}_input_gate_up"]
                output_gate_down = gate[f"{key}_output_gate_down"]
                output_gate_up = gate[f"{key}_output_gate_up"]

                share = dropout(x)
                share = input_gate_up(input_gate_down(share))
                share = lora_B(lora_A(share))
                share = output_gate_up(output_gate_down(share))
                result += share * scaling
            else:
                result += lora_B(lora_A(dropout(x))) * scaling

    def slice_forward(self, active_adapter, x, result):
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        sharelora_lora_A = self.sharelora_lora_A[active_adapter]
        sharelora_lora_B = self.sharelora_lora_B[active_adapter]

        # ----------------- slice mode -----------------
        for key in sharelora_lora_A.keys():
            lora_A = sharelora_lora_A[key]
            lora_B = sharelora_lora_B[key]

            weight_shape = (self.weight.shape[1], self.weight.shape[0])
            x = x.to(lora_A.weight.dtype)

            shape_A = lora_A.weight.T.shape
            shape_B = lora_B.weight.T.shape

            # center
            border_A = (shape_A[0] - weight_shape[0]) // 2
            border_B = (shape_B[1] - weight_shape[1]) // 2
            result += (
                              dropout(x) @ lora_A.weight.T[border_A:border_A + weight_shape[0], :] @
                              lora_B.weight.T[:, border_B
                                                 :border_B + weight_shape[1]]
                      ) * scaling

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep
