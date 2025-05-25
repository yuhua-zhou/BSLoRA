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

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class VBLoRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`VBLoRAConfig`].

    Paper: https://arxiv.org/abs/.

    Args:
        r (`int`, *optional*, defaults to `256`):
            VBLoRA parameter dimension ("rank").
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply Vera to. Only linear layers are supported.
        vblora_dropout (`float`):
            The dropout probability for Vera layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for Vera. Can be 'none', 'all' or 'vera_only'. If 'all' or 'vera_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from Vera layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the vblora layers with their default initialization. Don't change this
            setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the Vera transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the Vera
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=256, metadata={"help": "rank"})
    num_vectors: int = field(default=256, metadata={"help": ""})
    vector_length: int = field(default=256, metadata={"help": ""})
    topk: int = field(default=2, metadata={"help": "topk"})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with VBLoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    save_topk_logits: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only save the topk logits"
            )
        },
    )
    vblora_dropout: float = field(default=0.0, metadata={"help": "VBLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Vera. Can be 'none', 'all' or 'vera_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from VBLoRA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the VBLoRA layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.VBLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

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

import math
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)


from .config import VBLoRAConfig
from .layer import Linear, VBLoRALayer


class VBLoRAModel(BaseTuner):
    """
    Creates VBLoRA model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`VeraConfig`]): The configuration of the Vera model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Vera model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import VeraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = VeraConfig(r=128)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`VeraConfig`]): The configuration of the Vera model.
    """

    prefix: str = "vblora"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _init_vblora_vector_bank(self, config: VBLoRAConfig, adapter_name: str) -> None:
        self.vblora_vector_bank = nn.ParameterDict({})
        vblora_vector_bank = torch.zeros(config.num_vectors, config.vector_length)
        torch.nn.init.uniform_(vblora_vector_bank, -0.02, 0.02)
        self.vblora_vector_bank[adapter_name] = vblora_vector_bank

    def _pre_injection_hook(self, model: nn.Module, config: VBLoRAConfig, adapter_name: str) -> None:
        self._init_vblora_vector_bank(config, adapter_name)

    def _check_new_adapter_config(self, config: VBLoRAConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(vera_config, key):
        return check_target_module_exists(vera_config, key)

    def _create_and_replace(
        self,
        vera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = vera_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "vera_dropout": vera_config.vblora_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_weights": vera_config.init_weights,
        }
        kwargs["bias"] = bias
        # TODO: add quantization support

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                vblora_vector_bank=self.vblora_vector_bank,
                r=r,
                topk=vera_config.topk,
                num_vectors=vera_config.num_vectors,
                vector_length=vera_config.vector_length,
                vblora_dropout=vera_config.vblora_dropout,
                init_weights=vera_config.init_weights,
            )
        else:
            new_module = self._create_new_module(vera_config, self.vblora_vector_bank, vera_config.num_vectors,
                                                 vera_config.vector_length, vera_config.topk, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "vblora_" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vera_only":
                for m in model.modules():
                    if isinstance(m, VBLoRALayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(vera_config, vblora_vector_bank, num_vectors, vector_length, topk, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        r = kwargs.pop("r")
        new_module = Linear(
            target,
            vblora_vector_bank,
            adapter_name,
            r,
            num_vectors,
            vector_length,
            topk,
            bias=bias,
            **kwargs,
        )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, VBLoRALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        # we cannot use self.prefix as we want to include non-trainable vera parameters
        key_list = [key for key, _ in self.model.named_modules() if "vblora" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)

                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        # we cannot use self.prefix as we want to include non-trainable vera parameters
        key_list = [key for key, _ in self.model.named_modules() if "vblora" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, VBLoRALayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []

    # TODO: modify this method
    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        r"""
        This method merges the Vera layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        """
        Gets back the base model by removing all the VBLoRA modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def topk_to_logits(self, topk_indices_values):
        logits = torch.ones((topk_indicex_valuies.shape[0], topk_indicex_valuies.shape[1], self.config.num_vectors)) * float("-inf")
        indices, values = topk_indices_valuies
        logits.scatter_(1, indices, values)


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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class VBLoRALayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("vblora_logits_A", "vblora_logits_B", "vblora_vector_bank")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.topk = {}
        self.vblora_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.vblora_logits_A = nn.ParameterDict({})
        self.vblora_logits_B = nn.ParameterDict({})
        self.vblora_vector_bank = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
            self,
            adapter_name,
            vblora_vector_bank,
            r,
            topk,
            num_vectors,
            vector_length,
            vblora_dropout,
            init_weights,
    ):
        if r <= 0:
            raise ValueError(f"`r` {r} should be a positive integer value")
        if topk <= 0:
            raise ValueError(f"`topk` {topk} should be a positive integer value")

        if self.in_features % vector_length != 0:
            raise ValueError(f'`in_features` {self.in_features} must be divisible by `vector_length` {vector_length}')
        if self.out_features % vector_length != 0:
            raise ValueError(f'`out_features` {self.out_features} must be divisible by `vector_length` {vector_length}')

        self.r[adapter_name] = r
        self.topk[adapter_name] = topk
        if vblora_dropout > 0.0:
            vblora_dropout_layer = nn.Dropout(p=vblora_dropout)
        else:
            vblora_dropout_layer = nn.Identity()

        self.vblora_dropout.update(nn.ModuleDict({adapter_name: vblora_dropout_layer}))
        self.vblora_logits_A[adapter_name] = nn.Parameter(
            torch.zeros(self.in_features // vector_length, r, num_vectors), requires_grad=True)
        self.vblora_logits_B[adapter_name] = nn.Parameter(
            torch.zeros(self.out_features // vector_length, r, num_vectors), requires_grad=True)
        self.vblora_vector_bank = vblora_vector_bank

        if adapter_name not in vblora_vector_bank:
            # This means that this is not the first VeRA adapter. We have to add an entry in the dict for this adapter.
            if len(self.vblora_vector_bank) < 1:
                raise ValueError(
                    "The `vector bank` is empty. This should not happen. Please report this issue."
                )
            # we can take any of the existing adapter's parameters, as they should all be identical
            vblora_vector_bank_param = list(self.vblora_vector_bank.values())[0]
            self.vblora_vector_bank[adapter_name] = vblora_vector_bank_param

        if init_weights:
            self.reset_vblora_parameters(adapter_name)

        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)

        self.set_adapter(self.active_adapters)

    def reset_vblora_parameters(self, adapter_name):
        if adapter_name in self.vblora_logits_A.keys():
            with torch.no_grad():
                nn.init.normal_(self.vblora_logits_A[adapter_name], 0, 0.01)
                nn.init.normal_(self.vblora_logits_B[adapter_name], 0, 0.01)


class Linear(nn.Linear, VBLoRALayer):
    # Vera implemented in a dense layer
    def __init__(
            self,
            base_layer,
            vblora_vector_bank,
            adapter_name: str,
            r: int,
            num_vectors: int,
            vector_length: int,
            topk: int = 2,
            vblora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            is_target_conv_1d_layer: bool = False,
            init_weights: bool = True,
            **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        VBLoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, vblora_vector_bank, r, topk, num_vectors, vector_length, vblora_dropout,
                          init_weights, )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.vblora_logits_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vblora_logits_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    @staticmethod
    def _get_low_rank_matrix(logits, vblora_vector_bank, topk):
        top_k_logits, indices = logits.topk(topk, dim=-1)
        topk_weights = F.softmax(top_k_logits, dim=-1)
        return (topk_weights.unsqueeze(-1) * vblora_vector_bank[indices]).sum(-2)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        vblora_logits_A = self.vblora_logits_A[adapter]
        vblora_logits_B = self.vblora_logits_B[adapter]
        vblora_vector_bank = self.vblora_vector_bank[adapter]

        device = vblora_logits_A.device
        dtype = vblora_logits_A.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        if cast_to_fp32:
            vblora_logits_A = vblora_logits_A.float()
            vblora_logits_B = vblora_logits_B.float()
            vblora_vector_bank = vblora_vector_bank.float()

        A = Linear._get_low_rank_matrix(vblora_logits_A, vblora_vector_bank, self.topk).transpose(1, 2).reshape(-1,
                                                                                                                vblora_logits_A.shape[
                                                                                                                    1])
        B = Linear._get_low_rank_matrix(vblora_logits_B, vblora_vector_bank, self.topk).transpose(0, 1).reshape(
            vblora_logits_B.shape[1], -1)
        output_tensor = transpose(A @ B, self.fan_in_fan_out)
        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.vblora_logits_A.keys():
                    continue

                vblora_logits_A = self.vblora_logits_A[active_adapter]
                vblora_logits_B = self.vblora_logits_B[active_adapter]
                vblora_vector_bank = self.vblora_vector_bank[active_adapter]
                topk = self.topk[active_adapter]
                dropout = self.vblora_dropout[active_adapter]
                x = x.to(vblora_logits_A.dtype)
                # (tile, rank, vector_length) -> (tile, vector_length, rank) -> (tile x vector_length, rank)
                A = Linear._get_low_rank_matrix(vblora_logits_A, vblora_vector_bank, topk).transpose(1, 2).reshape(-1,
                                                                                                                   vblora_logits_A.shape[
                                                                                                                       1])
                # (tile, rank, vector_length) -> (rank, tile, vector_length) -> (rank, tile x vector_length)
                B = Linear._get_low_rank_matrix(vblora_logits_B, vblora_vector_bank, topk).transpose(0, 1).reshape(
                    vblora_logits_B.shape[1], -1)

                result = result + dropout(x) @ A @ B

        result = result.to(previous_dtype)
        return result