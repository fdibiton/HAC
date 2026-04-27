#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger


class LinearWarmupCosineDecayLR(LambdaLR):
    """
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer.
            total_steps: Total epochs (or iterations) for training.
            warmup_steps: Number of first few steps to do linear warmup.
            last_epoch: The index of last step (epoch or iteration). We named
                it `last_epoch` instead of `last_step` to keep the naming
                consistent with other LR schedulers in PyTorch.
        """
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)
    
        
def get_num_layer_for_transformer(var_name, num_max_layer):
    # delete all before "model."
    var_name = var_name.split("model.")[-1]
    # Visual layer format: blocks.{layer_num}.xxx
    # Textual layer format: resblocks.{layer_num}.xxx
    if var_name.startswith(("blocks", "resblocks")):
        layer_id = int(var_name.split('.')[1])
        num_layer = layer_id + 1 # 1 to 12
    elif var_name.startswith(("norm", "ln_final")):
        num_layer = num_max_layer + 1 # 13
    elif var_name.startswith(("visual_proj", "textual_proj")):
        num_layer = num_max_layer + 1 # 13
    elif var_name in ("logit_scale", "curv", "visual_alpha", "textual_alpha"):
        num_layer = num_max_layer + 1 # 13
    else:
        num_layer = 0
        
    return num_layer


def set_weight_decay_per_param(
    model: torch.nn.Module,
    weight_decay: float,
    gain_bias_decay: float | None = None,
    exclude_params: list[str] = [],
    other_params: list[str] = [], # ["logit_scale", "visual_alpha", "textual_alpha", "curv"]
    other_lr: float | None = None,
    other_decay: float | None = None,
    layer_decay: float | None = None,
    layer_decay_max_lr: float | None = None,
    num_layers: int = 12,
) -> list[dict]:
    """
    Set weight decay for trainable parameters of a model. This function allows
    setting different weight decay for normalization layers from rest of the
    model. The output param groups can be used to instantiate an optimizer.

    This function is adapted from the Torchvision ImageNet training script.

    Args:
        model: PyTorch module with trainable parameters.
        weight_decay: Weight decay for all params except normalization layers.
        gain_bias_decay: Weight decay for normalization layers and bias parameters
            everywhere in the model. If `None`, it defaults to `weight_decay`.
        exclude_params: List of parameter names whose weight decay should be zero.
            For example, this could be learnable softmax temperature parameter.
    """
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    gain_bias_decay = gain_bias_decay or weight_decay
    params = {"regular": [], "gain_bias": [], "excluded": [], "other": []}
    params_weight_decay = {
        "regular": weight_decay,
        "gain_bias": gain_bias_decay,
        "excluded": 0.0,
        "other": other_decay if other_decay is not None else weight_decay,
    }

    # Hold references to parameters (tensors) in this set to avoid adding
    # duplicates, because some modules have shared weights (word embeddings)
    # and they may get counted twice -- PyTorch does not like it.
    already_added_parameters = set()
    other_names = []
    
    if layer_decay is not None:
        # create 13 decay values for 12 layers + 1 (final_ln + projector + hyperbolic params)
        layer_decay_values = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in already_added_parameters:
                continue

            # Record current parameter as "visited".
            already_added_parameters.add(p)
            full_name = f"{prefix}.{name}" if prefix else name
            if any([exclude_name in name for exclude_name in exclude_params]):
            # if any([exclude_name in full_name for exclude_name in exclude_params]):
                # Check the exclude substrings in parameter name.
                params["excluded"].append(p)
            elif any([other_name in full_name for other_name in other_params]):
                # Check the projection substrings in parameter name.
                params["other"].append(p)
                other_names.append(full_name)
            elif isinstance(module, norm_classes) or "bias" in name:
                # Check the module type or `bias` in parameter name, this matching
                # is sufficient for ResNet-like and Transformer modules of PyTorch.
                params["gain_bias"].append(p)
            else:
                if layer_decay is not None:
                    layer_id = get_num_layer_for_transformer(full_name, num_layers)
                    this_decay = layer_decay_values[layer_id]
                    # print(full_name, layer_id, this_decay)
                    if this_decay != 1.0:
                        logger.info(f"Setting layer-wise lr decay for {full_name}: lr * {this_decay:.4f}")
                    group_name = f"layer_{layer_id}"
                    if group_name not in params:
                        params[group_name] = []
                        params_weight_decay[group_name] = params_weight_decay["regular"]
                    params[group_name].append(p)
                else:
                    params["regular"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)
    # Force deterministic param group ordering
    group_order = ["regular", "gain_bias", "excluded", "other"]
    if layer_decay is not None:
        for i in range(num_layers + 2):
            group_name = f"layer_{i}"
            if group_name in params:
                group_order.append(group_name)

    param_groups = []
    for key in group_order:
        if len(params[key]) > 0:
            group = {
                "params": params[key],
                "weight_decay": params_weight_decay[key],
                "name": key,
            }
            if key == "other" and other_lr is not None:
                group["lr"] = other_lr
            elif key.startswith("layer_") and layer_decay is not None:
                layer_id = int(key.split('_')[1])
                group["lr"] = layer_decay_values[layer_id] * layer_decay_max_lr

            param_groups.append(group)
    if len(params["other"]) > 0:
        logger.info("Added {} parameters with a different lr: {}".format(
            other_names, other_lr
        ))
    if layer_decay is not None:
        logger.info(f"Using layer-wise learning rate decay with layer_decay: {layer_decay}, num_layers: {num_layers}")
    # print param groups in order
    for group in param_groups:
        logger.info(f"Param group: {group['name']}, weight_decay={group['weight_decay']}, lr={group.get('lr', 'default')}, params={len(group['params'])}")
        
    return param_groups


def plot_lr(scheduler, total_steps):
    lrs = []
    for step in range(total_steps):
        lrm = scheduler.get_lr()  # fetch before stepping
        lrs.append(lrm)
        scheduler.step()

    import matplotlib.pyplot as plt
    steps = list(range(total_steps))
    plt.plot(steps, lrs, label="LR")
    plt.xlabel("Step")
    plt.ylabel("LR multiplier")
    plt.title("LR Schedule")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
