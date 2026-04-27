import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks import default as ddph
from torch.nn.parallel import DistributedDataParallel
from loguru import logger

import hac.utils.distributed as dist

def get_model_memory_info(model):
    param_bytes = 0
    grad_bytes = 0
    buffer_bytes = 0
    total_params = 0
    grad_params = 0

    for param in model.parameters():
        size = param.numel() * param.element_size()
        param_bytes += size
        total_params += param.numel()

        if param.requires_grad:
            grad_bytes += size
            grad_params += param.numel()

    for buffer in model.buffers():
        buffer_bytes += buffer.numel() * buffer.element_size()

    total_bytes = param_bytes + grad_bytes + buffer_bytes
    mb = lambda b: b / 1024**2

    logger.info("Model Memory Breakdown:")
    logger.info(f"  Parameters : {mb(param_bytes):.2f} MB")
    logger.info(f"  Gradients  : {mb(grad_bytes):.2f} MB (only {grad_params:,} trainable)")
    logger.info(f"  Buffers    : {mb(buffer_bytes):.2f} MB")
    logger.info("  -----------------------------")
    logger.info(f"  Total      : {mb(total_bytes):.2f} MB")
    logger.info(f"  Total Parameters: {total_params:,} (Trainable: {grad_params:,})")
    trainable_percent = 100.0 * grad_params / total_params if total_params > 0 else 0
    logger.info(f"  Trainable Parameters Percentage: {trainable_percent:.2f}% ({grad_params:,} / {total_params:,})")
    
    return mb(total_bytes)


def wrap_model_DDP(model, cfg, device):
    """
    Wraps the model in a DistributedDataParallel (DDP) wrapper if using multiple GPUs.
    Also optionally adds an FP16 compression hook if specified in the configuration.

    Args:
        model (nn.Module): The model to be wrapped.
        cfg (dict): Configuration dictionary containing training parameters.
        device (torch.device): The device on which the model is located.

    Returns:
        nn.Module: The wrapped model.
    """

    # Wrap model in DDP if using more than one GPUs.
    if dist.get_world_size() > 1:
        model = DistributedDataParallel(model, [device], **cfg.train.ddp)

        # Optionally add FP16 compression hook with AMP.
        if cfg.train.amp and cfg.train.ddp_fp16_compression:
            model.register_comm_hook(state=None, hook=ddph.fp16_compress_hook)

    return model


def de_parallel(model):
    """Remove DDP wrapper from model.

    Parameters
    ----------
    model: Pytorch model.
    """
    return (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )


def generate_lora_param_names(visual_blocks: list[int], textual_blocks: list[int], separate_qkv=False, components="attn", verbose=True) -> list[str]:
    params = []

    for i in visual_blocks:
        if "attn" in components:
            if separate_qkv:
                params.append(f"blocks.{i}.attn.q_proj")
                params.append(f"blocks.{i}.attn.k_proj")
                params.append(f"blocks.{i}.attn.v_proj")
            else:
                params.append(f"blocks.{i}.attn.qkv")
        if "proj" in components:
            params.append(f"blocks.{i}.attn.proj")
        if "ffn" in components:
            params.append(f"blocks.{i}.mlp.fc1")
            params.append(f"blocks.{i}.mlp.fc2")
    for i in textual_blocks:
        if "attn" in components:
            if separate_qkv:
                params.append(f"resblocks.{i}.attn.q_proj")
                params.append(f"resblocks.{i}.attn.k_proj")
                params.append(f"resblocks.{i}.attn.v_proj")
            else:
                params.append(f"resblocks.{i}.attn.qkv")
        if "proj" in components:
            params.append(f"resblocks.{i}.attn.proj")
        if "ffn" in components:
            params.append(f"resblocks.{i}.mlp.c_fc")
            params.append(f"resblocks.{i}.mlp.c_proj")
            
    if verbose: print(f"Generated LoRA parameter names: {params}")

    return params



