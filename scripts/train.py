#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Train a HyCoCLIP, MERU or CLIP model based on parameters specified by a config file.
"""
import argparse
import time
import random
from pathlib import Path
import socket
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
# Optional: suppress oneDNN notices
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Suppress absl logging (used internally by TF)
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
import copy
from dataclasses import replace

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from torch.cuda import amp
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from hydra.utils import instantiate

import hac.utils.distributed as dist
from hac.config import LazyConfig, LazyFactory
from hac.tokenizer import Tokenizer
from hac.utils.checkpointing import CheckpointManager
from hac.utils.timer import Timer
from hac.models import HyCoCLIP, AdaptedCLIP
from hac.utils.model_utils import (
    de_parallel, 
    get_model_memory_info, 
    wrap_model_DDP, 
)
from hac.utils.plain_mha import replace_mha_with_plain
from peft import get_peft_model
from hac.encoders.clip_adapters import get_adapted_encoder
from scripts.evaluate import run_eval

import warnings; warnings.filterwarnings("ignore", category=FutureWarning)


# fmt: off
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", help="Path to a .py config file.")
parser.add_argument(
    "--output-dir", default="./output",
    help="Path to a directory to save checkpoints and job logs.",
)
parser.add_argument(
    "--resume", action="store_true",
    help="Whether to resume training from `--output-dir`. This script will find "
    "the last saved checkpoint and resume training. It is user's responsibility "
    "to provide matching config file in `--config`.",
)
parser.add_argument(
    "--checkpoint-period", type=int, default=5000, help="Checkpoint saving period."
)
parser.add_argument(
    "--log-period", type=int, default=100,
    help="Log to stdout/tensorboard periodically (only main process).",
)
parser.add_argument(
    "--eval-period", type=int, default=None,
    help="Evaluate the model periodically (only main process).",
)
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs per machine."
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="Integer in [0, num_machines) to specifying machine ID.",
)
_random_port = random.randint(2000, 19999)
parser.add_argument(
    "--dist-url", default=f"tcp://127.0.0.1:{_random_port}",
    help="URL of the main process in distributed training, it defaults to "
    "localhost for single-machine training.",
)
parser.add_argument(
    "--eval-mode", type=str, default=None, choices=[None, "cls"],
    help="Evaluation mode to use. If None, the default mode in the config will be used."
)
parser.add_argument(
    "--max-iters", type=int, default=1000000,
    help="Maximum number of iterations to run the training loop. "
)
parser.add_argument(
    "--csv-path", type=str, default=None,
    help="Path to save evaluation results in CSV format at the end of training.",
)
parser.add_argument(
    "--keep_ckpts", action="store_true", default=False,
    help="Keep all checkpoints and do not delete any (only main process)."
)
parser.add_argument(
    "overrides", nargs="...", default=[], help="Config overrides (key-value pairs)."
)

# fmt: on


def main(_A: argparse.Namespace):
    # -------------------------------------------------------------------------
    #   BASIC SETUP FOR TRAINING JOB.
    # -------------------------------------------------------------------------
    # Create a config object and perform common setup.
    _C = LazyConfig.load(_A.config)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)
    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    if getattr(_C.train, "seed", None) is None:
        _C.train.seed = int(time.time())

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.train.seed + RANK)
    np.random.seed(_C.train.seed + RANK)
    torch.manual_seed(_C.train.seed + RANK)
    torch.backends.cudnn.deterministic = _C.train.cudnn_deterministic
    torch.backends.cudnn.benchmark = _C.train.cudnn_benchmark
    
    # update output folder if there are overrides
    if _A.overrides:   
        suffix_parts = []
        for override in _A.overrides:
            key, val = override.split("=", 1)
            short_key = key.split(".")[-1]  # e.g., "lr" from "optim.optimizer.lr"
            suffix_parts.append(f"{short_key}_{val}")
        suffix = "_".join(suffix_parts)
        _A.output_dir = f"{_A.output_dir}_{suffix}"

    # Create output directory and save config in it.
    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LazyConfig.save(_C, output_dir / "config.yaml")

    # Create a logger for each process which writes to a separate log-file.
    logger.add(output_dir / f"log-rank{RANK}.txt", format="{time} {level} {message}")

    # Print process info, config and args.
    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(f"RANK {RANK} using random seed: {_C.train.seed + RANK}")
    if dist.is_main_process():
        logger.info(OmegaConf.to_yaml(_C))

        logger.info("Command line args:")
        for arg in vars(_A):
            logger.info(f"{arg:<20}: {getattr(_A, arg)}")
        
    # setup evaluation objects
    evaluator_cls = None
    if _A.eval_mode in ["cls"]:
        _C_CLS = LazyConfig.load("configs/eval_zero_shot_classification_online.py")
        evaluator_cls = instantiate(_C_CLS.evaluator)

    # -------------------------------------------------------------------------
    #   INSTANTIATE ALL OBJECTS FOR TRAINING.
    # -------------------------------------------------------------------------
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if _A.num_gpus != 0
        else torch.device("cpu")
    )
    dataloader = LazyFactory.build_dataloader(_C)
    if dist.is_main_process(): logger.info(f"Training dataloader: {dataloader.__class__.__name__}")

    # instantiate val dataloader (if any in config)
    if "dataset_val" in _C and dist.is_main_process():
        dataloader_val = LazyFactory.build_val_dataloader(_C)
        logger.info(f"Validation dataloader: {dataloader_val.__class__.__name__}")
        # check evaluation period is set
        if _A.eval_period is None:
            logger.warning("Evaluation period NOT set: NO validation will be performed.")
    else:
        dataloader_val = None
        if dist.is_main_process(): logger.info("No validation dataloader found in config.")

    if _C.get("tokenizer"):
        tokenizer = LazyFactory.build_tokenizer(_C)
        logger.info(f"Using custom tokenizer: {tokenizer.__class__.__name__}")
    else:
        tokenizer = Tokenizer()

    model = LazyFactory.build_model(_C, device)

    if isinstance(model, AdaptedCLIP):
        # freeze the CLIP model parameters
        for p in model.visual.parameters():
            p.requires_grad = False
        for p in model.textual.parameters():
            p.requires_grad = False

        # init PEFT configs
        config = None
        visual_peft_config = None
        textual_peft_config = None
        mha_replaced = False
        # init ADAPTERS configs
        adapter_config = None
        visual_adapter_config = None
        textual_adapter_config = None

        # (optional) apply PEFT to the model
        if (_C.get("lora_config") or _C.get("peft_config")) or _C.get("visual_peft_config") or _C.get("textual_peft_config"):
            if dist.is_main_process(): logger.info("Applying PEFT to the model...")
            # apply same PEFT config to both encoders
            if _C.get("lora_config") or _C.get("peft_config"):
                config = LazyFactory.build_generic_config(_C, key="lora_config") or LazyFactory.build_generic_config(_C, key="peft_config")
                if dist.is_main_process():logger.info(f"PEFT config: {config}")
            else:
                visual_peft_config = LazyFactory.build_generic_config(_C, key="visual_peft_config")
                textual_peft_config = LazyFactory.build_generic_config(_C, key="textual_peft_config")
                if dist.is_main_process():
                    logger.info("Found separate PEFT configs for visual and textual encoders.")
                    logger.info(f"PEFT configs: \n Visual: {visual_peft_config} \n Textual: {textual_peft_config}")
            # replace the MHA with plain MHA
            if config or visual_peft_config:
                replace_mha_with_plain(model.visual, device=device)
            if config or textual_peft_config:
                replace_mha_with_plain(model.textual, device=device)
            mha_replaced = True
            if dist.is_main_process(): logger.info("Replaced MHA with PlainMHA in original CLIP model.")
            # rebuild the model with PEFT modules
            if config or visual_peft_config:
                model.visual = get_peft_model(model.visual, visual_peft_config or config)
            if config or textual_peft_config:
                model.textual = get_peft_model(model.textual, textual_peft_config or config)
            if dist.is_main_process(): logger.info("PEFT model created.")

        # optional: apply adapters to the model
        if _C.get("adapter_config") or (_C.get("visual_adapter_config") and _C.get("textual_adapter_config")):
            if dist.is_main_process(): logger.info("Applying ADAPTERS to the model...")
            # apply same ADAPTER to both encoders
            if _C.get("adapter_config"):
                adapter_config = LazyFactory.build_generic_config(_C, key="adapter_config")
                # (optional) leave out some blocks from the adaptation
                if hasattr(adapter_config, "leave_out") and not isinstance(adapter_config.leave_out, list):
                    adapter_config = replace(adapter_config, leave_out=list(adapter_config.leave_out))
                if dist.is_main_process():logger.info(f"ADAPTERS config: {adapter_config}")
            else:
                visual_adapter_config = LazyFactory.build_generic_config(_C, key="visual_adapter_config")
                # (optional) leave out some blocks from the adaptation
                if hasattr(visual_adapter_config, "leave_out") and not isinstance(visual_adapter_config.leave_out, list):
                    visual_adapter_config = replace(visual_adapter_config, leave_out=list(visual_adapter_config.leave_out))
                textual_adapter_config = LazyFactory.build_generic_config(_C, key="textual_adapter_config")
                # (optional) leave out some blocks from the adaptation
                if hasattr(textual_adapter_config, "leave_out") and not isinstance(textual_adapter_config.leave_out, list):
                    textual_adapter_config = replace(textual_adapter_config, leave_out=list(textual_adapter_config.leave_out))
            visual_config = LazyFactory.build_generic_config(_C, key="visual_config")
            textual_config = LazyFactory.build_generic_config(_C, key="textual_config")
            if dist.is_main_process():
                logger.info("Found separate ADAPTERS configs for visual and textual encoders.")
                logger.info(f"ADAPTERS configs: \n Visual: {visual_adapter_config} \n Textual: {textual_adapter_config}")
            if not mha_replaced:
                # replace the MHA with plain MHA
                replace_mha_with_plain(model.visual, device=device)
                model.visual = get_adapted_encoder(model.visual, "visual", visual_config, (adapter_config or visual_adapter_config))
                replace_mha_with_plain(model.textual, device=device)
                model.textual = get_adapted_encoder(model.textual, "textual", textual_config, (adapter_config or textual_adapter_config))
                if dist.is_main_process(): logger.info("Replaced MHA with PlainMHA in original CLIP model.")
            model.to(device)
            if dist.is_main_process(): logger.info("ADAPTERS applied to the model.")
            
        # last layer -> unfreeze final layer norm
        if getattr(model, "init_final_ln", False):
            # if init_final_ln is True, unfreeze the final layer norm
            for param in model.visual.norm.parameters(): param.requires_grad = True
            for param in model.textual.ln_final.parameters(): param.requires_grad = True
            if dist.is_main_process(): logger.info("Unfrozen final layer norm in visual and textual encoders.")
            
    # print only if main process
    if dist.is_main_process():
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                pass #logger.info(f"Freezing {name}")
            else:
                logger.info(f"Training {name}")
        # print how much memory is used by the model
        get_model_memory_info(model)

    # after model modifications, wrap the model in DDP (if using multiple GPUs)
    model = wrap_model_DDP(model, _C, device)

    optimizer = LazyFactory.build_optimizer(_C, model)
    scheduler = LazyFactory.build_lr_scheduler(_C, optimizer)
    scaler = amp.GradScaler(enabled=_C.train.amp)

    checkpoint_manager = CheckpointManager(
        _A.output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        keep_ckpts=_A.keep_ckpts,
    )
    start_iteration = checkpoint_manager.resume() if _A.resume else 0

    # Create an iterator from dataloader to sample batches perpetually.
    dataloader_iter = iter(dataloader)
    timer = Timer(start_iteration + 1, total_iterations=_C.train.num_iterations)

    # Create tensorboard writer, only in main process.
    if dist.is_main_process():
        tboard = SummaryWriter(log_dir=_A.output_dir)
        # add a writer for validation
        if _A.eval_period is not None:
            tboard_val = SummaryWriter(log_dir=_A.output_dir + "_val")

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, min(_C.train.num_iterations + 1, _A.max_iters + 1)):
        data_time = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - data_time

        timer.tic()
        optimizer.zero_grad()

        with amp.autocast(enabled=_C.train.amp):
            # Get image and text (tokens) from batch and pass through model.
            if isinstance(de_parallel(model), (HyCoCLIP, AdaptedCLIP)):
                tokens = tokenizer(batch["text"])
                box_tokens = tokenizer(batch["box_text"])
                output_dict = model(batch["image"].to(device),
                                    batch["box_image"].to(device),
                                    tokens,
                                    box_tokens) 
            else:
                tokens = tokenizer(batch["text"])
                output_dict = model(batch["image"].to(device), tokens)

            loss = output_dict["loss"]

        scaler.scale(loss).backward()

        if not dist.is_main_process():
            torch.distributed.barrier() # wait for main process to log gradients

        # unlock all processes in case of multiple GPUs
        if dist.is_main_process():
            torch.distributed.barrier()
                        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        timer.toc()
        
        # Perform validation.
        loss_val = None
        if _A.eval_period is not None and ((iteration % _A.eval_period == 0)): # or iteration == 1):
            # -------------------------------------------------------------------------
            #   VALIDATION LOOP
            # -------------------------------------------------------------------------
            if not dist.is_main_process():
                torch.distributed.barrier() # wait for main process to do validation
            
            if dist.is_main_process():
                model_eval = copy.deepcopy(de_parallel(model))
                model_eval.use_dist = False  # disable dist for evaluation
                model_eval.eval()
                num_val_samples = 0
                
                if dataloader_val is not None:
                    # Create an iterator from dataloader to sample batches perpetually.
                    dataloader_val_iter = iter(dataloader_val)
                
                    for iteration_val in tqdm(range(_C.val.num_iterations), desc=f"Val iter {iteration}"):
                        batch_val = next(dataloader_val_iter)
                        with torch.no_grad(), amp.autocast(enabled=_C.train.amp):
                            # Get image and text (tokens) from batch and pass through model.
                            tokens_val = tokenizer(batch_val["text"])
                            box_tokens_val = tokenizer(batch_val["box_text"])
                            output_dict_val = model_eval(batch_val["image"].to(device),
                                                        batch_val["box_image"].to(device),
                                                        tokens_val,
                                                        box_tokens_val) 
                            batch_size = batch_val["image"].shape[0]
                            if loss_val is None:
                                loss_val = (output_dict_val["loss"] * batch_size)
                            else:
                                loss_val += (output_dict_val["loss"] * batch_size)
                            num_val_samples += batch_size
                    # average the validation loss
                    loss_val = loss_val / num_val_samples
                
                # zero-shot classification evaluation
                imagenet_acc = None
                if evaluator_cls is not None:
                    results_dict_cls = evaluator_cls(model_eval)
                    imagenet_acc = results_dict_cls["imagenet"]
                
            # unlock all processes in case of multiple GPUs
            if dist.is_main_process():
                torch.distributed.barrier()

        # Log statistics to terminal and tensorboard.
        if iteration % _A.log_period == 0:
            
            if not dist.is_main_process():
                torch.distributed.barrier() # wait for main process to do logging
            
            if dist.is_main_process():
                timer_stats = (
                    f"Iter {timer.iteration} | Time (sec): {data_time:.3f} data, "
                    f"{timer.deltas[-1]:.3f} model | ETA: {timer.eta_hhmm}"
                )

                log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
                for key, value in output_dict["logging"].items():
                    if value is not None:
                        log_str += f" [{key} {value:.3f}]"

                logger.info(log_str)
                tboard.add_scalar("lr", scheduler.get_last_lr()[0], iteration)
                for group in optimizer.param_groups:
                    if group.get("name") == "other":
                        tboard.add_scalar("lr/other", group["lr"], iteration)
                    elif group.get("name") == "proj":
                        tboard.add_scalar("lr/proj", group["lr"], iteration)
                tboard.add_scalar("amp_scale", scaler.get_scale(), iteration)
                
                # validation active and a validation iteration
                if _A.eval_period is not None and (iteration % _A.eval_period == 0):
                    if loss_val is not None:
                        for name_val, _loss_val in output_dict_val["logging"].items():
                            if _loss_val is not None:
                                tboard_val.add_scalar(f"train/{name_val}", _loss_val, iteration)
                    # imagenet accuracy
                    if imagenet_acc is not None:
                        tboard_val.add_scalar(f"imagenet_acc", imagenet_acc, iteration)
                    tboard_val.flush()

                # always log training loss regardless of validation
                for name, _loss in output_dict["logging"].items():
                    if _loss is not None:
                        tboard.add_scalar(f"train/{name}", _loss, iteration)

                tboard.flush()
                
            # unlock all processes in case of multiple GPUs
            if dist.is_main_process():
                torch.distributed.barrier()

        # Save checkpoint to disk.
        if iteration % _A.checkpoint_period == 0 and dist.is_main_process():
            checkpoint_manager.step(iteration)

    # Save the final checkpoint.
    if dist.is_main_process():
        checkpoint_manager.final_step()
        
        # final evaluation
        if _A.eval_end is not None:
            final_checkpoint_path = os.path.join(_A.output_dir, "checkpoint_final.pth")
            if _A.eval_end in ["cls"]:
                _C_EVAL = LazyConfig.load(f"configs/eval_zero_shot_classification.py")
                _C_TRAIN = _C
                run_eval(_C_EVAL, _C_TRAIN, device, final_checkpoint_path, csv_path=_A.csv_path.replace(".csv", "_cls.csv"))
            

if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        # cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        # stdout = subprocess.check_output(cmd.split())
        # host_name = stdout.decode().splitlines()[0]
        # logger.info(f"Host name: {host_name}")
        # dist_url = f'tcp://{host_name}:{_random_port}'
        # logger.info(f"Distributed URL: {dist_url}")
        
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)

        dist_url = f"tcp://{IPAddr}:{_random_port}"

        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus,
            machine_rank=_A.machine_rank,
            dist_url=dist_url,
            args=(_A,),
        )
