#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

"""
Evaluate a trained model using implementations from `hycoclip.evaluation` module.
"""
from __future__ import annotations

import argparse
import os
import csv
import json

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger
from dataclasses import replace

from hac.config import LazyConfig, LazyFactory
from hac.evaluation.vqa import VQAEvaluator
from hac.utils.checkpointing import CheckpointManager
from hac.evaluation.classification import ZeroShotClassificationEvaluator
from hac.utils.plain_mha import replace_mha_with_plain
from hac.encoders.clip_adapters import get_adapted_encoder
from peft import get_peft_model


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--config", help="Path to an evaluation config file (.py)")
_AA("--checkpoint-path", help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--csv-path", help="Path to save evaluation results in CSV format.", default=None)


def run_eval(_C_EVAL, _C_TRAIN, device, checkpoint_path, csv_path=None, config_file=None):
    logger.info(f"Evaluating checkpoint in {checkpoint_path}...")
    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    evaluator = instantiate(_C_EVAL.evaluator)
    if "tokenizer" in _C_TRAIN:
        tokenizer = LazyFactory.build_tokenizer(_C_TRAIN)
        evaluator.tokenizer = tokenizer
        logger.info(f"Using custom tokenizer: {tokenizer.__class__.__name__}")
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    if (_C_TRAIN.get("lora_config")):
        logger.info("Applying PEFT to the model.")
        config = LazyFactory.build_generic_config(_C_TRAIN, key="lora_config")
        logger.info(f"PEFT config: {config}")
        # replace the MHA with plain MHA
        replace_mha_with_plain(model.visual, device=device)
        replace_mha_with_plain(model.textual, device=device)
        logger.info("Replaced MHA with PlainMHA in original CLIP model.")
        model.visual = get_peft_model(model.visual, config)
        model.textual = get_peft_model(model.textual, config)
        logger.info("PEFT model created.")
    elif _C_TRAIN.get("adapter_config") or _C_TRAIN.get("visual_adapter_config"):
        adapter_config = None
        if _C_TRAIN.get("adapter_config"):
            logger.info("Applying ADAPTERS to the model...")
            adapter_config = LazyFactory.build_generic_config(_C_TRAIN, key="adapter_config")
            # (optional) leave out some blocks from the adaptation
            if hasattr(adapter_config, "leave_out") and not isinstance(adapter_config.leave_out, list):
                adapter_config = replace(adapter_config, leave_out=list(adapter_config.leave_out))
            logger.info(f"ADAPTERS config: {adapter_config}")
        elif _C_TRAIN.get("visual_adapter_config"):
            visual_adapter_config = LazyFactory.build_generic_config(_C_TRAIN, key="visual_adapter_config")
            # (optional) leave out some blocks from the adaptation
            if hasattr(visual_adapter_config, "leave_out") and not isinstance(visual_adapter_config.leave_out, list):
                visual_adapter_config = replace(visual_adapter_config, leave_out=list(visual_adapter_config.leave_out))
            textual_adapter_config = LazyFactory.build_generic_config(_C_TRAIN, key="textual_adapter_config")
            # (optional) leave out some blocks from the adaptation
            if hasattr(textual_adapter_config, "leave_out") and not isinstance(textual_adapter_config.leave_out, list):
                textual_adapter_config = replace(textual_adapter_config, leave_out=list(textual_adapter_config.leave_out))
            logger.info("Found separate ADAPTERS configs for visual and textual encoders.")
            logger.info(f"ADAPTERS configs: \n Visual: {visual_adapter_config} \n Textual: {textual_adapter_config}")

        visual_config = LazyFactory.build_generic_config(_C_TRAIN, key="visual_config")
        textual_config = LazyFactory.build_generic_config(_C_TRAIN, key="textual_config")
        # replace the MHA with plain MHA
        replace_mha_with_plain(model.visual, device=device)
        model.visual = get_adapted_encoder(model.visual, "visual", visual_config, (adapter_config or visual_adapter_config))
        replace_mha_with_plain(model.textual, device=device)
        model.textual = get_adapted_encoder(model.textual, "textual", textual_config, (adapter_config or textual_adapter_config))
        model.to(device)
        logger.info("ADAPTERS applied to the model.")
    
    CheckpointManager(model=model).load(checkpoint_path)

    results_dict = evaluator(model)
    
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    checkpoint_folder = os.path.dirname(checkpoint_path)
    
    if isinstance(evaluator, VQAEvaluator) and evaluator.is_test:
        logger.info("Test set evaluation completed. Serializing predictions to JSON...")
        vqa_type = os.path.basename(config_file).split('.')[0].split("eval_")[-1]
        out_json_path = os.path.join(checkpoint_folder, f"{vqa_type}_test_predictions_" + checkpoint_name + ".json")
        with open(out_json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        # create empty txt file with same name to indicate test eval is done
        with open(out_json_path.replace('.json', '.txt'), 'w') as f:
            f.write("")
        logger.info(f"Predictions saved to {out_json_path}")
        return

    # Log results for copy-pasting to spreadsheet, including checkpoint path.
    header = ",".join(results_dict.keys())
    try:
        numbers = ",".join([f"{num:.1f}" for num in results_dict.values()])
    except TypeError: # confusion matrix case
        numbers = ",".join([f"{dct['accuracy'].mean():.1f}" for dct in results_dict.values()])

    logger.info(f"copypaste: {_A.checkpoint_path}")
    logger.info(f"\ncopypaste below:\n{header}\n{numbers}")
    
    # SAVE TO CSV
    if csv_path is None:
        # infer automatically csv path based on evaluator type
        if isinstance(evaluator, ZeroShotClassificationEvaluator):
            csv_path = os.path.join(checkpoint_folder, "classification_results_" + checkpoint_name + ".csv")
        elif isinstance(evaluator, VQAEvaluator):
            vqa_type = os.path.basename(config_file).split('.')[0].split("eval_")[-1]
            csv_path = os.path.join(checkpoint_folder, f"{vqa_type}_results_" + checkpoint_name + ".csv")
        else:
            raise ValueError("Unknown evaluator type, cannot determine CSV path.")
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # create file if does not exist, otherwise append
        mode = 'w' if not os.path.exists(csv_path) else 'a'
        with open(csv_path, mode, newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if mode == 'w':
                # write header only if file is created, first column is checkpoint folder
                writer.writerow([""] + list(results_dict.keys()))
            writer.writerow([checkpoint_folder.split(os.sep)[-1]] + [f"{num:.2f}" for num in results_dict.values()])
        logger.info(f"Results saved to {csv_path}")
        # print average of all metrics
        avg_metrics = sum(results_dict.values()) / len(results_dict.values())
        logger.info(f"Average metrics: {avg_metrics}")
        # print standard deviation of all metrics
        std_metrics = (sum((x - avg_metrics) ** 2 for x in results_dict.values()) / len(results_dict.values())) ** 0.5
        logger.info(f"Standard deviation of metrics: {std_metrics}")


def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    _C = LazyConfig.load(_A.config)
    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    run_eval(_C, _C_TRAIN, device, _A.checkpoint_path, _A.csv_path, _A.config)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
