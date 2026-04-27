#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import torchvision.transforms as T
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from tqdm import tqdm

from hac import lorentz as L
from hac.evaluation.catalog import DatasetCatalog
from hac.evaluation.class_names import CLASS_NAMES
from hac.models import AdaptedCLIP, HyCoCLIP, MERU, CLIPBaseline
from hac.tokenizer import Tokenizer


class ZeroShotClassificationEvaluator:
    """
    Evaluate trained models for zero-shot image classification, wherein the entire
    model is transferred to the downstream task without additional training. This
    protocol is similar to CLIP: the classifier weights are constructed by encoding
    text prompts of class labels using text encoder.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        datasets_and_prompts: dict[str, list[str]],
        data_dir: str | Path,
        image_size: int = 224,
        use_model_device: bool = False,
        confusion_matrix: bool = False,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
        """
        self._datasets_and_prompts = datasets_and_prompts
        self._data_dir = Path(data_dir).resolve()
        self._image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )
        self.tokenizer = Tokenizer()
        self.use_model_device = use_model_device
        self.confusion_matrix = confusion_matrix

    @torch.inference_mode()
    def __call__(self, model: HyCoCLIP | MERU | CLIPBaseline | AdaptedCLIP) -> dict[str, float]:
        model = model.eval()
        tokenizer = self.tokenizer

        # Collect results per task in this dict:
        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            # ----------------------------------------------------------------
            # Make zero-shot classifier using class name and prompts.
            # ----------------------------------------------------------------
            class_names = CLASS_NAMES[dname]

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for i, name in enumerate(class_names):
                class_prompts = [_pt.format(name) for _pt in prompts]

                class_prompt_tokens = tokenizer(class_prompts)
                class_feats = model.encode_text(class_prompt_tokens, project=False)

                if isinstance(model, (HyCoCLIP, MERU, AdaptedCLIP)):
                    # Ensemble in the tangent space, then project to Hyperboloid.
                    class_feats = class_feats.mean(dim=0) # ensemble over different prompts before going to hyperbolic space
                    class_feats = class_feats * model.textual_alpha.exp()
                    class_feats = L.exp_map0(class_feats, model.curv.exp())
                else:
                    # Ensemble prompt features: normalize -> average -> normalize.
                    class_feats = F.normalize(class_feats, dim=-1)
                    class_feats = class_feats.mean(dim=0)
                    class_feats = F.normalize(class_feats, dim=-1)

                all_class_feats.append(class_feats)
                
            classifier = torch.stack(all_class_feats, dim=0)
            logger.info(f"Classifier shape: {classifier.size()}")

            # Extract image features and labels from the test split of required dataset.
            loader = DataLoader(
                DatasetCatalog.build(
                    dname, self._data_dir, "test", self._image_transform
                ),
                batch_size=64,
            )
            image_feats, labels = _encode_dataset(loader, model, project=True)

            # Features returned by this function will be on CPU, move to device:
            image_feats = image_feats.to(model.device)

            # Measure model performance according to accuracy metric of the dataset.
            acc_meter = MulticlassAccuracy(DatasetCatalog.NUM_CLASSES[dname], dist_sync_on_step=False, sync_on_compute=False, average=("macro" if not self.confusion_matrix else "none"))
            # https://github.com/Lightning-AI/torchmetrics/issues/1899
            acc_meter.dist_sync_on_step=False
            acc_meter.sync_on_compute=False
            if self.use_model_device:
                acc_meter = acc_meter.to(model.device)
                
            if self.confusion_matrix:
            # confusion matrix
                confmat_meter = MulticlassConfusionMatrix(num_classes=DatasetCatalog.NUM_CLASSES[dname], normalize=None)
                confmat_meter.dist_sync_on_step=False
                confmat_meter.sync_on_compute=False
                if self.use_model_device:
                    confmat_meter = confmat_meter.to(model.device)

            # Evaluate in small batches of 256 instances.
            for _feats, _labels in zip(image_feats.split(256), labels.split(256)):
                # Compute pairwise similarity depending on model type:
                if isinstance(model, (HyCoCLIP, MERU, AdaptedCLIP)):
                    scores = L.pairwise_inner(_feats, classifier, model.curv.exp())
                else:
                    scores = _feats @ classifier.T
                    
                preds = scores.argmax(dim=1)

                if self.use_model_device:
                    acc_meter(scores, _labels.to(model.device))
                    if self.confusion_matrix:
                        confmat_meter(preds, _labels.to(model.device))
                else:
                    acc_meter(scores.cpu(), _labels)
                    if self.confusion_matrix:
                        confmat_meter(preds.cpu(), _labels)


            accuracy = acc_meter.compute() * 100.0
            if self.confusion_matrix:
                confmat = confmat_meter.compute()
                confmat_np = confmat.cpu().numpy()
                
                # sanity check
                diag = np.diag(confmat_np)
                row_sum = confmat_np.sum(axis=1)
                per_class_acc = diag / row_sum * 100.0
                assert np.allclose(per_class_acc, accuracy.cpu().numpy()), f"Mismatch between confusion matrix and accuracy computation: {per_class_acc} vs {accuracy.cpu().numpy()}"
            
                class_names = CLASS_NAMES[dname]
            
                results_dict[dname] = {
                    "accuracy": accuracy.cpu().numpy(),
                    "confusion_matrix": confmat_np,
                    "class_names": class_names,
                }
                # print accurary per class and confusion matrix        
                # --- compact aligned print: classes row, accuracies row ---
                w = max(6, min(14, max(len(n) for n in class_names)))  # column width
                name_row = "class: " + " ".join(f"{n:>{w}.{w}}" for n in class_names)
                acc_row =  "acc %: " + " ".join(f"{a:>{w}.1f}" for a in per_class_acc)
                
                logger.info(f"[{dname}] Overall accuracy: {accuracy.mean():.1f}%")
                logger.info(f"[{dname}] {name_row}")
                logger.info(f"[{dname}] {acc_row}")
                # --- full confusion matrix (counts) ---
                logger.info(f"[{dname}] Confusion matrix (counts):\n{confmat_np}") 
            else:
                results_dict[dname] = accuracy

                logger.info(
                    f"Zero-shot evaluation: {dname}, {len(image_feats)} images, "
                    f"{len(class_names)} classes [acc.: {accuracy:.1f}%] "
                )

        return results_dict


def _encode_dataset(
    data_loader: DataLoader,
    model: HyCoCLIP | MERU | CLIPBaseline | AdaptedCLIP,
    project: bool,
):
    """
    Extract image features and labels for a given dataset using the given model.

    Args:
        data_loader: PyTorch dataset or dataloader that serves instances/batches
            of `(image, label)` tuples.
        model: Model that implements `encode_image` method to extract features.
        project: Input argument to `model.encode_image`.
    """

    # Collect batches of extracted image features and labels (as-is from loader).
    all_image_feats, all_labels = [], []

    for images, labels in tqdm(data_loader, desc=f"Extracting image feats"):
        with torch.inference_mode():
            image_feats = model.encode_image(images.to(model.device), project)

        # all_image_feats.append(image_feats.cpu())
        all_image_feats.append(image_feats)
        all_labels.append(labels)

    logger.info(f"Extracted {len(all_image_feats)} batches of image features.")
    return torch.cat(all_image_feats, dim=0), torch.cat(all_labels, dim=0)
