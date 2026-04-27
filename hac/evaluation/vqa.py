from __future__ import annotations

from operator import is_
from pathlib import Path

import torch
import torchvision.transforms as T
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hac import lorentz as L
from hac.models import AdaptedCLIP, HyCoCLIP, MERU, CLIPBaseline
from hac.tokenizer import Tokenizer

import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

BATCH_SIZE = 128


class VQADataset(Dataset):
    # images stored in data_dir/name/images
    # annotations stored in data_dir/name/<name>_ann.json
    def __init__(self, dataset_dir, image_transform, tokenizer, use_qa_prompts=False, context_length=77, is_test=False):
        self.images_dir = Path(dataset_dir) / "images"
        self.annotations_file = Path(dataset_dir) / f"{dataset_dir.name}_ann.json"
        self.image_transform = image_transform
        self.use_qa_prompts = use_qa_prompts
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.is_test = is_test

        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = self.images_dir / item['image_path']
        question = item['question']
        answers = item['options'].values()
        correct_answer = item['answer']
        original_index = item.get('index', idx)
        
        if self.is_test:
            correct_idx = -1  # unknown in test set
        else:
            correct_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[correct_answer]

        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        if self.use_qa_prompts:
            prompts = [f"Question: {question} Answer: {answer}" for answer in answers]
        else:
            prompts =  [f"{question} {answer}" for answer in answers]
            
        tokenized_prompts = self.tokenizer(prompts) # list of tensors, 4 x Num_tokens (EOS = 49407)
        for idx, inst_tokens in enumerate(tokenized_prompts):
            if len(inst_tokens) > self.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.context_length]
                inst_tokens[-1] = eot_token
                tokenized_prompts[idx] = inst_tokens
            # Len -> context_length
            tokenized_prompts[idx] = self.pad_to_length(inst_tokens, self.context_length)
            
        tokenized_prompts = torch.stack(tokenized_prompts)  # 4 x context_length

        return image, tokenized_prompts, correct_idx, original_index # 3x224x224, 4x77, int
    
    @staticmethod
    def pad_to_length(tensor, length, pad_value=0):
        pad_len = length - tensor.size(0)
        if pad_len > 0:
            tensor = F.pad(tensor, (0, pad_len), value=pad_value)
        return tensor
    
    @staticmethod
    def collate_fn(batch):   
        images, all_tokenized_prompts, correct_indexes, original_indexes = zip(*batch)
        images = torch.stack(images) # BS x 3 x 224 x 224
        all_tokenized_prompts = torch.stack(all_tokenized_prompts) # BS x 4 x 77
        correct_indexes = torch.tensor(correct_indexes) # BS
        original_indexes = torch.tensor(original_indexes) # BS
        return images, all_tokenized_prompts, correct_indexes, original_indexes # BS x 3 x 224 x 224, BS x 4 x 77, BS, BS


class VQAEvaluator:
    def __init__(
        self,
        datasets: dict[str, list[str]],
        data_dir: str | Path,
        image_size: int = 224,
        use_model_device: bool = False,
        use_qa_prompts: bool = False,
        is_test: bool = False,
    ):
        self._datasets = datasets
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
        self.use_qa_prompts = use_qa_prompts
        self.is_test = is_test
        
    @torch.inference_mode()
    def __call__(self, model: HyCoCLIP | MERU | CLIPBaseline | AdaptedCLIP) -> dict[str, float]:
        model = model.eval()
        
        # count number of parameters
        #num_params = sum(p.numel() for p in model.parameters())
        #logger.info(f"Evaluating VQA with model having {num_params/1e6}M parameters")
        #exit(0)

        # Collect results per task in this dict:
        results_dict = {}
        test_results = []

        for dname in self._datasets:
            
            num_correct_total = 0
            num_samples_total = 0
            
            loader = DataLoader(
                VQADataset(
                    dataset_dir=self._data_dir / dname,
                    image_transform=self._image_transform,
                    tokenizer=self.tokenizer,
                    use_qa_prompts=self.use_qa_prompts,
                    is_test=self.is_test,
                ),
                batch_size=BATCH_SIZE,
                collate_fn=VQADataset.collate_fn,
            )
            # BS x 3 x 224 x 224, BS x 4 x 77, BS -> BS x D, BS x 4 x D, BS
            image_feats, prompts, correct_indexes, original_indexes = _encode_vqa_dataset(loader, model)
            
            # move to device of model
            image_feats = image_feats.to(model.device)
            prompts = prompts.to(model.device)
            correct_indexes = correct_indexes.to(model.device)
            
            # iterate over single samples
            for _im_feats, _prompts, _gt_idxs, _orig_idx in zip(
                image_feats, prompts, correct_indexes, original_indexes
            ):
                # _im_feats: D
                # _prompts: 4 x D
                # _gt_idxs: int
                
                if isinstance(model, (HyCoCLIP, MERU, AdaptedCLIP)):
                    # B1, D
                    # B2, D
                    scores = L.pairwise_inner(_im_feats.unsqueeze(0), _prompts, model.curv.exp()) # -> 1 x 4
                else:
                    # B1, D
                    # D, B2
                    scores = _im_feats.unsqueeze(0) @ _prompts.T # -> 1 x 4
                   
                pred_index = scores.argmax(dim=-1).squeeze(dim=0)  # 1
                
                if self.is_test:
                    test_results.append({
                        "index": _orig_idx.item(),
                        "prediction": {0: "A", 1: "B", 2: "C", 3: "D"}[pred_index.item()]
                    })
                else:
                    is_correct = (pred_index == _gt_idxs).item()
                    num_correct_total += is_correct
                    num_samples_total += 1
                
            if not self.is_test:
                accuracy = 100.0 * num_correct_total / num_samples_total
                results_dict[f"{dname}"] = accuracy
                logger.info(f"{dname}: {accuracy:.3f}%")
            
        return results_dict if not self.is_test else test_results
                

def _encode_vqa_dataset(
    data_loader: DataLoader,
    model: HyCoCLIP | MERU | CLIPBaseline | AdaptedCLIP,
    project: bool = True,
):
    all_image_feats, all_prompts, all_correct_indexes, all_original_indexes = [], [], [], []
    for images, prompts, correct_indexes, original_indexes in tqdm(data_loader, desc=f"Extracting image feats and prompts"):
        with torch.inference_mode():
            image_feats = model.encode_image(images.to(model.device), project)
            # flatten prompts: BS x 4 x 77 -> (BS*4) x 77
            bs, num_options, ctx_len = prompts.size()
            flat_prompts = prompts.view(bs * num_options, ctx_len).to(model.device)
            prompt_feats = model.encode_text(flat_prompts, project)
            # reshape back to BS x 4 x 77
            prompt_feats = prompt_feats.view(bs, num_options, -1)
            
        all_image_feats.append(image_feats)
        all_prompts.append(prompt_feats)
        all_correct_indexes.append(correct_indexes)
        all_original_indexes.append(original_indexes)
        
    logger.info(f"Extracted {len(all_image_feats)} batches of image and prompt features.")
    return torch.cat(all_image_feats, dim=0), torch.cat(all_prompts, dim=0), torch.cat(all_correct_indexes, dim=0), torch.cat(all_original_indexes, dim=0)  # N x D, N x 4 x D, N, N