#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

import math, os
from loguru import logger

import torch
from torch import nn
from torch.nn import functional as F

import hac.utils.distributed as dist
from hac import lorentz as L
from hac.encoders.text_encoders import TransformerTextEncoder


class CLIPBaseline(nn.Module):
    """
    Re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        # Linear layers to project image and text features such that they have
        # same size before computing dot-product similarity.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Project features to a unit hypersphere through L2 normalization.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Project features to a unit hypersphere through L2 normalization.
        """

        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)

        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        batch_size = image_feats.shape[0]
        targets = torch.arange(batch_size, device=image_logits.device)
        targets = targets + batch_size * self._rank

        loss = 0.5 * (
            F.cross_entropy(image_logits, targets)
            + F.cross_entropy(text_logits, targets)
        )
        output_dict = {
            "loss": loss,
            "logging": {"contrastive_loss": loss, "logit_scale": _scale},
        }
        return output_dict
        

class MERU(CLIPBaseline):
    """
    Implementation of MERU model that embeds images and text in a hyperbolic space.

    Reference: MERU paper (https://arxiv.org/abs/2304.09172)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """
        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(image_feats, self.curv.exp())

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.curv.exp())

        return text_feats

    def forward(
        self, images: torch.Tensor,
        tokens: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }


class HyCoCLIP(MERU):
    """
    Our HyCoCLIP model, that modifies MERU and CLIP to embed images, texts and their localized box 
    information hierarchically in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `MERU`.

        Args:
            use_boxes: Whether to use box images and texts for training.
        """
        super().__init__(visual, textual, embed_dim, curv_init, learn_curv, entail_weight, pixel_mean, pixel_std)
        assert use_boxes, "HyCoCLIP requires box images and texts to function."

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)


        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_image_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7   # inter-modal
            _local_aperture_thresh = 1.2    # intra-modal

            text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
            cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()
            
            entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss
            )

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "text_image_entailment_loss": text_image_entailment_loss,
                "box_text_image_entailment_loss": box_text_image_entailment_loss,
                "cross_image_entailment_loss": cross_image_entailment_loss,
                "cross_text_entailment_loss": cross_text_entailment_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }
        

class AdaptedCLIP(nn.Module):
    """
    Adaptation of a pre-trained, frozen CLIP model (MERU) that uses a Projection module 
    to pass from euclidean space to hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        contrastive_weight: float = 1.0,
        use_boxes: bool = True,
        checkpoint = None,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        init_proj: bool = True,
        learn_logit_scale: bool = True,
        learn_alpha_scale: bool = True,
        global_aperture_thresh: float = 0.7,
        local_aperture_thresh: float = 1.2,
        init_final_ln = False,
    ):
        """
        Args:
            clip_model: open_clip model (image encoder plus text encoder)
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
            contrastive_weight: Weight for the contrastive loss component.
            use_boxes: Whether to use box images and texts for training.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
            init_proj: Re-initialize the final CLIP projection (linear) layers.
            init_logit_scale: Whether to Re-initialize the logit scale parameter.
            learn_logit_scale: Whether to learn the logit scale parameter.
            learn_alpha_scale: Whether to learn the alpha scales for the visual and textual features.
            global_aperture_thresh: Threshold for the global aperture in the entailment loss.
            local_aperture_thresh: Threshold for the local aperture in the entailment loss.
            init_final_ln: Whether to initialize the final layer normalization.
        """
        super().__init__()
        self.visual: nn.Module = visual
        self.textual: TransformerTextEncoder = textual
        self.embed_dim: int = embed_dim
        self.init_proj: bool = init_proj
        self.learn_alpha_scale = learn_alpha_scale
        logger.info(f"Learning alpha scale: {self.learn_alpha_scale}.")
        self.global_aperture_thresh = global_aperture_thresh    # inter-modal
        self.local_aperture_thresh = local_aperture_thresh      # intra-modal
        self.init_final_ln = init_final_ln
        self.use_dist = True

        # Linear layers to project image and text features such that they have
        # same size before computing similarity in the hyperbolic space.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)
        
        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log(), requires_grad=learn_logit_scale)

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()
        
        self.use_boxes = use_boxes
        if not self.use_boxes: logger.warning("NOT USING BOXES for training.")
        
        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight
        self.contrastive_weight = contrastive_weight
        
        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        visual_alpha_init = embed_dim**-0.5
        self.visual_alpha = nn.Parameter(torch.tensor(visual_alpha_init).log(), requires_grad=self.learn_alpha_scale)
        textual_alpha_init = embed_dim**-0.5
        self.textual_alpha = nn.Parameter(torch.tensor(textual_alpha_init).log(), requires_grad=self.learn_alpha_scale)
            
        self.init_model(checkpoint_path=checkpoint)
        
    @property
    def device(self) -> torch.device:
        return self.logit_scale.device
    
    def init_model(self, checkpoint_path):
        """
        Initialize vanilla clip model and re-initialize the projection layers
        Args:
            checkpoint_path: Path to the checkpoint file.
        """

        assert os.path.exists(checkpoint_path), f"Checkpoint path does not exist: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        
        model_state = self.state_dict() # original model state dict
        model_keys = set(model_state.keys())
        adapter_model_keys = set([k for k in model_keys if "visual_adapter" in k or "textual_adapter" in k])
        checkpoint_keys = set(state_dict.keys()) # loaded model (checkpoint) state dict
        
        # Filter out checkpoint params to match the original model params
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Compute missing and unexpected keys
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        # List of parameters that are allowed to be missing (hycoclip specific)
        allowed_missing = {
            "curv",
            "visual_alpha",
            "textual_alpha",
            "textual.text_learnable_tokens",
            "textual.text_box_learnable_tokens",
        }.union(adapter_model_keys)

        if missing_keys - allowed_missing:
            raise ValueError(f"Missing unexpected keys: {missing_keys - allowed_missing}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        # load the model
        self.load_state_dict(filtered_state_dict, strict=False)
        
        # re-initialize the projection layers (optional)
        if self.init_proj:
            # CLIP-style initialization of projection layers.
            v_width = self.visual.width
            t_width = self.textual.width
            nn.init.normal_(self.visual_proj.weight, std=v_width**-0.5)
            nn.init.normal_(self.textual_proj.weight, std=t_width**-0.5)
            logger.info(f"Loaded model from {checkpoint_path} and re-initialized projection layers.")
        else:
            logger.info(f"Loaded model from {checkpoint_path} WITHOUT re-initializing projection layers.")
            
        if self.init_final_ln:
            # Initialize the final layer normalization (if any)
            self.visual.norm = nn.LayerNorm(self.visual.width)
            self.textual.ln_final = nn.LayerNorm(self.textual.width)
            logger.info("Initialized final layer normalization layers.")
            

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
              
        # project the image features to a common space
        image_feats = self.visual_proj(image_feats)
            
        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection. 
        if project:
            image_feats = self.project_visual(image_feats)

        return image_feats
    
    def project_visual(self, image_feats: torch.Tensor):
        image_feats = image_feats * self.visual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            image_feats = L.exp_map0(image_feats, self.curv.exp())
        return image_feats
     
    def encode_text(self, tokens: list[torch.Tensor], project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """
        K = self.textual.get_num_learnable_tokens()
        eot_idx = []
        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > (self.textual.context_length - K):
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: (self.textual.context_length - K)]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens
            if K > 0:
                eot_idx.append(len(inst_tokens) - 1)

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens, eot_indexes=eot_idx if K > 0 else None)

        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
            
        # project the text features to a common space
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = self.project_textual(text_feats)

        return text_feats
    
    
    def project_textual(self, text_feats: torch.Tensor):
        text_feats = text_feats * self.textual_alpha.exp()
        with torch.autocast(self.device.type, dtype=torch.float32):
            text_feats = L.exp_map0(text_feats, self.curv.exp())
        return text_feats
        

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor],
        use_dist: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            box_images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            box_tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            use_dist: Whether to collect features from all GPUs to increase negatives for contrastive loss.
        """
        
        use_dist = use_dist and self.use_dist

        if self.curv.requires_grad:
            # clamp only if the curvature is learnable
            self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        if self.use_boxes:
            box_image_feats = self.encode_image(box_images, project=True)
            box_text_feats = self.encode_text(box_tokens, project=True)

        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)
        
        def dist_gather(x): 
            return torch.cat(dist.gather_across_processes(x), dim=0) if use_dist else x
          
        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = dist_gather(image_feats)
        all_text_feats = dist_gather(text_feats)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            
            if self.contrastive_weight > 0:
                # Compute logits for contrastive loss.
                image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
                text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
                
                if self.use_boxes:
                    box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
                    box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)
                    
                # Compute cross entropy loss: we compute log probabilities and take the
                # diagonal elements as targets: image[i] should match text[i] in batch.
                # Shift the targets according to rank of GPU process (we assume that all
                # GPU processes have the same local batch size).
                batch_size = image_feats.shape[0]
                targets = torch.arange(batch_size, device=image_logits.device)
                targets = targets + batch_size * self._rank

                # Clamp temperature such that logits are not scaled more than 100x.
                # ln(100) = ~4.6052
                self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
                _scale = self.logit_scale.exp()

                ce_loss_image = nn.functional.cross_entropy(_scale * image_logits, targets)
                ce_loss_text = nn.functional.cross_entropy(_scale * text_logits, targets)
                ce_loss_box_image = nn.functional.cross_entropy(_scale * box_image_logits, targets) if self.use_boxes else 0
                ce_loss_box_text = nn.functional.cross_entropy(_scale * box_text_logits, targets) if self.use_boxes else 0
                
                contrastive_loss = (0.25 if self.use_boxes else 0.5) * (
                    ce_loss_image
                    + ce_loss_text
                    + ce_loss_box_image
                    + ce_loss_box_text
                )

            if self.entail_weight > 0:
                # Hyperbolic entailment loss: text should entail matching image.
                _angle = L.oxy_angle(text_feats, image_feats, _curv)
                _aperture = L.half_aperture(text_feats, _curv)

                if self.use_boxes:
                    _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
                    _box_aperture = L.half_aperture(box_text_feats, _curv)

                    _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
                    _box_image_aperture = L.half_aperture(box_image_feats, _curv)

                    _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
                    _box_text_aperture = L.half_aperture(box_text_feats, _curv)

                # Hyperparameters for apertures
                _global_aperture_thresh = self.global_aperture_thresh   # 0.7   # inter-modal
                _local_aperture_thresh = self.local_aperture_thresh     # 1.2   # intra-modal

                text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0) #.mean()
                text_image_entailment_loss = text_image_entailment_loss.mean()
                
                if self.use_boxes:
                    box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0) #.mean()
                    cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0) #.mean()
                    cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0) #.mean()

                    box_text_image_entailment_loss = box_text_image_entailment_loss.mean()
                    cross_image_entailment_loss = cross_image_entailment_loss.mean()
                    cross_text_entailment_loss = cross_text_entailment_loss.mean()

                entailment_loss = (0.5 if self.use_boxes else 1.0) * (
                    text_image_entailment_loss
                    + (box_text_image_entailment_loss if self.use_boxes else 0)
                    + (cross_image_entailment_loss if self.use_boxes else 0)
                    + (cross_text_entailment_loss if self.use_boxes else 0)
                )

            loss = 0
            if self.contrastive_weight > 0:
                contrastive_loss = self.contrastive_weight * contrastive_loss
                loss = loss + contrastive_loss
            if self.entail_weight > 0:
                entailment_loss = self.entail_weight * entailment_loss
                loss = loss + entailment_loss

        final_dict = {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss if self.contrastive_weight > 0 else None,
                "text_image_entailment_loss": text_image_entailment_loss if self.entail_weight > 0 else None,
                "box_text_image_entailment_loss": box_text_image_entailment_loss if (self.entail_weight > 0 and self.use_boxes) else None,
                "cross_image_entailment_loss": cross_image_entailment_loss if (self.entail_weight > 0 and self.use_boxes) else None,
                "cross_text_entailment_loss": cross_text_entailment_loss if (self.entail_weight > 0 and self.use_boxes) else None,
                "entailment_loss": entailment_loss if self.entail_weight > 0 else None,
                "logit_scale": _scale if self.contrastive_weight > 0 else None,
                "curv": _curv,
                "visual_alpha": self.visual_alpha.exp().item(),
                "textual_alpha": self.textual_alpha.exp().item(),
            },
        }

        return final_dict