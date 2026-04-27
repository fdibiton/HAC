#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Each config file should have four dicts or OmegaConf objects:
`dataset`, `model`, `optim`, and `train`.

User can compose config files by importing these objects and overriding specific
parameters. See examples in other training configs.

Reference: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
"""

from torch.optim import AdamW
from peft import LoraConfig

from hac.config import LazyCall as L
from hac.data.webdataset_mapper import GroundedDatasetTarMapper, ImageTextWebDataset
from hac.encoders.image_encoders import build_timm_vit
from hac.encoders.text_encoders import TransformerTextEncoder
from hac.models import AdaptedCLIP
from hac.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param
from hac.utils.model_utils import generate_lora_param_names


dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/train/GRIT/train/*.tar"],
    mapper=L(GroundedDatasetTarMapper)(
        image_transform="custom",
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)

model = L(AdaptedCLIP)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77, noise_alpha=0.1 # originally context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
    use_boxes=True,
    checkpoint="checkpoints/clip_vit_b.pth",
    init_proj = True,
    init_final_ln = True,
)

# AdamW with no weight decay for norm, bias, and other learnable scalars.
optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=[
                "logit_scale", "visual_alpha", "textual_alpha", "curv",
                "lora_" # no wd on LoRA
            ],
        ),
        lr=2.5e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        total_steps="${...train.num_iterations}", warmup_steps=4000
    ),
)


# Other parameters useful for training script.
train = dict(
    seed=0,
    amp=True,
    total_batch_size=768,
    num_iterations=30000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, static_graph=True, find_unused_parameters=True
    ),
    ddp_fp16_compression=True,
)

lora_config = L(LoraConfig)(
    r=128,
    lora_alpha=128,
    target_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"],
    lora_dropout=0.1,
    bias="none",
    exclude_modules=generate_lora_param_names(visual_blocks=list(range(0,8)),  # visual blocks 0-7
        textual_blocks=list(range(0,4)), # textual blocks 0-3
        separate_qkv=True,
        components=["attn", "proj", "ffn"]  # components to exclude from LoRA
    ),
    use_rslora = True
)

