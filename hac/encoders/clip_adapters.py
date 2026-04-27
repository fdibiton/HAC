import torch
import torch.nn as nn
from loguru import logger
import adapters
from copy import deepcopy

from adapters import AdapterModelInterface
from adapters import DoubleSeqBnConfig, SeqBnConfig, ParBnConfig, AdapterPlusConfig
from transformers.models.clip import CLIPTextConfig, CLIPVisionConfig


##################################
#                                #
#   ADAPTERS LIBRARY INTERFACE   #
#                                #
##################################
        
        
def get_plugin_interface(encoder_type, adapter_methods: list, separate_qkv: bool = False):
    """
    Get the plugin interface for custom models.
    https://docs.adapterhub.ml/plugin_interface.html
    
    Args:
        encoder_type: The encoder_type of the component, e.g., "visual", "textual"
        adapter_methods: List of adapter methods to be used, e.g., ["bottleneck", "lora", "reft"]
        separate_qkv: Whether to separate Q, K, V in the interface.
        
    Returns:
        plugin_interface (AdapterModelInterface): The plugin interface for the model.
    """
    if encoder_type == "visual":
        kwargs ={
            "adapter_methods": adapter_methods,
            "model_embeddings": "patch_embed",
            "model_layers": "blocks",
            "layer_self_attn": "attn",
            "layer_cross_attn": None,
            "attn_qkv_proj": "qkv",                 # combined Q, K, V projection
            "attn_o_proj": "proj",
            "layer_intermediate_proj": "mlp.fc1",   # up projection in MLP
            "layer_output_proj": "mlp.fc2",         # down projection in MLP
            "layer_pre_self_attn": "norm1",         # pre-attention layer norm
            "layer_pre_cross_attn": None,
            "layer_pre_ffn": "norm2",               # pre-feed-forward layer norm (none)        
            "layer_ln_1": None,                     # post-attention layer norm (none)
            "layer_ln_2": None,                     # post-feed-forward layer norm (none) 
        }
    elif encoder_type == "textual":
        kwargs ={
            "adapter_methods": adapter_methods,
            "model_embeddings": "token_embed",
            "model_layers": "resblocks",
            "layer_self_attn": "attn",
            "layer_cross_attn": None,
            "attn_qkv_proj": "qkv",                 # combined Q, K, V projection
            "attn_o_proj": "proj",
            "layer_intermediate_proj": "mlp.c_fc",  # up projection in MLP
            "layer_output_proj": "mlp.c_proj",      # down projection in MLP
            "layer_pre_self_attn": "ln_1",          # pre-attention layer norm
            "layer_pre_cross_attn": None,
            "layer_pre_ffn": "ln_2",                # pre-feed-forward layer norm (none)        
            "layer_ln_1": None,                     # post-attention layer norm (none)
            "layer_ln_2": None,                     # post-feed-forward layer norm (none) 
        }
    if separate_qkv:
        kwargs["attn_q_proj"] = "q_proj"
        kwargs["attn_k_proj"] = "k_proj"
        kwargs["attn_v_proj"] = "v_proj"
        del kwargs["attn_qkv_proj"]
            
    plugin_interface = AdapterModelInterface(**kwargs)
    
    return plugin_interface


def init_encoder_with_adapter(encoder, encoder_type, adapter_methods, adapter_config, separate_qkv: bool = False):
    interface = get_plugin_interface(encoder_type, adapter_methods, separate_qkv)
    adapters.init(encoder, interface=interface)
    # Add a new adapter
    encoder.add_adapter(encoder_type, adapter_config, set_active=True) 
    # Activate the adapter
    encoder.train_adapter(encoder_type)
    # print active adapters
    assert encoder.active_adapters, f"No active adapters found in {encoder_type} encoder. Please check the adapter initialization."
    logger.info(f"Active adapters in {encoder_type} encoder: {encoder.active_adapters}")
    
    return encoder


def get_bottleneck_config(config_type):
    """Adapters comes with pre-defined configurations for some bottleneck adapter architectures proposed in literature:
    - DoubleSeqBnConfig, as proposed by Houlsby et al. (2019) places adapter layers after both the multi-head attention and feed-forward block in each Transformer layer.
    - SeqBnConfig, as proposed by Pfeiffer et al. (2020) places an adapter layer only after the feed-forward block in each Transformer layer.
    - ParBnConfig, as proposed by He et al. (2021) places adapter layers in parallel to the original Transformer layers.
    - AdapterPlusConfig, as proposed by Steitz and Roth (2024) places adapter layers adapter layers after the multi-head attention and has channel wise scaling and houlsby weight initialization 
    """
    
    # Houlsby et al. (2019)
    if config_type == "double_seq_bn":
        config = DoubleSeqBnConfig(non_linearity="gelu")
    # Pfeiffer et al. (2020) 
    elif config_type == "seq_bn":
        config = SeqBnConfig(non_linearity="gelu")
    # He et al. (2021)
    elif config_type == "par_bn":
        config = ParBnConfig(non_linearity="gelu")
    # He et al. (2021)
    elif config_type == "scaled_par_bn":
        config = ParBnConfig(non_linearity="gelu", scaling="learned")
    # Steitz and Roth (2024)
    elif config_type == "adapter_plus":
        # Setting original_ln_after=False in bottleneck adapter is not supported
        config = AdapterPlusConfig(non_linearity="gelu", original_ln_after=True)
        
    return config


class VisionTransformerWrapper(nn.Module):
    base_model_prefix = "base_model"

    def __init__(self, encoder: nn.Module, config=None):
        if config is None:
            config = CLIPVisionConfig()
        super().__init__()
        self.base_model = encoder
        self.config = config
        encoder.config = config
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
        encoder.get_input_embeddings = lambda: encoder.patch_embed
        object.__setattr__(encoder, "base_model", self)
        self.support_prompt_tuning = False
        self.get_output_embeddings = lambda: None
        # ensure adapter-transformers sees a .device attribute
        encoder.__class__.device = property(lambda self: next(self.parameters()).device)
        
    def forward(self, x):
        return self.base_model(x)
        
    
class TextTransformerWrapper(nn.Module):
    base_model_prefix = "base_model"
    
    def __init__(self, encoder: nn.Module, config=None):
        if config is None:
            config = CLIPTextConfig()
        super().__init__()
        self.base_model = encoder
        self.config = config
        encoder.config = config
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
        encoder.get_input_embeddings = lambda: encoder.token_embed
        object.__setattr__(encoder, "base_model", self)
        self.support_prompt_tuning = False
        self.get_output_embeddings = lambda: None
        # ensure adapter-transformers sees a .device attribute
        encoder.__class__.device = property(lambda self: next(self.parameters()).device)
        
    def forward(self, x):
        return self.base_model(x)
    
ADAPTER_TYPES = {
    "DoubleSeqBnConfig": "bottleneck",
    "SeqBnConfig": "bottleneck",
    "ParBnConfig": "bottleneck",
    "AdapterPlusConfig": "bottleneck",
    "LoRAConfig": "lora",
    "VeraConfig": "lora",
    "IA3Config": "lora",
    "ReftConfig": "reft",
    "NoreftConfig": "reft",
    "DiReftConfig": "reft"
}
    
def get_adapted_encoder(encoder, encoder_type, encoder_config, adapter_config, separate_qkv=True):
    encoder_wrap_func = {
        "visual": VisionTransformerWrapper, 
        "textual": TextTransformerWrapper
        }.get(encoder_type)
    encoder_wrapped = encoder_wrap_func(encoder, config=encoder_config)
    adapter_type = ADAPTER_TYPES[adapter_config.__class__.__name__]
    # init encoder with adapter
    adapted_encoder = init_encoder_with_adapter(
        encoder_wrapped, 
        encoder_type, 
        [adapter_type], 
        adapter_config, 
        separate_qkv,
    )
    logger.info(f"Initialized {encoder_type} encoder with {adapter_config} adapter configuration.")
    logger.info("\n" + adapted_encoder.adapter_summary())
    
    return encoder # encoder changed by referece
            

if __name__ == "__main__":
    # test all adapters work correctly on AdaptedCLIP model
    from hac.models import AdaptedCLIP
    from hac.encoders.image_encoders import build_timm_vit
    from hac.encoders.text_encoders import TransformerTextEncoder
    from hac.utils.plain_mha import replace_mha_with_plain
    
    # get model
    model = AdaptedCLIP(
        visual=build_timm_vit(
            arch="vit_small_mocov3_patch16_224",
            global_pool="token",
            use_sincos2d_pos=True,
        ),
        textual=TransformerTextEncoder(
            arch="L12_W512", vocab_size=49408, context_length=77 # originally context_length=77
        ),
        visual_adapter=None,
        textual_adapter=None,
        embed_dim=512,
        curv_init=1.0,
        learn_curv=True,
        entail_weight=0.2,
        use_boxes=True,
        checkpoint="checkpoints/clip_vit_s.pth"    
    )
    
    VISION_CONFIG = CLIPVisionConfig(
        hidden_size=384,
        intermediate_size=1536,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu",
    )
    
    TEXTUAL_CONFIG = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="gelu",
        # This differs from `CLIPTokenizer`'s default and from openai/clip
        # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
    )
    
    CONFIG_DICT = {
        "bottleneck": ["double_seq_bn", "seq_bn", "par_bn", "scaled_par_bn", "adapter_plus"],
        "lora": ["lora", "ia3"],
        "reft": ["loreft", "noreft", "direft"]
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for SEPARATE_QKV in [True, False]:
        logger.info(f"Testing with separate Q, K, V: {SEPARATE_QKV}")
        
        for config_group in ["bottleneck", "lora", "reft"]:
            
            if not SEPARATE_QKV and config_group == "lora":
                logger.warning("Skipping lora configs with SEPARATE_QKV=False, as it requires separate Q, K, V.")
                continue
            
            configs = CONFIG_DICT[config_group]
            logger.info(f"Testing {config_group} adapter configurations: {configs}")
        
            for config in configs:
                logger.info(f"Testing {config} adapter configuration.")
                adapter_methods = [config_group]
                # create a copy of the model
                adapted_model = deepcopy(model)
                # init model with adapter
                for encoder_type in ["visual", "textual"]:
                    # replace MHA with plain MHA
                    encoder = adapted_model.visual if encoder_type == "visual" else adapted_model.textual
                    replace_mha_with_plain(encoder, separate_qkv=SEPARATE_QKV, device=device)
                    # get adapter config
                    if config_group == "bottleneck":
                        adapter_config = get_bottleneck_config(config)
                    else:
                        raise ValueError(f"Unknown adapter group: {config_group}")
                    # wrap encoder in a PreTrainedModel
                    if encoder_type == "visual":
                        encoder_wrapped = VisionTransformerWrapper(encoder, config=VISION_CONFIG)
                    else:
                        encoder_wrapped = TextTransformerWrapper(encoder, config=TEXTUAL_CONFIG)
                    # init encoder with adapter
                    adapted_encoder = init_encoder_with_adapter(
                        encoder_wrapped, encoder_type, adapter_methods, adapter_config, separate_qkv=SEPARATE_QKV
                    )
                    logger.info(f"Initialized {encoder_type} encoder with {config} adapter configuration.")
                    logger.info("\n" + adapted_encoder.adapter_summary())
                # move to device
                adapted_model.to(device)

                # test forward pass
                dummy_image = torch.rand(2, 3, 224, 224).to(device)
                dummy_text = torch.randint(0, 1000, (2, 77)).to(device)
                
                # check adapter modules are used in forward pass
                def ping_visual(*_): logger.info("►  visual adapter fired")
                def ping_textual(*_): logger.info("► textual adapter fired")
                layer0_visual = adapted_model.visual.get_adapter("visual")[0]
                layer0_visual = next(
                    layer0_visual[k]                     # first existing slot
                    for k in ("mh_adapter", "output_adapter", "selfattn_lora", "output_reft")
                    if k in layer0_visual
                )
                layer0_textual = adapted_model.textual.get_adapter("textual")[0]
                layer0_textual = next(
                    layer0_textual[k]                     # first existing slot
                    for k in ("mh_adapter", "output_adapter", "selfattn_lora", "output_reft")
                    if k in layer0_textual
                )
                handle1 = layer0_visual.register_forward_hook(ping_visual)
                handle2 = layer0_textual.register_forward_hook(ping_textual)
                
                with torch.no_grad():
                    _ = adapted_model.encode_image(dummy_image, project=True)
                    _ = adapted_model.encode_text(dummy_text, project=True)
                logger.info(f"Forward pass successful for {config} adapter configuration.")
                
                # remove hooks
                handle1.remove()
                handle2.remove()
                
                # check gradients
                # 1) enable grads only on adapters
                adapted_model.train()                     # be sure we're in train mode
                adapted_model.zero_grad()
                
                img_out  = adapted_model.encode_image(dummy_image,  project=True).sum()
                txt_out  = adapted_model.encode_text (dummy_text ,  project=True).sum()
                (img_out + txt_out).backward()
                
                # 2) check grads
                non_zero_adapt = 0     # adapter params that received grad
                non_zero_backb = 0     # backbone params that mistakenly received grad

                for name, p in adapted_model.named_parameters():
                    if not p.requires_grad:          # frozen backbone param
                        if p.grad is not None and p.grad.abs().sum() != 0:
                            non_zero_backb += 1
                    else:                            # trainable adapter param
                        if p.grad is not None and p.grad.abs().sum() != 0:
                            non_zero_adapt += 1

                logger.info(f"grad-check → adapters:{non_zero_adapt}  backbone:{non_zero_backb}")
                assert non_zero_backb == 0, "Backbone unexpectedly got gradients!"
    