import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip.transformer import text_global_pool
from copy import deepcopy


def safe_clone(module: nn.Module):
    clone = deepcopy(module)
    clone.load_state_dict(module.state_dict())
    return clone


class CLIPVisualEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: str):
        super(CLIPVisualEncoder, self).__init__()
        clip_model = get_clip_model(
            model_name=model_name,
            pretrained=pretrained,
            device="cpu",
        )
        self.model = safe_clone(clip_model.visual)
        if hasattr(self.model, "proj"):
            self.model.proj = None
        self.width = self.model.transformer.width

        del clip_model

    def get_width(self):
        return self.width
    
    # https://github.com/mlfoundations/open_clip/blob/bf5d49c112c82c738f7b34bde6e154760a711790/src/open_clip/model.py#L278
    def forward(self, image, normalize = False):
        features = self.model(image)
        return F.normalize(features, dim=-1) if normalize else features


class CLIPTextualEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: str):
        super(CLIPTextualEncoder, self).__init__()
        clip_model = get_clip_model(
            model_name=model_name,
            pretrained=pretrained,
            device="cpu",
        )
        self.model = safe_clone(clip_model.transformer)
        self.token_embedding = safe_clone(clip_model.token_embedding)
        self.positional_embedding = nn.Parameter(clip_model.positional_embedding.detach().clone())
        self.ln_final = safe_clone(clip_model.ln_final)
        self.text_pool_type = clip_model.text_pool_type # str
        self.width = self.model.width # int
        self.register_buffer("attn_mask", clip_model.attn_mask.clone())

        del clip_model

    def get_width(self):
        return self.width

    def cast_buffers(self, dtype, buffer_names): # for attn_mask
        """
        Cast all buffers of the model to a given dtype.
        """
        for name, buffer in self.named_buffers():
            if buffer is not None and name in buffer_names:
                setattr(self, name, buffer.to(dtype))

    def to(self, *args, **kwargs):
        """
        Override the to method to cast all buffers to the given dtype.
        """
        super().to(*args, **kwargs)
        if "dtype" in kwargs:
            self.cast_buffers(kwargs["dtype"], ["attn_mask"])
        return self
    
    # https://github.com/mlfoundations/open_clip/blob/bf5d49c112c82c738f7b34bde6e154760a711790/src/open_clip/model.py#L282
    def forward(self, text, normalize = False):
        cast_dtype = self.model.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.model(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = text_global_pool(x, text, self.text_pool_type) # take EOS token

        return F.normalize(x, dim=-1) if normalize else x


def get_clip_model(
    model_name: str, # e.g., "ViT-B-32"
    pretrained: str, # e.g., "laion2b_s34b_b79k",
    device: str = "cpu",
):
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    
    return model


def build_clip_visual_encoder(
    model_name: str, # e.g., "ViT-B-32"
    pretrained: str, # e.g., "laion2b_s34b_b79k",
):
    visual_encoder = CLIPVisualEncoder(
        model_name=model_name,
        pretrained=pretrained,
    )

    return visual_encoder


def build_clip_textual_encoder(
    model_name: str, # e.g., "ViT-B-32"
    pretrained: str, # e.g., "laion2b_s34b_b79k",
):
    textual_encoder = CLIPTextualEncoder(
        model_name=model_name,
        pretrained=pretrained,
    )

    return textual_encoder
    

def build_clip_tokenizer(
    model_name: str, # e.g., "ViT-B-32"
):
    tokenizer = open_clip.get_tokenizer(model_name)
    
    return tokenizer