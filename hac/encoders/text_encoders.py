#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from __future__ import annotations

import re
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def NEFTune(tokens, embeds_init, noise_alpha=5):
    # identify padding to exclude from noise addition
    pad_id = 0
    padding_mask = (tokens != pad_id) # (batch_size, seq_len)
    L = padding_mask.sum(dim=1) # (batch_size,)
    d = embeds_init.size(2)
    
    noise = torch.zeros_like(embeds_init).uniform_(-1,1) # sample noise vector, batch_size, seq_len, embed_dim
    # compute scaling factor
    scaling_factor = noise_alpha / torch.sqrt(L * d) # seq_len, embed_dim
    # compute scaled noise
    scaled_noise = noise * scaling_factor.view(-1, 1, 1).detach() # batch_size, 1, 1
    # mask out padding positions
    scaled_noise = scaled_noise * padding_mask.unsqueeze(-1).detach() # batch_size, seq_len (padded), embed_dim 
    # add noise to the original embedding
    noised_embed = embeds_init + scaled_noise
    return noised_embed


class _TransformerBlock(nn.Module):
    """
    Single transformer block comprising multi-head self-attention and MLP. Both
    modules are preceeding by layer normalization. This module is same as PyTorch
    builtin module `TransformerEncoderLayer` with arguments as
    (`norm_first=True, dropout=0, activation="gelu"`).

    We adapt this module from CLIP to easily load checkpoints of CLIP and other
    works that build on CLIP's code. Reference: https://github.com/openai/clip
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, lx, need_weights=False, attn_mask=attn_mask)[0]
        x = x + ax
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerTextEncoder(nn.Module):
    """
    Text encoder using multiple layers of transformer encoder blocks. It accepts
    tokenized text sequences, passes them through word/position embedding layers
    and further processes them through transformer layers.

    All transformer blocks are unidirectional "Pre-LN" variants by default:
    LayerNorm is placed before attention/MLP layers inside the residual block,
    and future positions are masked while computing self-attention.
    """

    def __init__(
        self,
        arch: str,
        vocab_size: int,
        context_length: int,
        grad_checkpointing: bool = False,
        text_learnable_tokens: int = 0,
        text_box_learnable_tokens: int = 0,
        learnable_tokens_pos: str = "front", # "front", "mid" "end"
        noise_alpha: int = 0, # if > 0, use NEFTune
    ):
        """
        Args:
            arch: Architecture config for transformer, describing layers, width,
                and number of attention heads. For example, `L12_W512_A8` has 1
                layer, 512 width, 8 heads. Width of MLP will always be `4 * W`,
                per transformer paper. `A` is optional and will default to
                (`A = H/64`) per transformer paper.
            vocab_size: Number of tokens in the output vocabulary.
            context_length: Maximum length of input captions; this is used to
                create a fixed positional embedding lookup table.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.grad_checkpointing = grad_checkpointing

        # Parse architecture str: layers, width, heads, feed-forward size.
        self.layers = int(re.search(r"L(\d+)", arch).group(1))
        self.width = int(re.search(r"W(\d+)", arch).group(1))

        # Find heads in architecture else use (H // 64) per (Vaswani et al.)
        _attn = re.search(r"A(\d+)", arch)
        self.heads = int(_attn.group(1)) if _attn else self.width // 64

        # Input sequences in forward pass will be right padded with zeroes.
        # `nn.Embedding` has a `padding_idx` argument to set their embedding as
        # zero. However, since the blocks are uni-directional, they will never
        # receive gradients for padded positions.
        self.token_embed = nn.Embedding(vocab_size, self.width)
        self.posit_embed = nn.Parameter(torch.empty(context_length, self.width))

        # Make a sequential module of transformer encoder blocks.
        _resblocks = [
            _TransformerBlock(self.width, self.heads) for _ in range(self.layers)
        ]
        self.resblocks = nn.ModuleList(_resblocks)
        self.ln_final = nn.LayerNorm(self.width)

        # Generate a unidirectional mask for self-attention. As per PyTorch API,
        # masked positions are set to `-inf`.
        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())

        # Initialize all modules like CLIP:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.posit_embed.data, std=0.01)

        out_proj_std = (2 * self.width * self.layers) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=self.width**-0.5)
            nn.init.normal_(block.attn.out_proj.weight, std=out_proj_std)
            nn.init.normal_(block.mlp[0].weight, std=(2 * self.width) ** -0.5)
            nn.init.normal_(block.mlp[2].weight, std=out_proj_std)
            
        self.learnable_tokens_pos = learnable_tokens_pos
            
        if text_learnable_tokens > 0:
            self.text_learnable_tokens = nn.Parameter(torch.zeros(text_learnable_tokens, self.width))
            nn.init.normal_(self.text_learnable_tokens, std=0.02)
        else:
            self.text_learnable_tokens = None

        if text_box_learnable_tokens > 0:
            self.text_box_learnable_tokens = nn.Parameter(torch.zeros(text_box_learnable_tokens, self.width))
            nn.init.normal_(self.text_box_learnable_tokens, std=0.02)
        else:
            self.text_box_learnable_tokens = None
            
        self.noise_alpha = noise_alpha
            
    def get_text_learnable_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.text_learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1) # num_l_tokens, embed_dim -> batch_size, num_l_tokens, embed_dim
    
    def get_text_box_learnable_tokens(self, batch_size: int) -> torch.Tensor | None:
        return self.text_box_learnable_tokens.unsqueeze(0).expand(batch_size, -1, -1) # num_l_tokens, embed_dim -> batch_size, num_l_tokens, embed_dim

    def get_num_learnable_tokens(self) -> int:
        return self.text_learnable_tokens.shape[0] if self.text_learnable_tokens is not None else 0

    def embed_tokens(self, text_tokens, eot_indexes: list[int] = None) -> torch.Tensor:
        """
        Returns embedded sequence. If learnable prompts are enabled, we
        concatenate them to the embedded token sequence (soft prompts).
        """
        token_emb = self.token_embed(text_tokens)  # shape: (batch_size, seq_length, width)
        batch_size = text_tokens.shape[0]
        
        if self.text_learnable_tokens is not None:
            text_learnable_tokens = self.get_text_learnable_tokens(batch_size).to(token_emb.dtype).to(token_emb.device)
            # prepend text prompts
            if self.learnable_tokens_pos == "front":
                token_emb = torch.cat([text_learnable_tokens, token_emb], dim=1)
            elif self.learnable_tokens_pos == "mid":
                # place token in the middle of the sequence
                mid_indexes =  [eot_idx // 2 for eot_idx in eot_indexes]
                chunks = []
                for i in range(batch_size):
                    mid_idx = int(mid_indexes[i])
                    chunks.append(torch.cat([token_emb[i, :mid_idx, :], text_learnable_tokens[i], token_emb[i, mid_idx:, :]], dim=0))
                token_emb = torch.stack(chunks, dim=0)
            elif self.learnable_tokens_pos == "back":
                back_indexes = eot_indexes
                chunks = []
                for i in range(batch_size):
                    back_idx = int(back_indexes[i])
                    chunks.append(torch.cat([token_emb[i, :back_idx, :], text_learnable_tokens[i], token_emb[i, back_idx:, :]], dim=0))
                token_emb = torch.stack(chunks, dim=0)

        return token_emb

    def forward(self, text_tokens: torch.Tensor, eot_indexes: list[int] = None) -> torch.Tensor:
        """
        Obtain features of input text tokens by passing them through transformer
        blocks. All self-attention layers only attend to past token (left side).

        text_tokens: Input text tokens of shape (batch_size, seq_length).
        """
        max_len = text_tokens.shape[-1]
        # update max_len to include learnable tokens
        K = self.get_num_learnable_tokens()
        max_len = max_len + K
        assert (max_len <= self.context_length), f"Input text length {max_len} exceeds model context length {self.context_length}."
        _posit_embed = self.posit_embed[:max_len, :]
        _attn_mask = self.attn_mask[:max_len, :max_len]

        # shape: (batch_size, context_length, width)
        token_embeddings = self.embed_tokens(text_tokens, eot_indexes=eot_indexes)
        token_embeddings = token_embeddings + _posit_embed # shape: (batch_size, seq_length, width)
        
        if self.noise_alpha > 0 and self.training:
            token_embeddings = NEFTune(text_tokens, token_embeddings, noise_alpha=self.noise_alpha)

        # Forward pass through transformer, optionally with grad checkpointing.
        textual_features = token_embeddings
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                # shape: (batch_size, context_length, width)
                textual_features = checkpoint(block, textual_features, _attn_mask)
            else:
                textual_features = block(textual_features, _attn_mask)

        textual_features = self.ln_final(textual_features)
        return textual_features