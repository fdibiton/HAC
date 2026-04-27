import torch.nn as nn
import torch
import torch.nn.functional as F
from loguru import logger

from timm.models.vision_transformer import Attention


# from https://github.com/huggingface/pytorch-image-models/blob/81900a6bae9a1a14a5b656ad01fd33f8a459dacf/timm/models/vision_transformer.py#L58
class PlainTimmAttention(nn.Module):
    """Drop-in replacement for timm Attention with split qkv"""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        proj_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # BS, num_tokens, embed_dim
        src_N = N
        if hasattr(self.k_proj, "prefix_length"):
            src_N += self.k_proj.prefix_length

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, src_N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, src_N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    # https://github.com/mc-lan/ClearCLIP/blob/ad68a404d55d48d27330b93554eb64a234ff717f/open_clip/transformer.py#L589
    def custom_attn(self, x, model_type='ClearCLIP'):

        B, N, C = x.shape # BS, num_tokens, embed_dim
        src_N = N

        #q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        #q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1) # -> [B*num_heads, N, head_dim]
        #k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        #v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # -> [B, num_heads, N, head_dim]
        k = self.k_proj(x).reshape(B, src_N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, src_N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q = q.contiguous().view(B*self.num_heads, N, self.head_dim)     # -> [B*num_heads, N, head_dim]
        k = k.contiguous().view(B*self.num_heads, src_N, self.head_dim) # -> [B*num_heads, src_N, head_dim]
        v = v.contiguous().view(B*self.num_heads, src_N, self.head_dim) # -> [B*num_heads, src_N, head_dim]

        if model_type == 'vanilla':
            qk_attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(qk_attn, dim=-1) # -> [B*num_heads, N, src_N]
        elif model_type == 'MaskCLIP':
            mask = torch.empty(q.shape[1], q.shape[1], dtype=q.dtype).to(q.device)
            mask.fill_(float('-inf'))
            mask.fill_diagonal_(0)
            mask = mask.unsqueeze(0).repeat(q.shape[0], 1, 1)
            attn_weights = F.softmax(mask, dim=-1)
        elif model_type == 'SCLIP':
            qq_attn = torch.bmm(q, q.transpose(1, 2)) * self.scale
            kk_attn = torch.bmm(k, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(qq_attn, dim=-1) + F.softmax(kk_attn, dim=-1)
        elif model_type == 'ClearCLIP':
            qq_attn = torch.bmm(q, q.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(qq_attn, dim=-1)

        x = torch.bmm(attn_weights, v) # -> [B*num_heads, N, head_dim]
        x = x.view(B, self.num_heads, N, self.head_dim).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_parameters(self, timm_module: nn.Module):
        """Copy weights from timm.models.vision_transformer.Attention"""
        assert hasattr(timm_module, 'qkv')
        assert hasattr(timm_module, 'proj')

        q_weight, k_weight, v_weight = timm_module.qkv.weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = timm_module.qkv.bias.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q_weight)
        self.k_proj.weight.data.copy_(k_weight)
        self.v_proj.weight.data.copy_(v_weight)
        self.q_proj.bias.data.copy_(q_bias)
        self.k_proj.bias.data.copy_(k_bias)
        self.v_proj.bias.data.copy_(v_bias)

        self.proj.weight.data.copy_(timm_module.proj.weight)
        self.proj.bias.data.copy_(timm_module.proj.bias) 

# https://github.com/KyanChen/MakeMultiHeadNaive/blob/master/main.py
class PlainMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            num_heads=16,
            dropout=0.,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False,
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def init_weights(self):
        pass

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True, # must stay for compatibility
            attn_mask=None,
            average_attn_weights=True, # must stay for compatibility
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        
        # https://github.com/pytorch/pytorch/blob/392fa75411a1f293e891395f005615b257c03eda/torch/nn/modules/activation.py#L1223
        is_batched = query.dim() == 3
        
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )
        assert key_padding_mask is None, "key_padding_mask is not supported in PlainMultiHeadAttention"

        # BS, seq, embed -> seq, BS, embed
        query, key, value = self.batch_first_to_seq_first(query, key, value, is_batched)

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        if hasattr(self.k_proj, "prefix_length"):
            prefix_len = self.k_proj.prefix_length
            src_len += prefix_len # BS, seq_len + prefix_length, embed_dim
        else:
            prefix_len = 0

        # separate q, k, v
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # https://github.com/pytorch/pytorch/blob/392fa75411a1f293e891395f005615b257c03eda/torch/nn/modules/activation.py#L1233
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )
        attn_mask = self.update_attn_mask_size(attn_mask, bsz, tgt_len, src_len, prefix_len=prefix_len)

        dropout_p = self.dropout if self.training else 0.

        # seq_len, BS * num_heads, head_dim -> BS * num_heads, seq_len, head_dim
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        # BS * num_heads, seq_len, head_dim -> BS, num_heads, seq_len, head_dim
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        # seq_len, BS, num_heads, head_dim -> BS * seq_len, num_heads, head_dim
        #attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        #attn_output = self.proj(attn_output)
        #attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        
        # --- FIX: keep 3D through proj so adapters see (B, L, C) ---
        attn_output = attn_output.transpose(1, 2).contiguous() # BS, num_heads, seq_len, head_dim -> BS, seq_len, num_heads, head_dim
        attn_output = attn_output.view(bsz, tgt_len, embed_dim) # BS, seq_len, num_heads * head_dim
        attn_output = self.proj(attn_output)  # BS, seq_len, embed_dim

        # https://github.com/pytorch/pytorch/blob/392fa75411a1f293e891395f005615b257c03eda/torch/nn/modules/activation.py#L1401
        if self.batch_first and is_batched:
            #return attn_output.transpose(1, 0), None
            return attn_output, None # (B, L, C)
        
        return attn_output.transpose(0, 1).contiguous(), None # (L, B, C)
    
    
    # https://github.com/pytorch/pytorch/blob/392fa75411a1f293e891395f005615b257c03eda/torch/nn/modules/activation.py#L1342
    def batch_first_to_seq_first(self, query, key, value, is_batched):
        
        if self.batch_first and is_batched:
            if key is value: # can assume always False
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else: # always True
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        return query, key, value
    
    
    def extend_causal_attn_mask(self, attn_mask: torch.Tensor, tgt_len: int, src_len: int):
        device = attn_mask.device
        prefix_len = src_len - tgt_len
        new_attn_mask_2d = torch.zeros((tgt_len, src_len), dtype=attn_mask.dtype, device=device)
        original_attn_mask_2d = attn_mask[..., :, :]
        
        # Fill prefix area (first `prefix_len` columns): all zero = fully visible
        new_attn_mask_2d[:, :prefix_len] = 0
        new_attn_mask_2d[:, prefix_len:] = original_attn_mask_2d

        assert torch.all(new_attn_mask_2d[:, src_len-tgt_len:] == original_attn_mask_2d), \
            "The right part of the new mask should be same as the original mask"

        return new_attn_mask_2d  # tgt_len, src_len
    

    def update_attn_mask_size(self, attn_mask: torch.Tensor, bsz:int, tgt_len: int, src_len: int, prefix_len: int):
        if attn_mask is not None:
            # extend mask if prefix length is greater than 0
            if prefix_len > 0:
                attn_mask = self.extend_causal_attn_mask(attn_mask, tgt_len, src_len)
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0) # 1, 1, tgt_len, src_len
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        return attn_mask


    def set_parameters(self, torch_tgt_module: nn.Module):
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim

        # separate q, k, v
        q_weight, k_weight, v_weight = torch_tgt_module.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = torch_tgt_module.in_proj_bias.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q_weight)
        self.k_proj.weight.data.copy_(k_weight)
        self.v_proj.weight.data.copy_(v_weight)
        self.q_proj.bias.data.copy_(q_bias)
        self.k_proj.bias.data.copy_(k_bias)
        self.v_proj.bias.data.copy_(v_bias)

        self.proj.weight.data.copy_(torch_tgt_module.out_proj.weight.data)
        self.proj.bias.data.copy_(torch_tgt_module.out_proj.bias.data)


def replace_mha_with_plain(model: nn.Module, device=None):
    plain_mha_cnt = 0
    plain_timm_mha_cnt = 0
    from_timm = False
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            plain_mha = PlainMultiHeadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                kdim=module.kdim,
                vdim=module.vdim,
                batch_first=module.batch_first,
            )
            plain_mha.set_parameters(module)
            parent = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent, name.rsplit('.', 1)[-1], plain_mha)
            plain_mha_cnt += 1
        elif isinstance(module, Attention):
            from_timm = True
            plain_mha = PlainTimmAttention(
                dim=module.qkv.in_features,
                num_heads=module.num_heads,
                qkv_bias=module.qkv.bias is not None,
                qk_norm=not isinstance(module.q_norm, nn.Identity),
                proj_bias=module.proj.bias is not None,
                attn_drop=module.attn_drop.p,
                proj_drop=module.proj_drop.p,
                norm_layer=type(module.q_norm)
            )
            plain_mha.set_parameters(module)
            parent = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent, name.rsplit('.', 1)[-1], plain_mha)
            plain_timm_mha_cnt += 1
        elif (hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")) or hasattr(module, "qkv") or module.__class__.__name__ == "Attention":
            # This is a not-supported Attention module
            raise AssertionError(
                f"Module {name} is not a supported MultiheadAttention or Timm Attention module"
            )
            
    # move all modules to the same device
    if device is not None:
        for name, module in model.named_modules():
            if isinstance(module, (PlainMultiHeadAttention, PlainTimmAttention)):
                module.to(device)
            
    assert plain_mha_cnt > 0 or plain_timm_mha_cnt > 0, "No MultiheadAttention to substitute found in the model"
    if plain_mha_cnt > 0:
        logger.info(f"Replaced {plain_mha_cnt} nn.MultiheadAttention modules with PlainMultiHeadAttention")
    if plain_timm_mha_cnt > 0:
        logger.info(f"Replaced {plain_timm_mha_cnt} Timm Attention modules with PlainTimmAttention")    
