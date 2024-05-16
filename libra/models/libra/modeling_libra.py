import math
import json
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import ModelOutput, logging

from libra.models.llama.modeling_llama import LlamaMLP, LlamaAttention, LlamaRMSNorm, _make_causal_mask, _expand_mask
from libra.models.libra.configuration_libra import LibraConfig
from libra.models.libra.tokenization_libra import LibraTokenizer
from libra.models.libra.modeling_libra_utils  import BaseLibraPreTrainedModel
from libra.common.registry import registry

logger = logging.get_logger(__name__)

def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.flatten(0,1)[indices.view(-1)].view(B, N_new, C)
    return out

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, use_2d_rope=False):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    if use_2d_rope:
        assert len(position_ids.shape) == 3 and position_ids.shape[1] == 2
        cos = cos[position_ids]  # [bs, 2, seq_len, dim]
        sin = sin[position_ids]  # [bs, 2, seq_len, dim]
        num_head = q.shape[1]
        cos = cos.repeat(1, num_head//2, 1, 1)
        sin = sin.repeat(1, num_head//2, 1, 1)
    else:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    if isinstance(k, list):
        k_embed = []
        for k_ in k:
            k_embed.append((k_ * cos) + (rotate_half(k_) * sin))
    else:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class LibraRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )



@dataclass
class LibraCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Libra's Base class for causal language model (or autoregressive) outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_hidden_states: Optional[torch.FloatTensor] = None
    past_vision_flag: Optional[torch.BoolTensor] = None

def cal_language_vision(x, vision_flag, language_module, vision_module, addition_mode=False):
    if addition_mode:
        language = x
        vision = x[vision_flag]
        if isinstance(language_module, list) or isinstance(language_module, tuple):
            for m in language_module:
                language = m(language)
        else:
            language = language_module(language)
        if isinstance(vision_module, list) or isinstance(vision_module, tuple):
            for m in vision_module:
                vision = m(vision)
        else:
            vision = vision_module(vision)
        
        language[vision_flag] = language[vision_flag] + vision
        return language
    else:
        language = x[~vision_flag]
        vision = x[vision_flag]

        if isinstance(language_module, list) or isinstance(language_module, tuple):
            for m in language_module:
                language = m(language)
        else:
            language = language_module(language)
        if isinstance(vision_module, list) or isinstance(vision_module, tuple):
            for m in vision_module:
                vision = m(vision)
        else:
            vision = vision_module(vision)

        additional_shape = language.shape[1:]
        placeholder = torch.empty(vision_flag.shape + additional_shape, dtype=language.dtype, device=language.device)
        placeholder[~vision_flag] = language
        placeholder[vision_flag] = vision
        return placeholder.contiguous()


class LibraLinear(nn.Module):
    __constants__ = ['in_features', 'down_ratio', 'out_features']
    in_features: int
    out_features: int
    weight_A: torch.Tensor
    weight_B: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False, down_ratio=4,
                 device=None, dtype=None, rank=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert in_features % down_ratio == 0
        assert bias == False, "Not checked yet."
        self.in_features = in_features
        self.out_features = out_features
        self.down_ratio = down_ratio
        self.rank = rank
        if rank is not None:
            self.weight_A = torch.nn.Parameter(torch.empty((rank, in_features), **factory_kwargs))
            self.weight_B = torch.nn.Parameter(torch.empty((out_features, rank), **factory_kwargs))
            self.down_ratio = out_features / rank
        else:
            self.weight_A = torch.nn.Parameter(torch.empty((out_features//down_ratio, in_features), **factory_kwargs))
            self.weight_B = torch.nn.Parameter(torch.empty((out_features, out_features//down_ratio), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))

        if self.rank is not None:
            torch.nn.init.constant_(self.weight_B, val=0.)
        else:
            torch.nn.init.kaiming_uniform_(self.weight_B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_B)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, mode=None) -> torch.Tensor:
        if mode == "pre":
            return F.linear(input, self.weight_A, None)
        elif mode == "post":
            return F.linear(hidden, self.weight_B, self.bias)
        hidden = F.linear(input, self.weight_A, None)
        output = F.linear(hidden, self.weight_B, self.bias)
        return output
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, down_ratio={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.down_ratio
        )

class LibraMLP(LlamaMLP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dropout: float,
        vision_dropout: float,
        down_ratio=4,
        addition_mode=False,
    ):
        super().__init__(hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         hidden_act=hidden_act,
                         dropout=dropout)
        self.addition_mode = addition_mode
        self.vision_gate_proj = LibraLinear(hidden_size, intermediate_size, bias=False, down_ratio=down_ratio)
        self.vision_down_proj = LibraLinear(intermediate_size, hidden_size, bias=False, down_ratio=down_ratio)
        self.vision_up_proj = LibraLinear(hidden_size, intermediate_size, bias=False, down_ratio=down_ratio)
        self.vision_dropout = nn.Dropout(vision_dropout)
    
    def forward(self, x, vision_flag):
        placeholder = torch.empty_like(x)
        language = x[~vision_flag]
        vision = x[vision_flag]

        language = self.dropout(self.down_proj(self.act_fn(self.gate_proj(language)) * self.up_proj(language)))
        vision = self.vision_dropout(self.vision_down_proj(self.act_fn(self.vision_gate_proj(vision)) * self.vision_up_proj(vision)))

        placeholder[~vision_flag] = language
        placeholder[vision_flag] = vision

        return placeholder


class LibraZero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class LibraAttention(LlamaAttention):
    def __init__(self, config: LibraConfig):
        super().__init__(config)

        self.vision_q_proj = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, down_ratio=config.vision_down_ratio)
        self.vision_k_proj = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, down_ratio=config.vision_down_ratio)
        self.vision_v_proj = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, down_ratio=config.vision_down_ratio)
        self.vision_o_proj = LibraLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, down_ratio=config.vision_down_ratio)

        self.vision_resid_drop = nn.Dropout(config.vision_resid_pdrop)
        
        self.use_bridge = config.use_bridge
        self.addition_mode = config.addition_mode
        if self.use_bridge:
            self.vision_v_bridge_on_language = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, rank=config.bridge_rank)
            self.vision_v_bridge_on_vision = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, rank=config.bridge_rank)

            self.vision_k_bridge_on_language = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, rank=config.bridge_rank)
            self.vision_k_bridge_on_vision = LibraLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, rank=config.bridge_rank)

        self.use_2d_rope = config.use_2d_rope

    def attn_with_bridge(
            self,
            attention_matrix,
            value_states,
            value_bridge_states,
            vision_flag,
            vision_flag_for_value,
    ):
        attention_matrix = attention_matrix.permute(1,0,2,3) # H, B, N, N
        value_states = value_states.permute(1,0,2,3) # H, B, N, C
        value_bridge_states = value_bridge_states.permute(1,0,2,3) # H, B, N, C

        # vision_attention_matrix = attention_matrix[:, vision_flag]
        # language_attention_matrix = attention_matrix[:, ~vision_flag]

        value_states_for_vision = value_states.clone()
        value_states_for_vision[:, ~vision_flag_for_value] = value_states_for_vision[:, ~vision_flag_for_value] + value_bridge_states[:, ~vision_flag_for_value]

        value_states_for_language = value_states.clone()
        value_states_for_language[:, vision_flag_for_value] = value_states_for_language[:, vision_flag_for_value] + value_bridge_states[:, vision_flag_for_value]

        # TODO: make it more efficient
        attn_output_for_vision =  torch.matmul(attention_matrix, value_states_for_vision)
        attn_output_for_language = torch.matmul(attention_matrix, value_states_for_language)

        attn_output = attn_output_for_vision
        attn_output[:, ~vision_flag] = attn_output_for_language[:, ~vision_flag]
        attn_output = attn_output.permute(1,0,2,3)

        return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        vision_flag: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.q_proj, vision_module=self.vision_q_proj, addition_mode=self.addition_mode).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_bridge:
            key_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.k_proj, vision_module=self.vision_k_proj, addition_mode=self.addition_mode)
        else:
            key_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.k_proj, vision_module=self.vision_k_proj, addition_mode=self.addition_mode).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.v_proj, vision_module=self.vision_v_proj, addition_mode=self.addition_mode).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_bridge:
            value_bridge_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.vision_v_bridge_on_language, vision_module=self.vision_v_bridge_on_vision).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_bridge_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.vision_k_bridge_on_language, vision_module=self.vision_k_bridge_on_vision)
            key_states_for_vision = key_states.clone()
            key_states_for_vision[~vision_flag] = key_states_for_vision[~vision_flag] + key_bridge_states[~vision_flag]
            key_states_for_language = key_states.clone()
            key_states_for_language[vision_flag] = key_states_for_language[vision_flag] + key_bridge_states[vision_flag]

            key_states_for_vision = key_states_for_vision.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states_for_language = key_states_for_language.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = [key_states_for_vision, key_states_for_language]

            kv_seq_len = key_states[0].shape[-2]
        else:
            kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.use_bridge:
                kv_seq_len += past_key_value[0][0].shape[-2]
            else:
                kv_seq_len += past_key_value[0].shape[-2]


        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, use_2d_rope=self.use_2d_rope)
        # [bsz, nh, t, hd]

        vision_flag_for_value = vision_flag
        if past_key_value is not None:
            # reuse k, v, self_attention
            if self.use_bridge:
                past_key_states_for_vision, past_key_states_for_language = past_key_value[0]
                key_states_for_vision = torch.cat([past_key_states_for_vision, key_states[0]], dim=2)
                key_states_for_language = torch.cat([past_key_states_for_language, key_states[1]], dim=2)
                key_states = [key_states_for_vision, key_states_for_language]
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            if self.use_bridge:
                value_bridge_states = torch.cat([past_key_value[2], value_bridge_states], dim=2)
                vision_flag_for_value = torch.cat([past_key_value[3], vision_flag], dim=1)

        if self.use_bridge:
            past_key_value = (key_states, value_states, value_bridge_states, vision_flag_for_value) if use_cache else None
        else:
            past_key_value = (key_states, value_states) if use_cache else None

        if self.use_bridge:
            attn_weights_for_vision = torch.matmul(query_states, key_states[0].transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights_for_language = torch.matmul(query_states, key_states[1].transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights_for_vision = attn_weights_for_vision.permute(1,0,2,3)
            attn_weights_for_language = attn_weights_for_language.permute(1,0,2,3)
            attn_weights = attn_weights_for_vision
            attn_weights[:, ~vision_flag] = attn_weights_for_language[:, ~vision_flag]
            attn_weights = attn_weights.permute(1,0,2,3)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_drop(attn_weights)

        if self.use_bridge:
            attn_output = self.attn_with_bridge(attention_matrix=attn_weights, value_states=value_states, value_bridge_states=value_bridge_states, vision_flag=vision_flag, vision_flag_for_value=vision_flag_for_value)
        else:
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


        attn_output = cal_language_vision(attn_output, vision_flag=vision_flag, language_module=[self.o_proj, self.resid_drop], vision_module=[self.vision_o_proj, self.vision_resid_drop], addition_mode=self.addition_mode)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class LibraDecoderLayer(nn.Module):
    def __init__(self, config: LibraConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.addition_mode = config.addition_mode
        self.self_attn = LibraAttention(config=config)
        self.mlp = LibraMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dropout=config.resid_pdrop,
            vision_dropout=config.vision_resid_pdrop,
            down_ratio=config.vision_down_ratio,
            addition_mode=config.addition_mode,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vision_post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        vision_flag: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.input_layernorm, vision_module=self.vision_input_layernorm)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            vision_flag=vision_flag,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.post_attention_layernorm, vision_module=self.vision_post_attention_layernorm)
        hidden_states = self.mlp(hidden_states, vision_flag=vision_flag)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

class LibraPreTrainedModel(BaseLibraPreTrainedModel):
    config_class = LibraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LibraDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, LibraLinear):
            module.weight_A.data.normal_(mean=0.0, std=std)
            if module.rank is not None or self.config.addition_mode:
                module.weight_B.data.zero_()
            else:
                module.weight_B.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LibraModel):
            module.gradient_checkpointing = value


class LibraModel(LibraPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LibraConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LibraDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.vision_vocab_size = config.vision_vocab_size
        self.vision_codebook_num = config.vision_codebook_num
        assert config.hidden_size % config.vision_codebook_num == 0
        self.vision_embed_tokens = nn.ModuleList([nn.Embedding(config.vision_vocab_size, config.hidden_size//config.vision_codebook_num) for _ in range(config.vision_codebook_num)])
        # self.vision_embed_tokens = nn.Embedding(config.vision_vocab_size, config.hidden_size)
        self.vision_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vision_drop = nn.Dropout(config.vision_embd_pdrop)

        self.concat_signals = config.concat_signals
        self.contiguous_signal_size = config.contiguous_signal_size
        self.norm_signals = config.norm_signals
        if self.concat_signals:
            self.vision_contiguous_signal_processor = nn.Linear(config.contiguous_signal_size+config.hidden_size, config.hidden_size, bias=False)
            if self.norm_signals:
                self.vision_signal_norm = LlamaRMSNorm(config.contiguous_signal_size+config.hidden_size, eps=config.rms_norm_eps)
            # self.vision_contiguous_signal_placeholder = nn.Parameter(torch.zeros(config.contiguous_signal_size))
        else:
            self.vision_contiguous_signal_processor = nn.Linear(config.contiguous_signal_size, config.hidden_size, bias=False)
        
        self.use_vision_position_embedding = config.use_vision_position_embedding
        if config.use_vision_position_embedding:
            self.vision_position_embedding = nn.Embedding(config.max_vision_token_length, config.hidden_size)
            # placeholder_position_embedding = torch.zeros([1, config.hidden_size])
            # self.register_buffer("placeholder_position_embedding", placeholder_position_embedding)
        

        self.max_vision_token_length = config.max_vision_token_length
        self.image_feature_resolution = config.image_feature_resolution
        assert self.image_feature_resolution**2 + 2 == self.max_vision_token_length

        self.use_2d_rope = config.use_2d_rope
        if config.use_2d_rope:
            pos_offset_h = torch.arange(1, self.image_feature_resolution+1)[..., None].expand(-1, self.image_feature_resolution)
            pos_offset_w = torch.arange(1, self.image_feature_resolution+1)[None, ...].expand(self.image_feature_resolution, -1)
            pos_offset = torch.stack([pos_offset_h, pos_offset_w], dim=-1)
            pos_offset = torch.cat(
                [
                    torch.zeros(1, 2, dtype=pos_offset.dtype),
                    pos_offset.view(-1, 2),
                    torch.zeros(2, 2, dtype=pos_offset.dtype),
                ], dim=0
            )
            self.register_buffer("pos_offset", pos_offset)


        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def get_inputs_embeds_from_multicodebook(self, input_ids, vision_flag, contiguous_signal=None, vision_indices=None):
        language = input_ids[0][~vision_flag]
        language = self.embed_tokens(language)

        vision_concat = []
        for i, sub_input_ids in enumerate(input_ids):
            vision = sub_input_ids[vision_flag]
            vision = self.vision_embed_tokens[i](vision)
            vision_concat.append(vision)
        vision_concat = torch.cat(vision_concat, dim=-1)

        if self.use_vision_position_embedding:
            vision_position_embedding = self.vision_position_embedding(vision_indices[vision_flag])
            vision_concat = vision_concat + vision_position_embedding

        if contiguous_signal is not None and self.concat_signals:
            if self.norm_signals:
                vision_embed = self.vision_signal_norm(torch.cat([vision_concat, contiguous_signal[vision_flag]], dim=-1))
            else:
                vision_embed = torch.cat([vision_concat, contiguous_signal[vision_flag]], dim=-1)
            vision_concat = self.vision_contiguous_signal_processor(vision_embed)
        else:
            if self.concat_signals:
                contiguous_signal_placeholder = vision_concat.new_zeros(list(vision_concat.shape[:-1])+[self.contiguous_signal_size])
                if self.norm_signals:
                    vision_embed = self.vision_signal_norm(torch.cat([vision_concat, contiguous_signal_placeholder], dim=-1))
                else:
                    vision_embed = torch.cat([vision_concat, contiguous_signal_placeholder], dim=-1)
                vision_concat = self.vision_contiguous_signal_processor(vision_embed)

        additional_shape = language.shape[1:]
        placeholder = torch.empty(vision_flag.shape + additional_shape, dtype=language.dtype, device=language.device)
        placeholder[~vision_flag] = language
        placeholder[vision_flag] = vision_concat


        return placeholder.contiguous()

    def get_2d_position_ids(self, vision_indices, attention_mask=None):
        mask = torch.logical_or(vision_indices==self.max_vision_token_length, vision_indices==0) # text, boi, eoi
        if attention_mask is not None:
            mask[attention_mask==0] = False
        mask = mask.long()
        mask[vision_indices==self.max_vision_token_length-1] = self.image_feature_resolution + 1
        # assert torch.sum(~(mask[:, 0].bool())) == 0 # ensure right padding and the first token must be a text token <s>
        position_ids = mask.cumsum(-1) - 1
        position_ids = position_ids[..., None].expand(-1, -1, 2)
        offset = self.pos_offset[vision_indices]
        position_ids = position_ids + offset

        if attention_mask is not None:
            position_ids[attention_mask==0] = 1 # for left padding

        return position_ids.permute(0,2,1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_flag: Optional[torch.BoolTensor] = None,
        contiguous_signal: Optional[torch.Tensor] = None,
        vision_indices: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            assert len(input_ids) == self.vision_codebook_num
            vision_flag_ = input_ids[0] >= self.vocab_size
            if vision_flag is not None:
                assert torch.equal(vision_flag_, vision_flag), "Inconsistent input_ids and vision_flag"
            vision_flag = vision_flag_
            batch_size, seq_length = input_ids.shape[1:]
        elif inputs_embeds is not None:
            assert vision_flag is not None
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            if isinstance(past_key_values[0][0], list):
                assert len(past_key_values[0][0]) == 2
                past_key_values_length = past_key_values[0][0][0].shape[2]
            else:
                past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            if self.use_2d_rope:
                assert past_key_values_length == 0, "TODO: fix it later to enable past_key_values_length>0"
                position_ids = self.get_2d_position_ids(vision_indices)
            else:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if self.use_2d_rope:
                position_ids = position_ids.long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            input_ids_ = input_ids.clone()
            input_ids_[:, vision_flag] -= self.vocab_size


            inputs_embeds = self.get_inputs_embeds_from_multicodebook(input_ids_, vision_flag=vision_flag, contiguous_signal=contiguous_signal, vision_indices=vision_indices)

            if contiguous_signal is not None and (not self.concat_signals):
                inputs_embeds = inputs_embeds + self.vision_contiguous_signal_processor(contiguous_signal)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        hidden_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.drop, vision_module=self.vision_drop)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    None,
                    vision_flag,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    vision_flag=vision_flag,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.norm, vision_module=self.vision_norm)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MultiLMHead(nn.Module):
    def __init__(self, head_num, input_dim, output_dim):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in range(head_num)])
    
    def forward(self, x):
        output = []
        for head in self.heads:
            output.append(head(x))
        return output

class LibraForCausalLM(LibraPreTrainedModel):
    def __init__(self, config: LibraConfig):
        super().__init__(config)
        self.model = LibraModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vision_prediction_mode = config.vision_prediction_mode
        self.unified_head = config.unified_head

        self.vision_codebook_num = config.vision_codebook_num
        # if config.vision_codebook_num > 1:
        self.vision_lm_head = MultiLMHead(head_num=config.vision_codebook_num,
                                            input_dim=config.hidden_size*2 if self.vision_prediction_mode=="2d" else config.hidden_size,
                                            output_dim=config.vision_vocab_size,)
        
        # else:
        #     self.vision_lm_head = nn.Linear(config.hidden_size*2 if self.vision_prediction_mode=="2d" else config.hidden_size, config.vision_vocab_size, bias=False)

        self.max_vision_token_length = config.max_vision_token_length
        self.image_feature_resolution = config.image_feature_resolution
        assert self.max_vision_token_length == self.image_feature_resolution**2 + 2
        self.vision_hidden_placeholder = nn.Parameter(torch.empty(config.hidden_size))
        self.vision_hidden_placeholder.data.normal_(mean=0.0, std=config.initializer_range)

        naive_placeholder = torch.zeros(config.hidden_size)
        self.register_buffer("naive_placeholder", naive_placeholder)

        
        vision_logits_placeholder = torch.full([1, config.vision_vocab_size], -float("inf"))
        language_logits_placeholder = torch.full([1, config.vocab_size], -float("inf"))
        self.register_buffer("vision_logits_placeholder", vision_logits_placeholder)
        self.register_buffer("language_logits_placeholder", language_logits_placeholder)

        eoi_to_newline_logits_placeholder = torch.full([1, 1, 1, config.vocab_size+config.vision_vocab_size], -float("inf"))
        self.newline_token_id = config.newline_token_id
        eoi_to_newline_logits_placeholder[:, :, :, self.newline_token_id] = float("inf")
        self.register_buffer("eoi_to_newline_logits_placeholder", eoi_to_newline_logits_placeholder)

        self.use_2d_rope = config.use_2d_rope
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def cal_vision_logits_inference(self, hidden_states, vision_flag, past_hidden_states, past_vision_flag, vision_indices):
        '''
        Expand the inputs to complete images, and slice the real part after process
        '''
        # only support generating token by token
        if past_hidden_states is not None:
            assert hidden_states.shape[1] == 1 and vision_flag.shape[1] == 1
            hidden_states = torch.cat([past_hidden_states, hidden_states], dim=1)
            vision_flag = torch.cat([past_vision_flag, vision_flag], dim=1)
        
        # extend vision flag and hidden states to complete image
        last_vision_indices = vision_indices[:, -1:]
        expanded_indices = last_vision_indices.expand(-1, self.max_vision_token_length)
        expanded_indices = expanded_indices + torch.arange(0, self.max_vision_token_length, device=vision_indices.device)
        expanded_vision_flag = (expanded_indices < self.max_vision_token_length)[:, 1:]

        expanded_vision_flag = torch.cat([vision_flag, expanded_vision_flag], dim=1)

        expanded_hidden_states = torch.cat(
            [
                hidden_states,
                self.naive_placeholder[None, None, ...].expand(hidden_states.shape[0], self.max_vision_token_length-1, -1)
            ], dim=1
        )

        real_token_flag = expanded_vision_flag.clone()
        real_token_flag[:, -(self.max_vision_token_length-1):] = False

        if past_hidden_states is not None:
            real_token_flag[:, :past_hidden_states.shape[1]] = False
        
        real_token_flag = real_token_flag[expanded_vision_flag]
        vision_logits = self.cal_vision_logits_train(expanded_hidden_states, expanded_vision_flag, real_token_flag)

        return vision_logits

    def cal_vision_logits_train(self, hidden_states, vision_flag, select_map=None, vision_indices=None):
        # assume all images are complete
        # assert self.training
        real_vision_flag = None
        if vision_indices is not None:
            incomplete_image_flag = vision_indices[:, -1] < self.max_vision_token_length-1
            if incomplete_image_flag.sum() > 0:
                pad_num = (self.max_vision_token_length-1) - vision_indices[:, -1][incomplete_image_flag].min()
                
                vision_flag = torch.cat(
                    [
                        vision_flag,
                        torch.full([vision_flag.shape[0], pad_num], fill_value=False, device=vision_flag.device, dtype=vision_flag.dtype)
                    ], dim=1
                )
                real_vision_flag = vision_flag.clone()
                vision_flag[incomplete_image_flag, -pad_num:] = True
                real_vision_flag = real_vision_flag[vision_flag]

                hidden_states = torch.cat(
                    [
                        hidden_states,
                        self.naive_placeholder[None, None, ...].expand(hidden_states.shape[0], pad_num, -1)
                    ], dim=1
                )

        vision = hidden_states[vision_flag]
        N, C = vision.shape
        assert N % (self.max_vision_token_length) == 0

        image_num = N//(self.max_vision_token_length)
        vision = vision.view(image_num, self.max_vision_token_length, C)

        hidden_to_predict_eoi = torch.cat(
            [
                vision[:, -2:-1, :], # left, last image token
                self.vision_hidden_placeholder[None, None, ...].expand(image_num, -1, -1), # up
            ], dim=-1
        )

        eoi_placeholder = torch.cat( # only for placeholder, since eoi does not contribute to loss
            [
                vision[:, -1:, :], # eoi
                self.vision_hidden_placeholder[None, None, ...].expand(image_num, -1, -1),
            ], dim=-1
        )
        
        vision_without_special = vision[:, 1:-1, :]
        vision_without_special = vision_without_special.view(image_num, self.image_feature_resolution, self.image_feature_resolution, C) # reshape to 2D
        
        vision_hidden_placeholder = self.vision_hidden_placeholder[None, None, None, ...]
        augmented_map = vision_hidden_placeholder.repeat(image_num, self.image_feature_resolution+1, self.image_feature_resolution+1, 1)
        augmented_map[:, 1, 0, :] = vision[:, 0, :] # boi
        augmented_map[:, 1:, 1:, :] = vision_without_special

        up = augmented_map[:, :-1, 1:, :]
        left = augmented_map[:, 1:, :-1, :]

        vision_without_special = torch.cat([up, left], dim=-1) # already add boi
        vision_without_special = vision_without_special.reshape(image_num, self.max_vision_token_length-2, 2*C)

        vision = torch.cat([vision_without_special, hidden_to_predict_eoi, eoi_placeholder], dim=1)

        vision = vision.view(-1, 2*C)

        if real_vision_flag is not None:
            vision = vision[real_vision_flag]

        if select_map is not None:
            vision = vision[select_map]

        vision_logits = self.vision_lm_head(vision)
        return vision_logits



    def cal_vl_logits(self, hidden_states, vision_flag, past_hidden_states=None, past_vision_flag=None, use_cache=False, vision_indices=None):
        if not self.unified_head:
            language = hidden_states[~vision_flag]
            language_logits = self.lm_head(language)
            assert len(language_logits.shape) == 2
            vision_logits_placeholder = self.vision_logits_placeholder.expand(language_logits.shape[0], -1)
            language_logits = torch.cat([language_logits, vision_logits_placeholder], dim=-1)
        
            if self.vision_prediction_mode == "2d":
                if use_cache:
                    vision_logits = self.cal_vision_logits_inference(hidden_states=hidden_states, vision_flag=vision_flag, past_hidden_states=past_hidden_states, past_vision_flag=past_vision_flag, vision_indices=vision_indices)
                else:
                    vision_logits = self.cal_vision_logits_train(hidden_states=hidden_states, vision_flag=vision_flag, vision_indices=vision_indices)
            else:
                vision = hidden_states[vision_flag]
                vision_logits = self.vision_lm_head(vision)

            assert len(vision_logits[0].shape) == 2
            language_logits_placeholder = self.language_logits_placeholder.expand(vision_logits[0].shape[0], -1)
            

            new_vision_logits = []
            for vision_sub_logits in vision_logits:
                new_vision_logits.append(torch.cat([language_logits_placeholder, vision_sub_logits], dim=-1))
            vision_logits = new_vision_logits
        
            logits_stacked = []
            for vision_sub_logits in vision_logits:
                additional_shape = language_logits.shape[1:]
                logits = torch.empty(vision_flag.shape + additional_shape, dtype=language_logits.dtype, device=language_logits.device)
                logits[~vision_flag] = language_logits
                logits[vision_flag] = vision_sub_logits
                logits_stacked.append(logits)
            logits_stacked = torch.stack(logits_stacked)
            return logits_stacked
    
        else:
            assert self.vision_prediction_mode != "2d" # TODO: implement later
            language_logits = self.lm_head(hidden_states)
            vision_logits = self.vision_lm_head(hidden_states)
            vision_logits = torch.stack(vision_logits)
            if use_cache:
                language_logits[vision_flag] = self.language_logits_placeholder
                vision_logits[:, ~vision_flag] = self.vision_logits_placeholder[None, ...]
            language_logits = language_logits[None, ...].expand(vision_logits.shape[0], -1, -1, -1)
            logits_concated = torch.cat([language_logits, vision_logits], dim=-1)
            return logits_concated
            

    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=LibraCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_indices: Optional[torch.LongTensor] = None,
        contiguous_signal: Optional[torch.Tensor] = None,
        past_hidden_states: Optional[torch.Tensor] = None,
        past_vision_flag: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, LibraCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_flag = (vision_indices < self.max_vision_token_length)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_flag=vision_flag,
            contiguous_signal=contiguous_signal,
            vision_indices=vision_indices,
        )

        hidden_states = outputs[0]
        
        # Now we have N logits
        logits_list = self.cal_vl_logits(hidden_states=hidden_states, vision_flag=vision_flag, use_cache=use_cache, vision_indices=vision_indices, past_hidden_states=past_hidden_states, past_vision_flag=past_vision_flag)

        if past_key_values is not None:
            assert not self.training
            eoi_mask = (vision_indices[:, -1] == self.max_vision_token_length-1) # self.max_vision_token_index-1 represents  <\img>
            logits_list[:, eoi_mask, -1:, :] = self.eoi_to_newline_logits_placeholder.expand(logits_list[:, eoi_mask, -1:, :].shape[0], logits_list[:, eoi_mask, -1:, :].shape[1], 1, -1) # if reach <\img>, do not use it to predict anything. Just append a "\n" token.


        if self.vision_prediction_mode == "2d" and use_cache:
            if past_hidden_states is not None:
                past_hidden_states = torch.cat([past_hidden_states, hidden_states], dim=1)
            else:
                past_hidden_states = hidden_states

            if past_vision_flag is not None:
                past_vision_flag = torch.cat([past_vision_flag, vision_flag], dim=1)
            else:
                past_vision_flag = vision_flag


        loss = None
        if labels is not None:
            assert len(labels) == self.vision_codebook_num
            loss = 0.
            for sub_labels, logits in zip(labels, logits_list):
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = sub_labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size + self.config.vision_vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss += loss_fct(shift_logits, shift_labels)
            loss = loss / self.vision_codebook_num

        if not return_dict:
            output = (logits_list,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LibraCausalLMOutputWithPast(
            loss=loss,
            logits=logits_list,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_hidden_states=past_hidden_states,
            past_vision_flag=past_vision_flag,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, vision_indices=None, contiguous_signal=None, past_hidden_states=None, past_vision_flag=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, :, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            if self.use_2d_rope:
                position_ids = self.model.get_2d_position_ids(vision_indices=vision_indices, attention_mask=attention_mask)
                if past_key_values:
                    position_ids = position_ids[:, :, -1].unsqueeze(-1)
            else:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if past_key_values:
            vision_indices = vision_indices[:, -1].unsqueeze(-1)
            contiguous_signal=None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "vision_indices": vision_indices,
                "contiguous_signal": contiguous_signal,
                "past_hidden_states": past_hidden_states,
                "past_vision_flag": past_vision_flag,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_hidden_states
        model_kwargs["past_hidden_states"] = outputs.past_hidden_states
        # update past_vision_flag
        model_kwargs["past_vision_flag"] = outputs.past_vision_flag

        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )
        
        if "vision_indices" in model_kwargs:
            vision_indices = model_kwargs["vision_indices"]
            next_vision_indices = vision_indices[:, -1].clone() + 1
            is_next_language = (next_vision_indices >= self.max_vision_token_length)
            next_vision_indices[is_next_language] = self.max_vision_token_length
            next_vision_indices = next_vision_indices[..., None]
            model_kwargs["vision_indices"] = torch.cat(
                    [vision_indices, next_vision_indices], dim=-1
                )
        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
    

@registry.register_model("libra_train_wrapper")
class LibraTrainWrapper(LibraPreTrainedModel):
    def __init__(self, config):
        temp_config = json.load(open(Path(config.pretrained, "config.json"), "r"))
        config_ = LibraConfig(**temp_config)
        del temp_config
        super().__init__(config_)

        custom_kwargs = config.get("custom_kwargs", {})
        self.module = LibraForCausalLM.from_pretrained(config.pretrained, **custom_kwargs)
        tokenizer_kwargs = config.get("tokenizer_kwargs", {})
        self.tokenizer = LibraTokenizer(config.pretrained, **tokenizer_kwargs)
        self.change_pad_token_to_eos(pad_token_id=self.tokenizer.text_tokenizer.pad_token_id, eos_token_id=self.tokenizer.text_tokenizer.eos_token_id)

        model_kwargs = config.get("model_kwargs", {})
        # self.llm_lr_scale = model_kwargs.get("llm_lr_scale", 0.1)
        # self.save_and_load_model_without_wrapper = model_kwargs.get("save_and_load_model_without_wrapper", False)
        # self.disable_ignore_key_on_save = model_kwargs.get("disable_ignore_key_on_save", False)

        pretrained_weight = config.get("pretrained_weight", None)
        if pretrained_weight is not None:
            state_dict = torch.load(pretrained_weight, map_location="cpu")
            new_state_dict = {}

            # to be compatible with the old verision. remove it latter
            has_wrapper = False
            has_module_wrapper = False
            for key, value in state_dict.items():
                if key.startswith('model.model.'):
                    has_wrapper = True
                if key.startswith('module.model.'):
                    has_module_wrapper = True
            assert not (has_wrapper and has_module_wrapper), "'has_wrapper' and 'has_module_wrapper' cannot both be True."
            if has_wrapper:
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]
                        new_state_dict[new_key] = value
            if has_module_wrapper:
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]
                        new_state_dict[new_key] = value

                state_dict = new_state_dict

            missing, unexpected = self.module.load_state_dict(state_dict, strict=False)
            print("missing keys: ", missing)
            print("unexpected keys: ", unexpected)
        
        frozen_language = model_kwargs.get("frozen_language", False)
        if frozen_language:
            for key, p in self.module.named_parameters():
                if not "vision" in key:
                    p.requires_grad = False

        freeze_vision_value = model_kwargs.get("freeze_vision_value", False)
        if freeze_vision_value:
            for key, p in self.module.named_parameters():
                if "vision_v_proj" in key:
                    p.requires_grad = False
        
        freeze_text_embedding = model_kwargs.get("freeze_text_embedding", False)
        if freeze_text_embedding:
            for key, p in self.module.named_parameters():
                if ".embed_tokens" in key:
                    p.requires_grad = False
        
        freeze_vision_embedding = model_kwargs.get("freeze_vision_embedding", False)
        if freeze_vision_embedding:
            for key, p in self.module.named_parameters():
                if ".vision_embed_tokens" in key:
                    p.requires_grad = False

        if model_kwargs.get("debug", False):
            for key, p in self.module.named_parameters():
                if "vision_lm_head" not in key:
                    p.requires_grad = False


    # @property
    # def _keys_to_ignore_on_save(self): # TODO: check this doesn't break anything
    #     # if self.disable_ignore_key_on_save:
    #     #     return None
    #     ignore_key_list = []
    #     for key, p in self.named_parameters():
    #         if not p.requires_grad:
    #             ignore_key_list.append(key)
    #     return ignore_key_list

    @classmethod
    def get_model_from_config(cls, config):
        custom_kwargs = config.get("custom_kwargs", {})
        model = LibraForCausalLM.from_pretrained(config.pretrained, torch_dtype="auto", **custom_kwargs)
        tokenizer_kwargs = config.get("tokenizer_kwargs", {})
        tokenizer = LibraTokenizer(config.pretrained, **tokenizer_kwargs)
        return model, tokenizer

    def change_pad_token_to_eos(self, pad_token_id=0, eos_token_id=2):
        # change padding token to eos token to avoid nan because padding token has a different scale with other tokens.
        pad_idx = pad_token_id
        eos_idx = eos_token_id
        embed_tokens = self.module.get_input_embeddings()
        embed_tokens.weight.data[pad_idx] = embed_tokens.weight.data[eos_idx].clone()
    
    def get_labels(self, inputs ,label_mask_position_map):
        labels = inputs["input_ids"].clone()
        labels[:, inputs["attention_mask"]==0] = -100
        labels[labels==self.tokenizer.image_tokenizer.boi_token_id] = -100
        labels[labels==self.tokenizer.text_tokenizer.bos_token_id] = -100

        labels = labels.permute(1, 2, 0)


        for label, label_mask_position in zip(labels, label_mask_position_map):
            for pos_map in label_mask_position:
                start, end = pos_map
                label[start:end] = -100
        labels = labels.permute(2, 0, 1)
        return labels


    def forward(self, samples, return_loss=None, **kwargs):
        # return_loss is for evaluation, only a placeholder here
        # assert self.training
        inputs = self.tokenizer(samples,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.text_tokenizer.model_max_length,
                                truncation=True,
                                )

        labels = self.get_labels(inputs, samples["label_mask_position_map"])
        return self.module(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            vision_indices=inputs["vision_indices"],
            contiguous_signal=inputs["coninous_signal"],
            labels=labels,
            use_cache=False,
            **kwargs,
            )

    @classmethod
    def from_config(cls, config):
        return cls(config)
    
    # def get_optimizer_parameters(self):
    #     decay_parameters_ = get_parameter_names(self, [nn.LayerNorm ,LlamaRMSNorm])
    #     decay_parameters = []
    #     for n in decay_parameters_:
    #         # if "vision_gate" in n:
    #         #     continue
    #         if "bias" in n:
    #             continue
    #         decay_parameters.append(n)

    #     param_group_dict = {
    #         "scaled_lr_with_wd": {
    #             "params": [],
    #             "lr_scale": self.llm_lr_scale,
    #             "use_weight_decay": True,
    #         },
    #         "scaled_lr_without_wd": {
    #             "params": [],
    #             "lr_scale": self.llm_lr_scale,
    #             "use_weight_decay": False,
    #         },
    #         "unscaled_lr_with_wd": {
    #             "params": [],
    #             "lr_scale": 1.0,
    #             "use_weight_decay": True,
    #         },
    #         "unscaled_lr_without_wd": {
    #             "params": [],
    #             "lr_scale": 1.0,
    #             "use_weight_decay": False,
    #         },
    #         }
        
    #     for n, p in self.named_parameters():
    #         if not p.requires_grad:
    #             continue
            
    #         if "vision" not in n:
    #             lr_scale = self.llm_lr_scale
    #         else:
    #             lr_scale = 1.0
            
    #         if n in decay_parameters:
    #             use_weight_decay = True
    #         else:
    #             use_weight_decay = False
            
    #         if lr_scale==self.llm_lr_scale and use_weight_decay: param_group_dict["scaled_lr_with_wd"]["params"].append(p)
    #         elif lr_scale==self.llm_lr_scale and not use_weight_decay: param_group_dict["scaled_lr_without_wd"]["params"].append(p)
    #         elif lr_scale==1.0 and use_weight_decay: param_group_dict["unscaled_lr_with_wd"]["params"].append(p)
    #         elif lr_scale==1.0 and not use_weight_decay: param_group_dict["unscaled_lr_without_wd"]["params"].append(p)
    #         else: raise NotImplementedError
            
    #     optimizer_grouped_parameters = [v for _, v in param_group_dict.items() if len(v["params"]) > 0]
    #     return optimizer_grouped_parameters
    

    






        




