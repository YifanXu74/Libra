from typing import Optional, Tuple
import warnings

import torch

# import transformers
# from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from libra.models.libra.modeling_libra import apply_rotary_pos_emb, LibraModel, LibraAttention, cal_language_vision

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
import torch.nn.functional as F

# try:
#     from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
# except ImportError:
#     from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
# from flash_attn.bert_padding import unpad_input, pad_input
# import logging

def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if self.use_bridge:
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim*2), indices_k
        )
    else:
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        padding_mask = padding_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    vision_flag: Optional[torch.BoolTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.q_proj, vision_module=self.vision_q_proj).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.k_proj, vision_module=self.vision_k_proj).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.v_proj, vision_module=self.vision_v_proj).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    if self.use_bridge:
        bridge_value_states = cal_language_vision(hidden_states, vision_flag=vision_flag, language_module=self.vision_v_bridge_on_language, vision_module=self.vision_v_bridge_on_vision).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = torch.cat([value_states, bridge_value_states], dim=-1)


    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # TODO: llama does not have dropout in the config??
    # It is recommended to use dropout with FA according to the docs
    # when training.
    dropout_rate = self.attn_drop_rate  if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        warnings.warn(
            "The input hidden states seems to be silently casted in float32, this might be related to"
            " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            " float16."
        )

        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

    
    
    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, q_len
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=True,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout_rate, softmax_scale=None, causal=True
        )
    

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    
    attn_output = cal_language_vision(attn_output, vision_flag=vision_flag, language_module=[self.o_proj, self.resid_drop], vision_module=[self.vision_o_proj, self.vision_resid_drop])

    return attn_output, None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask if 0 in attention_mask else None


def replace_llama_attn_with_flash_attn():
    raise NotImplementedError("Libra does not support flashattention for now.")
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    LibraModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    LibraAttention.forward = forward
    setattr(LibraAttention, "_upad_input", _upad_input)