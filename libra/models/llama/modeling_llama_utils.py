from libra.models.llama.modeling_llama import *
from transformers.utils import ModelOutput
from dataclasses import dataclass
from transformers import LogitsProcessor

_CONFIG_FOR_DOC = "LlamaConfig"

class NoNewlineLogitsProcessor(LogitsProcessor):
    def __init__(self, newline_token_id: int, eos_token_id: int):
        if not isinstance(newline_token_id, int) or not isinstance(eos_token_id, int):
            raise ValueError(f"token ids have to be an integer")
        self.newline_token_id = newline_token_id
        self.eos_token_id = eos_token_id
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = input_ids[... ,-1] == self.newline_token_id
        no_eos_token_mask = torch.ones(scores.shape, dtype=torch.bool, device=input_ids.device)
        no_eos_token_mask[..., self.eos_token_id] = False
        no_eos_token_mask[~mask] = False
        scores[no_eos_token_mask] = -float("inf")
        return scores

class ValidImageLogitsProcessor(LogitsProcessor):
    def __init__(self, valid_image_token_length: int, boi_token_id: int, eoi_token_id: int, image_logits_offset: int, logits_size: int):
        assert math.sqrt(valid_image_token_length)%1 == 0, "only support square images, and valid_image_token_length does not consider <img> and <\img> tokens"
        self.augmented_valid_image_token_length = valid_image_token_length + 2
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_logits_offset = image_logits_offset
        
        self.gen_image_mask_preserved = torch.ones([1, logits_size], dtype=torch.bool)
        self.gen_image_mask_preserved[:, :self.image_logits_offset] = False
        self.gen_image_mask_preserved[:, self.eoi_token_id] = False
        self.gen_image_mask_preserved[:, self.boi_token_id] = False

        self.stop_image_mask_preserved = torch.zeros([1, logits_size], dtype=torch.bool)
        self.stop_image_mask_preserved[:, self.eoi_token_id] = True

    def process_score(self, input_ids, scores):
        assert len(input_ids.shape)==2, "Currently, input_ids must shaped as (batch_size, num_seq)"
        
        B = input_ids.shape[0]

        reversed_ids = torch.flip(input_ids, dims=[-1])
        text_mask = reversed_ids < self.image_logits_offset
        cumulative_sum = torch.cumsum(text_mask, dim=-1)
        latest_image_count = torch.sum(cumulative_sum==0, dim=-1)

        gen_image_mask = (latest_image_count > 0) * (latest_image_count < self.augmented_valid_image_token_length-1) # generate image tokens at next step
        stop_image_mask = latest_image_count == self.augmented_valid_image_token_length - 1 # generate <\img> tokens at next step
        valid_image_mask = latest_image_count == self.augmented_valid_image_token_length # if a full image is generated, validate that the last image token is <\img>
        invalid_image_mask = latest_image_count > self.augmented_valid_image_token_length # if generated image tokens exceed valid length, raise error

        if sum(invalid_image_mask) > 0:
            raise ValueError("You have generated an invalid image.")
        # scores[gen_image_mask]

        gen_image_mask = gen_image_mask[...,None] # B, 1
        gen_image_mask_preserved = self.gen_image_mask_preserved.to(input_ids.device).expand(B, -1) # B, vocab_size
        scores[gen_image_mask * (~gen_image_mask_preserved)] = -float("inf")

        stop_image_mask = stop_image_mask[...,None] # B, 1
        stop_image_mask_preserved = self.stop_image_mask_preserved.to(input_ids.device).expand(B, -1)
        scores[stop_image_mask * (~stop_image_mask_preserved)] = -float("inf")

        invalid_image_sign = input_ids[valid_image_mask][:, -1] != self.eoi_token_id
        if invalid_image_sign.sum() > 0:
            raise ValueError("Find images that do not end with <\img> tokens")
        
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_scores = []
        for sub_input_ids, sub_scores in zip(input_ids, scores):
            new_scores.append(self.process_score(sub_input_ids, sub_scores))
        return torch.stack(new_scores)









@dataclass
class MetaCausalLMOutputWithPast(ModelOutput):
    logits: Union[torch.FloatTensor, Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None
    text_logits: Optional[torch.FloatTensor] = None
    image_logits: Optional[torch.FloatTensor] = None
    token_type: Optional[torch.LongTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class MetaWrapperOutput(ModelOutput):
    loss: torch.FloatTensor = None



class MetaLlamaForCausalLM(LlamaForCausalLM):
    '''
    always output last hidden states
    '''
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # with torch.no_grad():
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        output_hidden_states = outputs.hidden_states

        return MetaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,
            attentions=outputs.attentions,
            last_hidden_states=hidden_states,
        )