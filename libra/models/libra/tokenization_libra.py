import torch

from libra.models.llama.tokenization_llama_fast import LlamaTokenizerFast as LlamaTokenizer



from libra.models.libra.image_tokenizer import ImageTokenizer
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from transformers import BatchEncoding
from omegaconf import OmegaConf
from pathlib import Path
from transformers.tokenization_utils import TextInput, logger, AddedToken, re
import logging

MAX_TOKEN_LENGTH = 2048

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def get_encoded_image_size(image, downsample=16):
    # need debug
    assert len(image.shape) == 3
    _, H, W = image.shape
    return H//downsample * W//downsample


class LibraTextTokenizer(LlamaTokenizer):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )

    # def tokenize(self, text: TextInput, **kwargs) -> List[str]:
    #     '''
    #     Cancel strip '\n' when encoding added tokens <img_ph>, <img_gen>
    #     '''


    #     # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
    #     all_special_tokens_extended = {
    #         str(t): t for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
    #     }

    #     text, kwargs = self.prepare_for_tokenization(text, **kwargs)

    #     if kwargs:
    #         logger.warning(f"Keyword arguments {kwargs} not recognized.")

    #     # TODO: should this be in the base class?
    #     if hasattr(self, "do_lower_case") and self.do_lower_case:
    #         # convert non-special tokens to lowercase
    #         escaped_special_toks = [
    #             re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
    #         ]
    #         pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
    #         text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

    #     no_split_token = set(self.unique_no_split_tokens)
    #     tokens = self.tokens_trie.split(text)
    #     added_token = set(self.get_added_vocab().keys())
    #     # ["This is something", "<special_token_1>", "  else"]
    #     for i, token in enumerate(tokens):
    #         if token in no_split_token:
    #             tok_extended = all_special_tokens_extended.get(token, None)
    #             left = tokens[i - 1] if i > 0 else None
    #             right = tokens[i + 1] if i < len(tokens) - 1 else None

    #             if token in added_token:
    #                 # We strip left and right by " "
    #                 if right:
    #                     tokens[i + 1] = right.lstrip(" ")
    #                 if left:
    #                     tokens[i - 1] = left.rstrip(" ")
    #             else:
    #                 if isinstance(tok_extended, AddedToken):
    #                     if tok_extended.rstrip and right:
    #                         # A bit counter-intuitive but we strip the left of the string
    #                         # since tok_extended.rstrip means the special token is eating all white spaces on its right
    #                         tokens[i + 1] = right.lstrip()
    #                     # Strip white spaces on the left
    #                     if tok_extended.lstrip and left:
    #                         tokens[i - 1] = left.rstrip()  # Opposite here
    #                 else:
    #                     # We strip left and right by default
    #                     if right:
    #                         tokens[i + 1] = right.lstrip()
    #                     if left:
    #                         tokens[i - 1] = left.rstrip()
    #     # ["This is something", "<special_token_1>", "else"]
    #     tokenized_text = []
    #     for token in tokens:
    #         # Need to skip eventual empty (fully stripped) tokens
    #         if not token:
    #             continue
    #         if token in no_split_token:
    #             tokenized_text.append(token)
    #         else:
    #             tokenized_text.extend(self._tokenize(token))
    #     # ["This", " is", " something", "<special_token_1>", "else"]
    #     return tokenized_text



class LibraTokenizer(torch.nn.Module):
    def __init__(self, pretrained_model_path, vision_config_overwrite = {}, **kwargs):
        super().__init__()
        # debug: add add_eos_token
        self.text_tokenizer = self.init_text_tokenizer(pretrained_model_path, **kwargs)
        self.image_tokenizer_offset = self.text_tokenizer.vocab_size
        self.image_tokenizer = self.init_image_tokenizer(pretrained_model_path, self.image_tokenizer_offset, vision_config_overwrite)
        max_vision_token_length = self.image_tokenizer.max_vision_token_length
        img_indices_ph = torch.arange(0, max_vision_token_length, dtype=torch.long)[None, :]
        self.register_buffer("img_indices_ph", img_indices_ph)

        self.raw_output = kwargs.get("raw_output", False)

        # self.codebook_size = self.image_tokenizer.codebook_size
        self.num_codebook = self.image_tokenizer.num_codebook

    @property
    def device(self):
        # return list(self.parameters())[0].device
        return self.image_tokenizer.device
    
    @property
    def dtype(self):
        # return list(self.parameters())[0].dtype
        return self.image_tokenizer.dtype

    @classmethod
    def init_text_tokenizer(cls, pretrained_model_path, **kwargs):
        tokenizer = LibraTextTokenizer.from_pretrained(pretrained_model_path, **kwargs)
        tokenizer.add_tokens("<img_ph>")
        tokenizer.add_tokens("<img_gen>")
        tokenizer.img_ph_token_id = tokenizer.convert_tokens_to_ids("<img_ph>")
        tokenizer.img_gen_token_id = tokenizer.convert_tokens_to_ids("<img_gen>")
        tokenizer.pad_token = tokenizer.unk_token
        
        # For training, eos_token should be added. For inference, eos_token should not be added.
        # tokenizer.add_eos_token = add_eos_token
        return tokenizer
    
    @classmethod
    def init_image_tokenizer(cls, pretrained_model_path, offset, vision_config_overwrite={}):
        vision_tokenizer_config_name = "vision_tokenizer_config.yaml"
        image_tokenizer_config_path = Path(pretrained_model_path, vision_tokenizer_config_name)
        config = OmegaConf.load(image_tokenizer_config_path)
        if config.get("ckpt_path", None) is not None:
            config.ckpt_path = str(Path(pretrained_model_path, config.ckpt_path))
        if config.params.get("ckpt_path", None) is not None:
            config.params["ckpt_path"] = str(Path(pretrained_model_path, config.params["ckpt_path"]))
        if config.params["ddconfig"].get("encoder_name", None) is not None:
            config.params["ddconfig"]["encoder_name"] = str(Path(pretrained_model_path, config.params["ddconfig"]["encoder_name"]))
        config.update(**vision_config_overwrite)
        return ImageTokenizer.from_config(config, token_offset=offset)

    @torch.no_grad()
    def forward(self, samples, **kwargs):
        '''
        samples: [{"language": "a cute dog <img1_ph> and a cat <img2_ph>. I like them.", "vision": [img1, img2]}, ...]
        language_predefined: {"input_ids", "attention_mask"} if not None, directly use language_predefined without tokenizing language.
        outputs: token id: torch.tensor[a cute dog [\BOI] [img1] [\EOI] and a cat [\BOI] [img2] [\EOI]. I like them. , ...]
        '''
        # get inputs
        if not (isinstance(samples, list) or isinstance(samples, tuple)):
            samples = [samples]
        texts = []
        images = []
        contiguous_ignore_signs = []
        for sample in samples:
            lang = sample.get("language", None)
            vision = sample.get("vision", None)
            contiguous_ignore_sign = sample.get("contiguous_ignore_sign", None)
            if lang is not None:
                if isinstance(lang, list) or isinstance(lang, tuple):
                    texts += lang
                else:
                    texts.append(lang)
            if vision is not None:
                if isinstance(vision, list) or isinstance(vision, tuple):
                    images += vision
                else:
                    images.append(vision) 
            if contiguous_ignore_sign is not None:
                if isinstance(contiguous_ignore_sign, list) or isinstance(contiguous_ignore_sign, tuple):
                    contiguous_ignore_signs += contiguous_ignore_sign
                else:
                    contiguous_ignore_signs.append(contiguous_ignore_sign)
        texts = texts if len(texts) > 0 else None
        if  len(images) > 0:
            images = [img.to(self.device) for img in images]
            if len(images[0].shape) == 3:
                images = torch.stack(images)
            elif len(images[0].shape) == 4:
                images = torch.cat(images)
            else:
                raise ValueError("Invalid vision inputs.")
        else:
            images = None
        
        if len(contiguous_ignore_signs) > 0:
            if isinstance(contiguous_ignore_signs[0], torch.Tensor):
                contiguous_ignore_signs = torch.cat(contiguous_ignore_signs)
            else:
                contiguous_ignore_signs = torch.tensor(contiguous_ignore_signs, device=self.device)
        else:
            contiguous_ignore_signs = None

        
        has_image_flag = sample.get("has_image", None)
        if has_image_flag is not None:
            has_image_flag = torch.tensor(has_image_flag, device=self.device, dtype=torch.bool)

  
        if texts is None or len(texts) == 0:
            has_texts = False
        else:
            assert isinstance(texts, list)
            has_texts = True
        
        if images is None or len(images) == 0:
            has_images = False
        else:
            assert isinstance(images, torch.Tensor)
            has_images = True

        
        if (not has_texts) and (not has_images):
            raise ValueError("Empty inputs")
        elif not has_texts:
            raise NotImplementedError

        if kwargs.pop('return_tensors', "pt") != "pt":
            raise ValueError("return_tensors = \"pt\" is fixed, and should not be specified to other values.")

        truncation = kwargs.pop('truncation', False)
        max_length = kwargs.pop('max_length', self.text_tokenizer.model_max_length)
        

        text_inputs = self.text_tokenizer(texts, return_tensors="pt", return_length=True, **kwargs).to(self.device)
        if (text_inputs["length"] > MAX_TOKEN_LENGTH).sum():
            logging.warning("The input token length ecceeds the max number that the model can hold. This may cause performance degradation or OOM.")

            
        img_ph_token_mask = (text_inputs["input_ids"] == self.text_tokenizer.img_ph_token_id)

        input_ids = text_inputs["input_ids"]
        img_gen_token_mask = (input_ids==self.text_tokenizer.img_gen_token_id)
        input_ids[img_gen_token_mask] = self.image_tokenizer.boi_token_id

        input_ids = input_ids[None, ...].repeat(self.image_tokenizer.num_codebook, 1, 1)

        if has_images:
            images = images.to(self.dtype)
            image_inputs = self.image_tokenizer(images)

            if has_image_flag is not None:
                image_inputs["input_ids"] = image_inputs["input_ids"][:, has_image_flag]
                image_inputs["encoder_feat"] = image_inputs["encoder_feat"][has_image_flag]

            input_ids[:, img_ph_token_mask] = image_inputs["input_ids"].flatten(1,2)
        input_attention_mask = text_inputs['attention_mask']

        # get vision_indices
        vision_indices = torch.full(input_attention_mask.shape, self.image_tokenizer.max_vision_token_length, dtype=torch.long, device=self.device)
        if has_images:
            img_token_length = self.image_tokenizer.get_token_length(images=images)
            vision_indices[img_ph_token_mask] = self.img_indices_ph[:, :img_token_length].expand(image_inputs["input_ids"].shape[1], -1).flatten(0,1)
        else:
            vision_indices[img_gen_token_mask] = 0


        if has_images:
            continous = image_inputs["encoder_feat"]
            special_token_placeholder = torch.zeros([continous.shape[0], 1, continous.shape[2]], device=continous.device, dtype=continous.dtype)
            continous = torch.cat(
                [
                    special_token_placeholder,
                    continous,
                    special_token_placeholder
                ], dim=1
            )
            if contiguous_ignore_signs is not None:
                continous[contiguous_ignore_signs] = 0

            coninous_signal = torch.zeros([input_ids.shape[1], input_ids.shape[2], continous.shape[-1]], dtype=continous.dtype, device=continous.device)
            coninous_signal[img_ph_token_mask] = continous.flatten(0, 1).contiguous()
        else:
            coninous_signal = None

        if truncation:
            input_ids = input_ids[:, :, :max_length]
            input_attention_mask = input_attention_mask[:, :max_length]
            vision_indices = vision_indices[:, :max_length]
            if coninous_signal is not None:
                coninous_signal = coninous_signal[:, :max_length]

        if self.raw_output:
            return {
            "input_ids": input_ids.contiguous(),
            "attention_mask": input_attention_mask.contiguous(),
            "vision_indices": vision_indices.contiguous(),
            "coninous_signal": coninous_signal,
            }
        else:
            return BatchEncoding({
                "input_ids": input_ids.contiguous(),
                "attention_mask": input_attention_mask.contiguous(),
                "vision_indices": vision_indices.contiguous(),
                "coninous_signal": coninous_signal,
            })
    
    @classmethod
    def find_index(cls, tensor, value):
        return torch.nonzero(tensor==value).squeeze(-1)
    
    def batch_decode(
        self,
        token_ids: Union[List[int], List[List[int]], "torch.Tensor"],
        **kwargs,
        ):

        token_ids = token_ids.permute(1,0,2)
        return [
            self.decode(seq, **kwargs) 
            for seq in token_ids
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "torch.Tensor"],
        **kwargs,
    ):
        pure_text_token_ids_list = []
        pure_image_token_ids_list = []
        for sub_token_id in token_ids:
            pure_text_token_ids, pure_image_token_ids = self.prepare_decode(sub_token_id)
            pure_text_token_ids_list.append(pure_text_token_ids)
            pure_image_token_ids_list.append(pure_image_token_ids)
        
        pure_text_token_ids = pure_text_token_ids_list[0]
        pure_image_token_ids = pure_image_token_ids_list


        # decoding
        decoded_text = self.text_tokenizer.decode(pure_text_token_ids, **kwargs)
        decoded_image = self.image_tokenizer.decode(pure_image_token_ids)

        if len(decoded_image) == 0 or len(decoded_image[0]) == 0:
            decoded_image = None

        return {
            'language': decoded_text,
            'vision': decoded_image,
        }
    
    def prepare_decode(self, token_ids):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, device=self.device)
        
        assert len(token_ids.shape) == 1 # num_codebook, ids
        
        # img_token_mask = token_ids >= self.img_tokenizer_offset
        # filter out image tokens
        boi_token_indice = self.find_index(token_ids, self.image_tokenizer.boi_token_id)
        eoi_token_indice = self.find_index(token_ids, self.image_tokenizer.eoi_token_id)
        assert len(boi_token_indice) == len(eoi_token_indice), "Incomplete images are found during decoding."
        if len(boi_token_indice) == 0:
            # no image
            has_image = False
        else:
            has_image = True
        
        # split text and image
        token_ids = token_ids.tolist()
        if has_image:
            pure_text_token_ids = []
            pure_image_token_ids = []
            cur_idx = 0
            for boi_idx, eoi_idx in zip(boi_token_indice, eoi_token_indice):
                assert boi_idx <= eoi_idx, "EOI token occurs before BOI token"
                assert cur_idx <= boi_idx, "need debug"
                text_tokens = token_ids[cur_idx:boi_idx]
                image_tokens = token_ids[boi_idx:eoi_idx+1]
                cur_idx += len(text_tokens) + len(image_tokens)
                pure_image_token_ids.append(image_tokens)
                text_tokens.append(self.text_tokenizer.img_ph_token_id)
                pure_text_token_ids += text_tokens
            pure_text_token_ids += token_ids[cur_idx:-1]
        else:
            pure_text_token_ids = token_ids
            pure_image_token_ids = []
        
        return pure_text_token_ids, pure_image_token_ids