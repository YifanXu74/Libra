import torch
from libra.models.libra.taming.models.vqgan import VQModel
import math
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import os

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class ImageTokenizer(torch.nn.Module):
    def __init__(self, cfg, token_offset, **kwargs):
        super().__init__()
        config = cfg
        self.codebook_size = config.params.get("codebook_size")
        self.num_codebook = config.params.get("num_codebook")

        self.model = VQModel(ignore_keys=["loss."], **config.params)
        ckpt_path = config.get('ckpt_path', None)
        if ckpt_path is not None:
            raise NotImplementedError("Please load image tokenizer weight through params.ckpt_path.")
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            self.model.load_state_dict(sd, strict=True) # debug
        else:
            ckpt_path = config.params.get("ckpt_path")

        if "_f16_" in os.path.basename(ckpt_path) and "_f8_" in os.path.basename(ckpt_path):
            raise NotImplementedError
        if "_f16_" in os.path.basename(ckpt_path):
            self.downsample_ratio = 16
        elif "_f8_" in os.path.basename(ckpt_path):
            self.downsample_ratio = 8
        else:
            self.downsample_ratio = None

        freeze_image_tokenizer = config.get("freeze", True)
        if freeze_image_tokenizer:
            self.model.eval()
            self.model.train = disabled_train
            for key, p in self.model.named_parameters():
                p.requires_grad = False

        self.offset = token_offset
        # self.vocab_size = self.model.get_input_embeddings().weight.shape[0] + 2
        self.boi_token_id = token_offset + len(self) - 2 # need debug
        self.eoi_token_id = token_offset + len(self) - 1 # need debug
        self.max_vision_token_length = config.max_vision_token_length # TODO: only support images with a fix size now
        self.vocab_size = self.codebook_size + 2
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def dtype(self):
        return self.model.dtype
    
    def __len__(self) -> int:
        return  self.codebook_size + 2 # need debug

    def get_token_length(self, images: torch.Tensor):
        if self.downsample_ratio is None:
            return self.max_vision_token_length
        _, _, H, W = images.shape
        assert H==W
        vector_resolution = H // self.downsample_ratio
        return vector_resolution**2 + 2
    
    @torch.no_grad()
    def forward(self, x, add_boi_token=True, add_eoi_token=True, return_tensors=True, return_encoder_feat=True):
        return self.encode(x, add_boi_token=add_boi_token, add_eoi_token=add_eoi_token, return_tensors=return_tensors, return_encoder_feat=return_encoder_feat)
    
    @torch.no_grad()
    def encode(self, x, add_boi_token=True, add_eoi_token=True, return_tensors=True, return_encoder_feat=True):
        assert return_tensors
        z, _, indices, encoder_feat = self.model.encode(x, return_encoder_feat=return_encoder_feat)
        if len(indices.shape) == 3: # codebook_num=1
            indices = indices[..., None]
        indices = indices.permute(3, 0, 1, 2)
        indices += self.offset
        indices = indices.flatten(2, 3)
        boi = torch.full([indices.shape[0], indices.shape[1] if add_boi_token else 0, 1], self.boi_token_id, device=indices.device, dtype=indices.dtype)
        eoi = torch.full([indices.shape[0], indices.shape[1] if add_eoi_token else 0, 1], self.eoi_token_id, device=indices.device, dtype=indices.dtype)
        input_ids = torch.cat([boi, indices, eoi], dim=-1)

        attention_mask = torch.ones(input_ids[0].shape, dtype=torch.long, device=z.device)

        return {
            "input_ids": input_ids,
            "image_size": list(z.shape[2:]),
            "attention_mask": attention_mask,
            "encoder_feat": encoder_feat.flatten(2,3).permute(0, 2, 1).contiguous(),

        }

    @torch.no_grad()
    def decode(self, x: Union[List, torch.tensor]):
        if len(x) == 0 or len(x[0]) == 0:
            return x
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        if len(x.shape) == 2:
            x = x[None, ...]
        elif len(x.shape) == 3:
            pass
        else:
            raise NotImplementedError

        if self.boi_token_id in x:
            x = x[:, :, 1:-1] # exclude <img> and <\img> tokens
            
        # reshape input ids to 2d. need debug: currently, the image decoder only support square images
        Q, B, N = x.shape
        resolution = math.sqrt(N)
        if resolution % 1 != 0:
            raise ValueError('Input images are invalid. Currently, the image decoder only support square images.')
        H = W = int(resolution)
        x = x.reshape(Q, B, H, W)
        x = x.permute(1, 2, 3, 0)

        x = x - self.offset
        # discrete_embs = self.model.embedding(x)
        return self.model.decode_code(x)

    @classmethod
    def from_config(cls, config, **kwargs):
        model = cls(config, **kwargs)
        # ckpt_path = config.image_tokenizer.get('ckpt_path', None)
        # if ckpt_path is not None:
        #     sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        #     missing, unexpected = model.model.load_state_dict(sd, strict=True)
        return model
    
    




