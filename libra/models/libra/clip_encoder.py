import torch
import torch.nn as nn

from libra.models.clip import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import math
from collections.abc import Iterable

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False, square_output=False, select_layer=-2):
        super().__init__()

        self.square_output = square_output
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = select_layer
        self.select_feature = 'patch'

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if isinstance(self.select_layer, Iterable):
            image_features = []
            for idx in self.select_layer:
                image_features.append(image_forward_outs.hidden_states[idx])
            image_features = torch.cat(image_features, dim=-1)
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def reshape_to_square(self, feats):
        B, N, C = feats.shape
        H = W = int(math.sqrt(N))
        assert H*W == N
        return feats.view(B, H, W, C).permute(0, 3, 1, 2)
    
    @torch.no_grad()
    def forward(self, images, square_output=None):
        square_output = self.square_output if square_output is None else square_output
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(self.dtype)
                if square_output:
                    image_feature = self.reshape_to_square(image_feature)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(self.dtype)
            if square_output:
                image_features = self.reshape_to_square(image_features)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
