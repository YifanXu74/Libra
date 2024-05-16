import re
import torch
from libra.common.registry import registry
from libra.data.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import torch
from transformers import CLIPImageProcessor


def remove_html_tags(val: str) -> str:
    # Remove all HTML tags using regex
    val = re.sub(r"<.*?>", "", val)
    return val


class VQGANNormalize(torch.nn.Module):
    def forward(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return 2.*tensor - 1.
    

class LibraResize(transforms.Resize):
    def forward(self, img, size=None, **kwargs):
        return F.resize(img, size if size else self.size, self.interpolation, self.max_size, self.antialias)

class LibraCenterCrop(transforms.CenterCrop):
    def forward(self, img, size=None, **kwargs):
        return F.center_crop(img, size if size else self.size)
    
class LibraToTensor(transforms.ToTensor):
    def __call__(self, pic, **kwargs):
        return F.to_tensor(pic)

class LibraCompose(transforms.Compose):
    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, **kwargs)
        return img

class Expand2Square(torch.nn.Module):
    def __init__(self, background_color=(0,0,0)):
        super().__init__()
        self.background_color =background_color

    def forward(self, img, **kwargs):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.background_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.background_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

class Identity:
    def __call__(self, img, **kwargs):
        return img


@registry.register_processor("libra_image_eval")
class LibraEvalImageProcessor(BaseProcessor):
    def __init__(
        self, pretrained_path,
        ):
        self.transform = self.build_transforms(pretrained_path)

    @classmethod
    def build_transforms(cls, pretrained_path):
        processor = CLIPImageProcessor.from_pretrained(pretrained_path)
        background_color = tuple(int(x*255) for x in processor.image_mean)
        return LibraCompose([Expand2Square(background_color), processor])

    def __call__(self, item, image_size=None):
        output = self.transform(item, return_tensors='pt')["pixel_values"][0]
        return output

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        pretrained_path = cfg.get("pretrained_path", None)
        return cls(
            pretrained_path=pretrained_path,
        )


@registry.register_processor("libra_image")
class LibraImageProcessor(BaseProcessor):
    def __init__(
        self, pretrained_path,
        ):
        self.transform = CLIPImageProcessor.from_pretrained(pretrained_path)

    @classmethod
    def build_transforms(cls, pretrained_path):
        return CLIPImageProcessor.from_pretrained(pretrained_path)

    def __call__(self, item, image_size=None):
        output = self.transform(item, return_tensors='pt')["pixel_values"][0]
        return output

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        pretrained_path = cfg.get("pretrained_path", None)
        return cls(
            pretrained_path=pretrained_path,
        )
    

@registry.register_processor("libra_caption")
class LibraCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50, lowercase=False, remove_html=True):
        self.prompt = prompt
        self.max_words = max_words
        self.lowercase = lowercase
        self.remove_html = remove_html

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 256)
        lowercase = cfg.get("lowercase", False)
        remove_html = cfg.get("remove_html", True)

        return cls(prompt=prompt, max_words=max_words, lowercase=lowercase, remove_html=remove_html)

    def pre_caption(self, caption):
        if self.remove_html:
            caption = remove_html_tags(caption)
        caption = re.sub(
            r"([*#~])",
            " ",
            caption,
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        # caption = re.sub(
        #     r"([©™])",
        #     "",
        #     caption,
        # )
        # caption = caption.rstrip("\n")
        caption = caption.strip()

        if self.lowercase:
            caption = caption.lower()

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption