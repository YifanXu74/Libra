import torch
import torch.nn.functional as F

import importlib

from libra.models.libra.taming.modules.diffusionmodules.model import Encoder, Decoder
from libra.models.libra.clip_encoder import CLIPVisionTower
from libra.models.libra.taming.modules.quantization.lookup_free_quantization import LFQ

from torchvision import transforms
import warnings

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQModel(torch.nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 codebook_size=512,
                 num_codebook=2,
                 disable_loss=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.automatic_optimization = False

        self.encoder_name = ddconfig.get("encoder_name", "default")
        self.use_clip = "clip" in self.encoder_name
        self.only_auto_encoder = ddconfig.get("only_auto_encoder", False)
        if "clip" in self.encoder_name:
            select_layer = ddconfig.get("select_layer", -2)
            self.encoder = CLIPVisionTower(vision_tower=self.encoder_name, square_output=True, select_layer=select_layer)
            self.inv_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                 std = [1/v for v in self.encoder.image_processor.image_std]),
                                            transforms.Normalize(mean = [-v for v in self.encoder.image_processor.image_mean],
                                                                 std = [ 1., 1., 1. ]),
                                            ])
        else:
            self.encoder = Encoder(**ddconfig)



        self.decoder = Decoder(**ddconfig)
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        
        self.quantize = LFQ(dim=embed_dim, 
                            codebook_size=codebook_size,
                            num_codebooks=num_codebook,
                            entropy_loss_weight=0.1,
                            commitment_loss_weight=1.,
                            diversity_gamma = 2.5,
                            )

        self.quant_conv = torch.nn.Conv2d(self.encoder.vision_tower.vision_model.config.hidden_size * len(self.encoder.select_layer) if self.use_clip else ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    @property
    def device(self):
        return self.decoder.conv_in.weight.device
    
    @property
    def dtype(self):
        return self.decoder.conv_in.weight.dtype
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x, return_encoder_feat=False):
        encoder_feat = self.encoder(x)
        h = self.quant_conv(encoder_feat)
        quant, emb_loss, info = self.quantize(h)

        if return_encoder_feat:
            return quant, emb_loss, info, encoder_feat
        else:
            return quant, emb_loss, info
    
    def encode_without_quant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        return h, None, None

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.indices_to_codes(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if self.use_clip:
            return x.float()

        in_channels = self.encoder.in_channels
        if len(x.shape) == 3:
            x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            return x.float()
        
        B, S1, S2, S3 = x.shape

        if S1 == S3:
            warnings.warn("Can't figure out the input format. Set to (B, C, H, W) by default. Make sure this is correct!")
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            return x.float()

        if S1 == in_channels:
            return x.float()
        elif S3 == in_channels:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            return x.float()

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        ##########################
        # Optimize Autoencode #
        ##########################
        x = self.get_input(batch, self.image_key)

        if self.use_clip: # dddebug
            x_gt = self.clip_to_rgb(x)
        else:
            x_gt = x

        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x_gt, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        if self.only_auto_encoder:
            opt_disc.zero_grad()
        else:
            ##########################
            # Optimize Discriminator #
            ##########################
            x = self.get_input(batch, self.image_key)

            if self.use_clip: # dddebug
                x_gt = self.clip_to_rgb(x)
            else:
                x_gt = x

            xrec, qloss = self(x)
            discloss, log_dict_disc = self.loss(qloss, x_gt, xrec, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()


    def configure_optimizers(self):
        lr = self.learning_rate
        if self.use_clip:
            encoder_opt_params = list()
        else:
            encoder_opt_params = list(self.encoder.parameters())

        opt_ae = torch.optim.AdamW(
                                  encoder_opt_params+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)

        if self.use_clip: # dddebug
            x = self.clip_to_rgb(x)
            # xrec = self.clip_to_rgb(xrec) # dddebug

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def clip_to_rgb(self, x, clip=False):
        assert len(x.shape) == 4 # B, C, H, W
        # x = x.permute(0, 2, 3, 1)
        x = self.inv_transform(x)
        # x = torch.clip(x, min=0, max=1)
        x = 2.*x - 1.
        if clip:
            x = torch.clip(x, min=0., max=1.)
        return x


    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x