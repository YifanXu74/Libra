from typing import List
from transformers.trainer import Trainer, is_sagemaker_mp_enabled, get_parameter_names, logger, nn
from libra.models.llama.modeling_llama import LlamaRMSNorm


ALL_LAYERNORM_LAYERS = [nn.LayerNorm, LlamaRMSNorm]

class LibraTrainer(Trainer):    
    def rewrite_optimizer_params(self, grouped_params):
        new_grouped_params = []
        for group in grouped_params:
            if "lr_scale" in group.keys():
               lr_scale = group.pop("lr_scale")
               group["lr"] = self.args.learning_rate * lr_scale

            if "use_weight_decay" in group.keys():
                use_weight_decay = group.pop("use_weight_decay")
                if use_weight_decay:
                    group["weight_decay"] = self.args.weight_decay
                else:
                    group["weight_decay"] = 0.0

            new_grouped_params.append(group)

        return new_grouped_params

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters

    def create_optimizer(self):
        '''
        add a new function to enable different learning rate scales for each parameter 
        through opt_model.get_optimizer_parameters()
        '''
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            if hasattr(opt_model, "get_optimizer_parameters"):
                optimizer_grouped_parameters = self.rewrite_optimizer_params(opt_model.get_optimizer_parameters())
            else:
                decay_parameters = self.get_decay_parameter_names(opt_model)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


class EMA():
    '''
    NOTE: NOT AVAILABLE
    
    Basic usage:
        1. Initialize
            ema = EMA(model, 0.999)
            ema.register()

        2. During training: update shadow weights after update the parameters 
            def train():
                optimizer.step()
                ema.update()

        3. 1) Before evaluation: apply shadow weights
           2) After evaluation: recover the original parameters
            def evaluate():
                ema.apply_shadow()
                # evaluate
                ema.restore()
    '''
    def __init__(self, model, decay=0.99, use_cpu=True):
        self.model = model
        self.decay = decay
        self.use_cpu = use_cpu
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.use_cpu:
                    self.shadow[name] = self.shadow[name].to("cpu")

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = self.shadow[name].to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                del new_average
                if self.use_cpu:
                    self.shadow[name] = self.shadow[name].to("cpu")

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                if self.use_cpu:
                    self.backup[name] = self.backup[name].to("cpu")
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].to(param.device)
        self.backup = {}