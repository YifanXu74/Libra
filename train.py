# ----------------------------------------------------------------- #
#######################################
# customized system settings
# must be called at the very beginning
#######################################
from utils.train_utils import initialize_system_settings_for_training
initialize_system_settings_for_training()
# ----------------------------------------------------------------- #

import argparse
import torch
from omegaconf import OmegaConf

from transformers.trainer import TrainingArguments
from trainer import LibraTrainer
from libra.common.config import Config
from libra.common.registry import registry
from utils.train_utils import setup_logger, DebugModel
from libra.data.utils import concat_datasets, reorg_datasets_by_split

# imports modules for registration
from libra.data import *
from libra.models import *

def build_model(cfg, no_model=False):
    if no_model: # for debug
        return DebugModel()
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    if cfg.run_cfg.get("bf16", False):
        model = model.to(torch.bfloat16)
    return model

def build_dataset_and_collate_fns(cfg, **kwargs):
    '''
    TODO: support multi-datasets with dataset ratios
    '''
    ###########################
    # build base datasets
    ###########################
    datasets = dict()
    datasets_config = cfg.datasets_cfg
    assert len(datasets_config) > 0, "At least one dataset has to be specified."
    
    for name in datasets_config:
        dataset_config = datasets_config[name]

        builder = registry.get_builder_class(name)(dataset_config, **kwargs)
        dataset = builder.build_datasets()

        datasets[name] = dataset
    
    ###########################
    # concatenate datasets 
    ###########################
    datasets = reorg_datasets_by_split(datasets)
    datasets = concat_datasets(datasets)

    ###########################
    # get collator
    ###########################
    split_names = sorted(datasets.keys())
    collate_fns = {}
    for split in split_names:
        dataset = datasets[split]
        if isinstance(dataset, tuple) or isinstance(dataset, list):
            collate_fns[split] = [getattr(d, "collater", None) for d in dataset]
        else:
            collate_fns[split] = getattr(dataset, "collater", None)

    return datasets, collate_fns

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", default=None, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file "
        "e.g., --options run.resume_from_checkpoint=True"
    )
    parser.add_argument("--local_rank", default=-1, type=int, help="for distributed training")
    args = parser.parse_args()
    return args

def resolve_special_types(cfg):
    '''
    OmegaConf do not preserve some input types (e.g., dict).
    '''
    special_types = {}
    if "accelerator_config" in cfg.run_cfg:
        accelerator_config = cfg.run_cfg.pop("accelerator_config")
        special_types["accelerator_config"] = OmegaConf.to_container(accelerator_config, resolve=True)
    return special_types

if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    cfg = Config(args)
    cfg.pretty_print()

    model = build_model(cfg)
    datasets, collate_fns = build_dataset_and_collate_fns(cfg)

    special_types = resolve_special_types(cfg)
    hf_args = TrainingArguments(**cfg.run_cfg, **special_types,
                                local_rank=args.local_rank,
                                )
    trainer = LibraTrainer(
        model=model,
        args=hf_args,
        train_dataset=datasets['train'],
        data_collator=collate_fns['train'],
        eval_dataset=datasets.get("val", None),
        )
    
    trainer.train()