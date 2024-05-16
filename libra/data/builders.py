import logging
import os
import warnings

import torch.distributed as dist
from libra.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from libra.common.registry import registry
from libra.data.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf

from libra.data.datasets.laion_dataset import LaionDataset
from libra.data.datasets.caption_datasets import CaptionDataset
from libra.data.datasets.instruction_dataset import LazySupervisedDataset

def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg

class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None, **kwargs):
        super().__init__()

        if isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

        # additional processors, each specified by a name in string.
        self.kw_processors = {}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)
        
        kw_proc_cfg = self.config.get("kw_processor")
        if kw_proc_cfg is not None:
            for name, cfg in kw_proc_cfg.items():
                self.kw_processors[name] = self._build_proc_from_cfg(cfg)
        
    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        custom_params = self.config.get("custom_params", {})
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                # debug: remove cache path now
                # if not os.path.isabs(ann_path):
                #     ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            # debug: remove cache path now
            # if not os.path.isabs(vis_path):
            #     # vis_path = os.path.join(utils.get_cache_path(), vis_path)
            #     vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                **custom_params,
            )

        return datasets


@registry.register_builder("libra_laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    def build(self):
        custom_params = self.config.get("custom_params", {})
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"  # laion dataset only has train split

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
            **custom_params,
        )
        return datasets

@registry.register_builder("libra_coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionDataset

@registry.register_builder("instruction")
class InstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LazySupervisedDataset

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        custom_params = self.config.get("custom_params", {})
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_info = build_info.get("images")

        datasets = dict()
        split = "train"  # instruction dataset only has train split

        # create datasets
        dataset_cls = self.train_dataset_cls

        # annotation path
        ann_path = ann_info.storage
        assert isinstance(ann_path, str)

        # visual data storage path
        vis_path = vis_info.storage
        
        datasets[split] = dataset_cls(
                ann_path=ann_path,
                vis_processor=self.vis_processors[split],
                vis_root=vis_path,
                **custom_params,
            )

        return datasets