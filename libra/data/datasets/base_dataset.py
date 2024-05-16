import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class WebDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor


class LibraChainDataset(ChainDataset):
    def collater(self, samples):
        return self.datasets[0].collater(samples)
    
    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total


class LibraConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def set_image_size(self):
        for dataset in self.datasets:
            dataset.set_image_size()

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        # all_keys = set()
        # for s in samples:
        #     all_keys.update(s)

        # shared_keys = all_keys
        # for s in samples:
        #     shared_keys = shared_keys & set(s.keys())

        # samples_shared_keys = []
        # for s in samples:
        #     samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        # return self.datasets[0].collater(samples_shared_keys)
        return self.datasets[0].collater(samples)
