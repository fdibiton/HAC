from __future__ import annotations

import copy
import glob
import random
from typing import Callable

import webdataset as wds
import wordsegment as ws
from loguru import logger
from torch.utils.data import IterableDataset
from torchvision import transforms as T

import hac.utils.distributed as dist
from hac.utils.transforms import get_transform
from hac.data.prompts import AddRandomPrompts

ws.load()


def plot_img(img_tensor, title=""):
    import matplotlib.pyplot as plt
    import numpy as np

    img = img_tensor.squeeze().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


class ImageTextWebDataset(IterableDataset):
    """
    Iterable dataset that serves instances from a lot of TAR file shards.
    This class uses `WebDataset <https://github.com/webdataset/webdataset>`_
    internally, and expects TAR files to be arranged in a compatible format.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: A callable to transform a single dataset dict (image and
                annotations). May implement data augmentation and tokenization.
            buffer_size: Size of the internal buffer of instances. Data is read
                sequentially from TAR files into this buffer and served randomly.
                Shuffling will be disabled if this is set to zero.
            infinite_stream: Yield an infinite stream of instances if this is
                True. In such cases, the user must terminate this iterator manually
                (e.g. run a fixed sized for-loop in training code).
            seed: Random seed for buffer shuffling. If provided, this dataloader
                will load batches deterministically across different runs (only if
                batch size and number of GPUs/CPUs are same). This seed can either
                be same or different per GPU process for multi-GPU training.
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        # Convert a single path (glob) to a list.
        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        # Expand all glob patterns to list a full list of individual TAR files.
        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        # Sort paths; webdataset performs a deterministic shuffle (internally).
        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        # Shard the TAR file paths as per number of GPU processes to avoid loading
        # duplicates.
        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        self.tarfiles = self.tarfiles[_rank::_world_size]
        logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        # Decode images using PIL and apply custom mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            # Sample an infinite stream of dataset dicts.
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            # Run for one epoch and stop:
            yield from pipeline
            
class GroundedDatasetTarMapper:
    """
    Mapper to pre-process image-text instances from Grounded dataset TAR files.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
        prompts_p=0,
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
        """
        if isinstance(image_transform, str):
            self.image_transform = get_transform(image_transform) # e.g. "stacked_ra_3_5"
        else:
            self.image_transform = T.Compose(image_transform)
        if prompts_p > 0:
            self.prompt_adder = AddRandomPrompts(p=prompts_p)
        else:
            self.prompt_adder = None

    def __call__(self, dataset_dict: dict):
        num_boxes = int(dataset_dict["numparents.txt"])
        random_box = random.randrange(num_boxes)
        text = dataset_dict["child.txt"]
        box_text = dataset_dict[f"parent{random_box:03d}.txt"]
        if self.prompt_adder is not None and len(box_text) < len(text):
            box_text = self.prompt_adder.apply(box_text)
        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": text,
            "box_image": self.image_transform(dataset_dict[f"parent{random_box:03d}.jpg"]),
            "box_text": box_text,
        }
