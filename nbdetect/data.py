import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .model import LABEL_TO_INDEX


@dataclass(frozen=True)
class Record:
    image_path: Path
    label: str


def load_split_records(dataset_root: Path, split: str) -> List[Record]:
    """Read annotated sessions under dataset_root/<split>."""
    dataset_root = dataset_root.expanduser().resolve()
    split_dir = (dataset_root / split).resolve()
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist.")

    records: List[Record] = []
    for session_dir in sorted(split_dir.iterdir()):
        annotations_csv = session_dir / "annotations.csv"
        images_dir = session_dir / "images"
        if not annotations_csv.exists() or not images_dir.exists():
            continue
        with annotations_csv.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                label = row.get("label")
                filename = row.get("filename")
                if label not in LABEL_TO_INDEX or not filename:
                    continue
                image_path = images_dir / filename
                if image_path.exists():
                    records.append(Record(image_path=image_path, label=label))
    if not records:
        raise RuntimeError(f"No annotated samples found in {split_dir}.")
    return records


def create_transforms(image_size: int = 224, min_size: int = 128, max_size: int = 720):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = v2.Compose(
        [
            v2.RandomResize((min_size, max_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
                    )
                ],
                p=0.9,
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            v2.ToTensor(),
            normalize,
        ]
    )
    eval_tf = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


class NailBitingDataset(Dataset):
    def __init__(self, records: Sequence[Record], transform: v2.Compose) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        tensor = self.transform(image)
        label = torch.tensor(LABEL_TO_INDEX[record.label], dtype=torch.long)
        return tensor, label
