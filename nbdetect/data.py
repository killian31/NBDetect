import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .model import LABEL_TO_INDEX


@dataclass(frozen=True)
class Record:
    image_path: Path
    label: str


def load_records(dataset_root: Path) -> List[Record]:
    """Walk every session directory and read annotations.csv entries."""
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist.")

    records: List[Record] = []
    for session_dir in sorted(dataset_root.iterdir()):
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
        raise RuntimeError(f"No annotated samples found in {dataset_root}.")
    return records


def split_records(
    records: Sequence[Record], val_ratio: float = 0.2, seed: int = 13
) -> Tuple[List[Record], List[Record]]:
    """Stratified split to maintain label balance."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    buckets: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        buckets[record.label].append(record)

    rng = random.Random(seed)
    train_records: List[Record] = []
    val_records: List[Record] = []

    for label, bucket in buckets.items():
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        val_count = max(1, int(len(bucket_copy) * val_ratio))
        val_records.extend(bucket_copy[:val_count])
        train_records.extend(bucket_copy[val_count:])

    if not train_records or not val_records:
        raise RuntimeError("Not enough data to create stratified split.")

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def create_transforms(image_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)],
                p=0.9,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


class NailBitingDataset(Dataset):
    def __init__(self, records: Sequence[Record], transform: transforms.Compose) -> None:
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
