import os
import json
from typing import Tuple, Optional, List

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors


class GenericYOLODataset(Dataset):
    """
    Generic dataset for loading images + YOLO-format annotations and
    converting them to the format expected by PyTorch detection models.
    """
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transform: Optional[v2.Transform] = None,
        target_size: Tuple[int, int] = (300, 300),
        bg_offset: int = 0
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.bg_offset = bg_offset

        # Fallback: no augmentation, just tensor conversion.
        self.transform = transform if transform is not None else v2.Compose([
            v2.Resize(target_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        self.img_names = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)
        ])

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # boxes are now computed in ORIGINAL pixel space

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])

                    # YOLO normalized -> pixel coords in the ORIGINAL image.
                    xmin = (x_c - w / 2) * orig_w
                    ymin = (y_c - h / 2) * orig_h
                    xmax = (x_c + w / 2) * orig_w
                    ymax = (y_c + h / 2) * orig_h

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id + self.bg_offset)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        # Wrapping boxes as a tv_tensor is what lets v2 transforms detect them
        # and apply the same geometric transform as the image (flip, rotate,
        # crop, resize...) instead of leaving them untouched.
        bboxes = tv_tensors.BoundingBoxes(
            boxes_tensor, format="XYXY", canvas_size=(orig_h, orig_w)
        )

        sample = {"image": image, "boxes": bboxes, "labels": labels_tensor}
        sample = self.transform(sample)

        target = {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
        }
        return sample["image"], target


def detection_collate_fn(batch):
    return tuple(zip(*batch))


def compute_mean_std(img_dir: str, label_dir: str, batch_size: int = 32, image_size: Tuple[int, int] = (300, 300)) -> Tuple[List[float], List[float]]:
    base_transform = v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    dataset = GenericYOLODataset(img_dir=img_dir, label_dir=label_dir, transform=base_transform, target_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=2)

    n_pixels = 0
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)

    for images, _ in loader:
        images_stacked = torch.stack(images)
        b, c, h, w = images_stacked.shape
        n_pixels += b * h * w
        sum_ += images_stacked.sum(dim=[0, 2, 3])
        sum_sq += (images_stacked ** 2).sum(dim=[0, 2, 3])

    mean = sum_ / n_pixels
    std = (sum_sq / n_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def get_transforms(image_size: Tuple[int, int], mean: Optional[List[float]], std: Optional[List[float]]):
    if mean is None: mean = [0.485, 0.456, 0.406]
    if std is None: std = [0.229, 0.224, 0.225]

    train_transform = v2.Compose([
        v2.Resize(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomRotation(15),
        v2.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        v2.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),
        # Drops boxes that became degenerate (zero area, fully out of frame)
        # after crop/rotation — must run after the geometric transforms above.
        v2.SanitizeBoundingBoxes(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    val_transform = v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return train_transform, val_transform


def get_detection_loaders(
    dataset_root: str,
    batch_size: int = 4,
    num_workers: int = 2,
    image_size: Tuple[int, int] = (300, 300),
    use_computed_stats: bool = True
) -> Tuple[DataLoader, DataLoader]:

    train_img_dir = os.path.join(dataset_root, "train", "images")
    train_label_dir = os.path.join(dataset_root, "train", "labels")

    val_img_dir = os.path.join(dataset_root, "valid", "images")
    val_label_dir = os.path.join(dataset_root, "valid", "labels")
    if not os.path.exists(val_img_dir):
        val_img_dir, val_label_dir = train_img_dir, train_label_dir

    stats_path = os.path.join(dataset_root, "stats.json")

    if use_computed_stats:
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            mean, std = stats["mean"], stats["std"]
        else:
            mean, std = compute_mean_std(train_img_dir, train_label_dir, batch_size, image_size)
            with open(stats_path, "w") as f:
                json.dump({'mean': mean, 'std': std}, f, indent=4)
    else:
        mean, std = None, None

    train_transform, val_transform = get_transforms(image_size, mean, std)

    train_set = GenericYOLODataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=train_transform, target_size=image_size)
    val_set = GenericYOLODataset(img_dir=val_img_dir, label_dir=val_label_dir, transform=val_transform, target_size=image_size)

    print(f"[Data] Dataset '{os.path.basename(dataset_root)}' chargé. Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader