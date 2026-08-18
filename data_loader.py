import os
import torch
from PIL import Image
import torchvision.transforms as T
from typing import Tuple, Optional
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, List

class GenericYOLODataset(Dataset):
    """
    Dataset générique pour charger des images et des annotations au format YOLO 
    et les convertir au format attendu par les modèles de détection PyTorch.
    """
    def __init__(
        self, 
        img_dir: str, 
        label_dir: str, 
        transform: Optional[T.Compose] = None, 
        target_size: Tuple[int, int] = (300, 300),
        bg_offset: int = 0
    ):
        """
        Args:
            img_dir (str): Chemin vers le dossier contenant les images.
            label_dir (str): Chemin vers le dossier contenant les fichiers .txt (YOLO).
            transform (callable, optional): Pipeline de transformations torchvision.
            target_size (tuple): Taille de redimensionnement cible (Largeur, Hauteur).
            bg_offset (int): Décalage à appliquer aux IDs de classe (1 par défaut car 0 = Background en PyTorch).
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.bg_offset = bg_offset
        
        # On accepte le transform externe, sinon conversion minimale en tenseur
        self.transform = transform if transform is not None else T.ToTensor()
        
        # Filtrer et trier les images valides pour garantir l'alignement
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        self.img_names = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)
        ])

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        # 1. Chargement de l'image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 2. Application des transformations (Redimensionnement automatique inclus si présent dans le transform)
        image_tensor = self.transform(image)

        # 3. Recherche du fichier de label YOLO associé (même nom, extension .txt)
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
                        
                    # Format YOLO officiel : class_id, x_center, y_center, width, height (normalisés entre 0 et 1)
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])

                    # Conversion 1 : YOLO -> Pascal VOC (coordonnées normalisées xmin, ymin, xmax, ymax)
                    xmin = x_c - w / 2
                    ymin = y_c - h / 2
                    xmax = x_c + w / 2
                    ymax = y_c + h / 2

                    # Conversion 2 : Normalisé [0, 1] -> Pixel absolu (ex: basculer sur l'échelle 300x300)
                    xmin *= self.target_size[0]
                    ymin *= self.target_size[1]
                    xmax *= self.target_size[0]
                    ymax *= self.target_size[1]

                    boxes.append([xmin, ymin, xmax, ymax])
                    
                    # Appliquer le décalage pour le Background de PyTorch (class_id + 1)
                    labels.append(class_id + self.bg_offset)

        # Gestion des images de fond (sans aucun objet détecté)
        if len(boxes) == 0:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros(0, dtype=torch.int64)
        else:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target_labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": target_boxes,
            "labels": target_labels
        }

        return image_tensor, target
    


def detection_collate_fn(batch):
    return tuple(zip(*batch))

def compute_mean_std(img_dir: str, label_dir: str, batch_size: int = 32, image_size: Tuple[int, int] = (300, 300)) -> Tuple[List[float], List[float]]:
    # Utilise le dataset générique en mode brut pour calculer les stats
    base_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
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

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(15),
        # transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform

def get_detection_loaders(
    dataset_root: str,  # Exemple: "data/Chess" ou "data/Vehicles"
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

    # Instanciation générique
    train_set = GenericYOLODataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=train_transform, target_size=image_size)
    val_set = GenericYOLODataset(img_dir=val_img_dir, label_dir=val_label_dir, transform=val_transform, target_size=image_size)

    print(f"[Data] Dataset '{os.path.basename(dataset_root)}' chargé. Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader