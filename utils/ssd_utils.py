import torch
from typing import Dict, Any, Optional, Union, Tuple, List

def generate_ssd_priors(
    feature_map_sizes: List[Tuple[int, int]],
    image_size: Tuple[int, int] = (300, 300),
    min_sizes: Optional[List[float]] = [30, 60, 111, 162, 213, 264],
    max_sizes: Optional[List[float]] = [60, 111, 162, 213, 264, 315],
    aspect_ratios: Optional[List[List[float]]] = [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    clip: bool = True,
) -> torch.Tensor:
    """
    Génère dynamiquement les boîtes ancres (priors) pour SSD.

    Args:
        feature_map_sizes: Liste des paires (H, W) pour chaque niveau de feature map.
        image_size: Tuple (hauteur, largeur) de l'image d'entrée.
        min_sizes: Tailles minimales des ancres par niveau (en pixels).
        max_sizes: Tailles maximales des ancres par niveau (en pixels).
        aspect_ratios: Liste des ratios d'aspect par niveau (ex: [[2], [2, 3], ...]).
        clip: Si True, borne les coordonnées dans [0, 1].
    """
    num_levels = len(feature_map_sizes)
    img_h, img_w = image_size

    # Si min_sizes/max_sizes ne sont pas définis, calcul linéaire des échelles
    if min_sizes is None or max_sizes is None:
        min_ratio, max_ratio = 15, 90  # 15% à 90% de la taille de l'image
        step = (max_ratio - min_ratio) / max(1, num_levels - 1)
        min_sizes = [img_w * (min_ratio + i * step) / 100.0 for i in range(num_levels)]
        max_sizes = [img_w * (min_ratio + (i + 1) * step) / 100.0 for i in range(num_levels)]

    # Si aspect_ratios non fourni, configuration standard 2 ou 2+3
    if aspect_ratios is None:
        aspect_ratios = [[2] if i in (0, num_levels - 1) else [2, 3] for i in range(num_levels)]

    priors = []
    for k, (f_h, f_w) in enumerate(feature_map_sizes):
        min_s = min_sizes[k]
        max_s = max_sizes[k]
        ratios = aspect_ratios[k]

        for i in range(f_h):
            for j in range(f_w):
                # Centre normalisé [0, 1]
                cx = (j + 0.5) / f_w
                cy = (i + 0.5) / f_h

                # 1. Ancre carrée basée sur min_size
                priors.append([cx, cy, min_s / img_w, min_s / img_h])

                # 2. Ancre carrée intermédiaire (moyenne géométrique)
                s_k_prime = (min_s * max_s) ** 0.5
                priors.append([cx, cy, s_k_prime / img_w, s_k_prime / img_h])

                # 3. Ancres rectangulaires selon ratios d'aspect
                for ar in ratios:
                    ar_sqrt = ar ** 0.5
                    priors.append([cx, cy, (min_s * ar_sqrt) / img_w, (min_s / ar_sqrt) / img_h])
                    priors.append([cx, cy, (min_s / ar_sqrt) / img_w, (min_s * ar_sqrt) / img_h])

    priors = torch.tensor(priors, dtype=torch.float32)
    if clip:
        priors = torch.clamp(priors, 0.0, 1.0)

    return priors

# def generate_ssd_priors() -> torch.Tensor:
#     """ Génère les 8732 boîtes ancres par défaut pour SSD300 """
#     feature_maps = [37, 18, 9, 5, 3, 1]
#     min_sizes = [30, 60, 111, 162, 213, 264]
#     max_sizes = [60, 111, 162, 213, 264, 315]
#     aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

#     priors = []
#     for k, f in enumerate(feature_maps):
#         for i in range(f):
#             for j in range(f):
#                 cx = (j + 0.5) / f
#                 cy = (i + 0.5) / f

#                 s_k = min_sizes[k] / 300.0
#                 priors.append([cx, cy, s_k, s_k])

#                 s_k_prime = (s_k * (max_sizes[k] / 300.0)) ** 0.5
#                 priors.append([cx, cy, s_k_prime, s_k_prime])

#                 for ar in aspect_ratios[k]:
#                     priors.append([cx, cy, s_k * (ar ** 0.5), s_k / (ar ** 0.5)])
#                     priors.append([cx, cy, s_k / (ar ** 0.5), s_k * (ar ** 0.5)])

#     priors = torch.tensor(priors, dtype=torch.float32)
#     return torch.clamp(priors, 0.0, 1.0)


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    A, B = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union