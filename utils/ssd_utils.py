import torch

def generate_ssd_priors() -> torch.Tensor:
    """ Génère les 8732 boîtes ancres par défaut pour SSD300 """
    feature_maps = [38, 19, 10, 5, 3, 1]
    min_sizes = [30, 60, 111, 162, 213, 264]
    max_sizes = [60, 111, 162, 213, 264, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    priors = []
    for k, f in enumerate(feature_maps):
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f

                s_k = min_sizes[k] / 300.0
                priors.append([cx, cy, s_k, s_k])

                s_k_prime = (s_k * (max_sizes[k] / 300.0)) ** 0.5
                priors.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    priors.append([cx, cy, s_k * (ar ** 0.5), s_k / (ar ** 0.5)])
                    priors.append([cx, cy, s_k / (ar ** 0.5), s_k * (ar ** 0.5)])

    priors = torch.tensor(priors, dtype=torch.float32)
    return torch.clamp(priors, 0.0, 1.0)


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