import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calcule l'Intersection over Union (IoU) entre deux ensembles de boîtes.
    
    Args:
        boxes1 (Tensor[N, 4]): Boîtes au format [xmin, ymin, xmax, ymax]
        boxes2 (Tensor[M, 4]): Boîtes au format [xmin, ymin, xmax, ymax]
        
    Returns:
        Tensor[N, M]: Matrice des IoU calculés entre chaque boîte de boxes1 et boxes2
    """
    # Calcul des aires des boîtes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Coordonnées des rectangles d'intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-8)


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Associe les prédictions du modèle aux vérités terrain (GT) via l'IoU.
    Utile pour identifier les Vrais Positifs (TP) et Faux Positifs (FP).
    
    Returns:
        TP (Tensor[N]): Masque binaire (1 si la prédiction est un TP, 0 sinon)
        FP (Tensor[N]): Masque binaire (1 si la prédiction est un FP, 0 sinon)
        matched_gt_indices (Tensor[N]): Index du GT associé à chaque prédiction (-1 si aucun match)
    """
    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]

    tp = torch.zeros(num_preds, dtype=torch.float32)
    fp = torch.zeros(num_preds, dtype=torch.float32)
    matched_gt_indices = torch.full((num_preds,), -1, dtype=torch.long)

    if num_gts == 0:
        fp[:] = 1.0
        return tp, fp, matched_gt_indices

    if num_preds == 0:
        return tp, fp, matched_gt_indices

    # Tri des prédictions par score décroissant pour traiter les plus confiantes en premier
    indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[indices]
    pred_labels = pred_labels[indices]

    ious = box_iou(pred_boxes, gt_boxes)  # Matrice [N, M]
    gt_detected = torch.zeros(num_gts, dtype=torch.bool)

    for i in range(num_preds):
        best_iou = -1.0
        best_gt_idx = -1

        for j in range(num_gts):
            # L'appariement ne se fait que si la classe correspond
            if pred_labels[i] == gt_labels[j]:
                iou_val = ious[i, j].item()
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j

        # Validation du match avec le seuil d'IoU
        if best_iou >= iou_threshold and best_gt_idx != -1:
            if not gt_detected[best_gt_idx]:
                tp[i] = 1.0  # Vrai Positif
                gt_detected[best_gt_idx] = True
                matched_gt_indices[i] = best_gt_idx
            else:
                fp[i] = 1.0  # Double détection = Faux Positif
        else:
            fp[i] = 1.0  # Mauvaise localisation ou mauvaise classe = Faux Positif

    # On remet les masques TP/FP dans l'ordre originel des prédictions
    inv_indices = torch.argsort(indices)
    return tp[inv_indices], fp[inv_indices], matched_gt_indices[inv_indices]


def compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calcule l'Average Precision (AP) sous la courbe Precision-Recall (méthode d'intégration COCO).
    """
    # On force le départ à 0 et la fin à 1 pour le rappel
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Enveloppe de précision monotone décroissante (on prend le max à droite de chaque point)
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # Calcul de l'intégrale (Aire sous la courbe par segments rectangulaires)
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def compute_dataset_tp_fp(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Rassemble les prédictions de tout le dataset et calcule les TP, FP, 
    les scores et le nombre total de vérités terrain (GT) par classe.
    """
    # Listes pour accumuler les stats par classe
    cls_tp: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
    cls_fp: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
    cls_scores: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
    cls_gt_counts = torch.zeros(num_classes, dtype=torch.long)
    
    # Structure pour stocker les informations de la matrice de confusion
    # (true_class, pred_class)
    cm_pairs = []

    for pred, target in zip(predictions, targets):
        p_boxes, p_labels, p_scores = pred["boxes"], pred["labels"], pred["scores"]
        g_boxes, g_labels = target["boxes"], target["labels"]

        # Compter les GTs par classe
        for c in range(num_classes):
            cls_gt_counts[c] += (g_labels == c).sum().item()

        # Appariement géométrique IoU
        tp, fp, matched_gt_indices = match_predictions_to_gt(
            p_boxes, p_labels, p_scores, g_boxes, g_labels, iou_threshold
        )

        gt_detected = torch.zeros(g_boxes.shape[0], dtype=torch.bool)

        # Ventilation par classe pour l'AP et la Matrice de Confusion
        for i in range(p_labels.shape[0]):
            c = p_labels[i].item()
            if c >= num_classes:
                continue

            cls_tp[c].append(tp[i].item())
            cls_fp[c].append(fp[i].item())
            cls_scores[c].append(p_scores[i].item())

            # Logique Matrice de Confusion
            gt_idx = matched_gt_indices[i].item()
            if tp[i] == 1.0 and gt_idx != -1:
                cm_pairs.append((g_labels[gt_idx].item(), c))
                gt_detected[gt_idx] = True
            elif fp[i] == 1.0:
                if gt_idx != -1: # Confusion de label
                    cm_pairs.append((g_labels[gt_idx].item(), c))
                    gt_detected[gt_idx] = True
                else: # FP sur du vide
                    cm_pairs.append((num_classes, c))

        # Récupération des Faux Négatifs (GT manqués) pour la matrice de confusion
        for j in range(g_boxes.shape[0]):
            if not gt_detected[j]:
                cm_pairs.append((g_labels[j].item(), num_classes))

    return {
        "cls_tp": cls_tp,
        "cls_fp": cls_fp,
        "cls_scores": cls_scores,
        "cls_gt_counts": cls_gt_counts,
        "cm_pairs": cm_pairs
    }


def compute_detection_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Wrapper global calculant la mAP et les APs par classe.
    """
    data = compute_dataset_tp_fp(predictions, targets, num_classes, iou_threshold)
    aps = []

    for c in range(num_classes):
        scores = np.array(data["cls_scores"][c])
        if len(scores) == 0:
            aps.append(0.0)
            continue

        tp = np.array(data["cls_tp"][c])
        fp = np.array(data["cls_fp"][c])
        total_gt = data["cls_gt_counts"][c].item()

        if total_gt == 0:
            continue

        # Tri complet par score décroissant (Crucial pour l'AP !)
        sort_idx = np.argsort(scores)[::-1]
        tp, fp = tp[sort_idx], fp[sort_idx]

        # Sommes cumulées pour obtenir l'évolution des TP et FP
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        # Calcul des vecteurs Précision et Rappel
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        recalls = tp_cum / (total_gt + 1e-8)

        # Intégration de l'aire sous la courbe (votre fonction)
        ap = compute_average_precision(recalls, precisions)
        aps.append(ap)

    return {"mAP": float(np.mean(aps)) if aps else 0.0}


def detection_precision_recall_f1(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcule les métriques F1, Précision et Rappel (Micro & Macro) via la fonction noyau.
    """
    data = compute_dataset_tp_fp(predictions, targets, num_classes, iou_threshold)
    
    tp_per_class = torch.tensor([sum(data["cls_tp"][c]) for c in range(num_classes)])
    fp_per_class = torch.tensor([sum(data["cls_fp"][c]) for c in range(num_classes)])
    gt_per_class = data["cls_gt_counts"]

    # Métriques Micro (Globales)
    total_tp = tp_per_class.sum().item()
    total_fp = fp_per_class.sum().item()
    total_gt = gt_per_class.sum().item()

    precision_micro = total_tp / (total_tp + total_fp + 1e-8)
    recall_micro = total_tp / (total_gt + 1e-8)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)

    # Métriques Macro (Moyenne des classes)
    precisions_c = tp_per_class / (tp_per_class + fp_per_class + 1e-8)
    recalls_c = tp_per_class / (gt_per_class + 1e-8)
    f1s_c = 2 * (precisions_c * recalls_c) / (precisions_c + recalls_c + 1e-8)

    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precisions_c.mean().item(),
        "recall_macro": recalls_c.mean().item(),
        "f1_macro": f1s_c.mean().item()
    }


def detection_confusion_matrix_torch(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Génère la matrice de confusion (C+1, C+1) en utilisant la fonction noyau.
    """
    data = compute_dataset_tp_fp(predictions, targets, num_classes, iou_threshold)
    cm = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.long)
    
    for true_cls, pred_cls in data["cm_pairs"]:
        cm[true_cls, pred_cls] += 1
        
    return cm


def raw_compute_map(cls_tp: dict, cls_fp: dict, cls_scores: dict, cls_gt_counts: torch.Tensor, num_classes: int, **kwargs) -> Dict[str, float]:
    """Calcule la mAP directement à partir des TP/FP/Scores pré-extraits."""
    aps = []
    for c in range(num_classes):
        scores = np.array(cls_scores[c])
        total_gt = cls_gt_counts[c].item()
        if len(scores) == 0 or total_gt == 0:
            aps.append(0.0)
            continue

        tp = np.array(cls_tp[c])
        fp = np.array(cls_fp[c])

        # Tri par score unique pour cette classe
        sort_idx = np.argsort(scores)[::-1]
        tp, fp = tp[sort_idx], fp[sort_idx]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        recalls = tp_cum / (total_gt + 1e-8)

        ap = compute_average_precision(recalls, precisions)
        aps.append(ap)

    return {"mAP": float(np.mean(aps)) if aps else 0.0}


def raw_compute_precision_recall_f1(cls_tp: dict, cls_fp: dict, cls_gt_counts: torch.Tensor, num_classes: int, **kwargs) -> Dict[str, float]:
    """Calcule le F1-Score directement à partir des TP/FP pré-extraits."""
    tp_per_class = torch.tensor([sum(cls_tp[c]) for c in range(num_classes)])
    fp_per_class = torch.tensor([sum(cls_fp[c]) for c in range(num_classes)])

    total_tp = tp_per_class.sum().item()
    total_fp = fp_per_class.sum().item()
    total_gt = cls_gt_counts.sum().item()

    precision_micro = total_tp / (total_tp + total_fp + 1e-8)
    recall_micro = total_tp / (total_gt + 1e-8)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)

    precisions_c = tp_per_class / (tp_per_class + fp_per_class + 1e-8)
    recalls_c = tp_per_class / (cls_gt_counts + 1e-8)
    f1s_c = 2 * (precisions_c * recalls_c) / (precisions_c + recalls_c + 1e-8)

    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precisions_c.mean().item(),
        "recall_macro": recalls_c.mean().item(),
        "f1_macro": f1s_c.mean().item()
    }