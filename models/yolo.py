import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import nms
from core.config import Config
from core.model_base import Model  # Ton modèle de base


class YOLOHead(nn.Module):
    """ Tête de prédiction YOLO customisée projetant sur une grille S x S """
    def __init__(self, in_channels: int, num_classes: int, grid_size: int = 10):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.out_channels = 1 + 4 + num_classes
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, self.out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.conv(x)


class SimplifiedYoloLoss(nn.Module):
    """ Perte YOLO renvoyant un dictionnaire compatible avec ton Trainer """
    def __init__(self, num_classes: int, grid_size: int = 10, coord_weight: float = 5.0, noobj_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.coord_weight = coord_weight
        self.noobj_weight = noobj_weight
        
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predictions, targets, img_size=(300, 300)):
        B, _, S, S = predictions.shape
        
        obj_mask = torch.zeros((B, S, S), device=predictions.device)
        target_boxes = torch.zeros((B, 4, S, S), device=predictions.device)
        target_classes = torch.zeros((B, S, S), dtype=torch.long, device=predictions.device)
        
        for b in range(len(targets)):
            boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]
            
            for idx in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[idx]
                label = labels[idx]
                
                xc = (xmin + xmax) / 2.0 / img_size[0]
                yc = (ymin + ymax) / 2.0 / img_size[1]
                w = (xmax - xmin) / img_size[0]
                h = (ymax - ymin) / img_size[1]
                
                i = min(max(int(xc * S), 0), S - 1)
                j = min(max(int(yc * S), 0), S - 1)
                
                obj_mask[b, j, i] = 1.0
                target_boxes[b, :, j, i] = torch.tensor([xc, yc, w, h], device=predictions.device)
                target_classes[b, j, i] = label

        pred_obj = predictions[:, 0, :, :]
        pred_boxes = predictions[:, 1:5, :, :]
        pred_classes = predictions[:, 5:, :, :]
        
        has_obj = (obj_mask == 1.0)
        no_obj = (obj_mask == 0.0)
        
        loss_obj = self.bce(pred_obj[has_obj], obj_mask[has_obj]) / B
        loss_noobj = (self.bce(pred_obj[no_obj], obj_mask[no_obj]) * self.noobj_weight) / B
        
        if has_obj.sum() > 0:
            p_boxes = pred_boxes.permute(0, 2, 3, 1)[has_obj]
            t_boxes = target_boxes.permute(0, 2, 3, 1)[has_obj]
            loss_box = (self.mse(p_boxes, t_boxes) * self.coord_weight) / B
            
            p_classes = pred_classes.permute(0, 2, 3, 1)[has_obj]
            t_classes = target_classes[has_obj]
            loss_class = self.ce(p_classes, t_classes) / B
        else:
            loss_box = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            loss_class = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
        # 🎯 C'EST ICI LA CLÉ : On renvoie un dictionnaire pour ton Trainer !
        return {
            "loss_obj": loss_obj,
            "loss_noobj": loss_noobj,
            "loss_box": loss_box,
            "loss_class": loss_class
        }


class YOLOResNetNetwork(nn.Module):
    """ Réseau principal compatible avec l'interface de ton Trainer """
    def __init__(self, num_classes: int, grid_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.head = YOLOHead(in_channels=512, num_classes=num_classes, grid_size=grid_size)
        
        # Le réseau possède sa propre fonction de perte
        self.criterion = SimplifiedYoloLoss(num_classes=num_classes, grid_size=grid_size)
        
    def forward(self, images, targets=None):
        # 1. Empilage des images en tenseur si c'est une liste (comportement standard du DataLoader)
        if isinstance(images, list):
            images = torch.stack(images)
            
        features = self.backbone(images)
        features = self.dropout(features)
        predictions = self.head(features)
        
        # 2. Si le modèle est en mode entraînement ET qu'on a des cibles : on renvoie le dictionnaire de pertes
        if self.training and targets is not None:
            return self.criterion(predictions, targets)
            
        # 3. Sinon (en évaluation / inférence), on renvoie les prédictions brutes
        return predictions


class YOLOModel(Model):
    """ Modèle YOLO s'intégrant parfaitement dans ton framework """
    def __init__(self, config: Config, grid_size: int = 10, dropout: float = 0.1, **kwargs):
        self.grid_size = grid_size
        self.dropout = dropout
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return f"YOLO_ResNet34_Grid{self.grid_size}"

    def build_model(self) -> nn.Module:
        print(f"[INFO] Construction du Réseau YOLO ResNet34 (Grille : {self.grid_size}x{self.grid_size})...")
        model = YOLOResNetNetwork(
            num_classes=self.num_classes,
            grid_size=self.grid_size,
            dropout=self.dropout
        )
        self._initialize_custom_weights(model)
        return model
    
    def _initialize_custom_weights(self, model: nn.Module):
        for m in model.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_specific_params(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "grid_size": self.grid_size,
            "dropout": self.dropout,
            "input_size": (300, 300),
            "architecture": "YOLO_ResNet34"
        }

    def predict(self, images, confidence_threshold=0.15, iou_threshold=0.45, img_size=(300, 300)):
        """ Décodage des prédictions YOLO pour le Visualizer """
        self.model.eval()
        predictions_list = []
        
        batch_tensor = torch.stack([img.to(self.device) for img in images]) if isinstance(images, list) else images.to(self.device)
        
        with torch.no_grad():
            preds = self.model(batch_tensor)
            
        B, _, S, S = preds.shape
        
        for b in range(B):
            pred_b = preds[b]
            obj_scores = torch.sigmoid(pred_b[0])
            boxes = pred_b[1:5]
            class_probs = torch.softmax(pred_b[5:], dim=0)
            
            keep_mask = obj_scores > confidence_threshold
            keep_indices = torch.nonzero(keep_mask)
            
            b_boxes, b_scores, b_labels = [], [], []
            
            for idx in keep_indices:
                j, i = idx[0].item(), idx[1].item()
                score = obj_scores[j, i].item()
                xc, yc, w, h = boxes[:, j, i].cpu().numpy()
                class_id = torch.argmax(class_probs[:, j, i]).item()
                
                xmin = (xc - w / 2.0) * img_size[0]
                ymin = (yc - h / 2.0) * img_size[1]
                xmax = (xc + w / 2.0) * img_size[0]
                ymax = (yc + h / 2.0) * img_size[1]
                
                b_boxes.append([xmin, ymin, xmax, ymax])
                b_scores.append(score)
                b_labels.append(class_id)
                
            if len(b_boxes) > 0:
                t_boxes = torch.tensor(b_boxes, dtype=torch.float32, device=self.device)
                t_scores = torch.tensor(b_scores, dtype=torch.float32, device=self.device)
                t_labels = torch.tensor(b_labels, dtype=torch.long, device=self.device)
                
                keep_nms = nms(t_boxes, t_scores, iou_threshold)
                predictions_list.append({
                    "boxes": t_boxes[keep_nms].cpu(),
                    "scores": t_scores[keep_nms].cpu(),
                    "labels": t_labels[keep_nms].cpu()
                })
            else:
                predictions_list.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros(0, dtype=torch.float32),
                    "labels": torch.zeros(0, dtype=torch.long)
                })
                
        return predictions_list