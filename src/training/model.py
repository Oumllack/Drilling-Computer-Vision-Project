import torch
import torch.nn as nn
import torchvision.models as models
import ssl

# Désactiver temporairement la vérification SSL
ssl._create_default_https_context = ssl._create_unverified_context

class DefectModel(nn.Module):
    def __init__(self, num_classes=5):  # 5 classes: MT_Free, MT_Fray, MT_Crack, MT_Break, MT_Blowhole
        super(DefectModel, self).__init__()
        
        # Charger ResNet50 pré-entraîné
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remplacer la dernière couche fully connected
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # Ajouter du dropout pour réduire l'overfitting
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x) 