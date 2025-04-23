import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional
import logging
from pathlib import Path

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        """
        Initialise le gestionnaire de modèles.
        
        Args:
            model_dir: Répertoire où sont stockés les modèles
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_detection_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Charge le modèle de détection.
        
        Args:
            model_path: Chemin vers le modèle pré-entraîné (optionnel)
            
        Returns:
            Modèle de détection
        """
        try:
            if model_path and Path(model_path).exists():
                self.logger.info(f"Chargement du modèle de détection depuis {model_path}")
                model = torch.load(model_path)
            else:
                self.logger.info("Chargement du modèle de détection par défaut")
                model = self._create_default_detection_model()
            return model
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle de détection: {e}")
            raise
            
    def load_classification_model(self, model_path: Optional[str] = None) -> nn.Module:
        """
        Charge le modèle de classification.
        
        Args:
            model_path: Chemin vers le modèle pré-entraîné (optionnel)
            
        Returns:
            Modèle de classification
        """
        try:
            if model_path and Path(model_path).exists():
                self.logger.info(f"Chargement du modèle de classification depuis {model_path}")
                model = torch.load(model_path)
            else:
                self.logger.info("Chargement du modèle de classification par défaut")
                model = self._create_default_classification_model()
            return model
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle de classification: {e}")
            raise
            
    def _create_default_detection_model(self) -> nn.Module:
        """
        Crée un modèle de détection par défaut.
        
        Returns:
            Modèle de détection
        """
        # Utiliser un modèle pré-entraîné de détection d'objets
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Modifier la dernière couche pour notre cas d'utilisation
        num_classes = 4  # Nombre de classes de défauts
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        return model
        
    def _create_default_classification_model(self) -> nn.Module:
        """
        Crée un modèle de classification par défaut.
        
        Returns:
            Modèle de classification
        """
        # Utiliser un modèle pré-entraîné de classification d'images
        model = models.resnet50(pretrained=True)
        # Modifier la dernière couche pour notre cas d'utilisation
        num_classes = 4  # Nombre d'états possibles
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    def save_model(self, model: nn.Module, name: str):
        """
        Sauvegarde un modèle.
        
        Args:
            model: Modèle à sauvegarder
            name: Nom du fichier de sauvegarde
        """
        try:
            save_path = self.model_dir / f"{name}.pth"
            torch.save(model, save_path)
            self.logger.info(f"Modèle sauvegardé dans {save_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            raise 