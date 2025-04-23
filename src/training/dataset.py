import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
import numpy as np

class DefectDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Initialise le dataset de défauts.
        
        Args:
            data_dir: Répertoire des données
            transform: Transformations à appliquer aux images
        """
        self.data_dir = Path(data_dir)
        
        # Transformations par défaut si aucune n'est spécifiée
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Définir les classes
        self.classes = {
            "MT_Free": 0,
            "MT_Fray": 1,
            "MT_Crack": 2,
            "MT_Break": 3,
            "MT_Blowhole": 4
        }
        
        # Charger les chemins des images et leurs étiquettes
        self.images = []
        self.labels = []
        
        # Parcourir les sous-répertoires
        for class_name, class_idx in self.classes.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Charger l'image
        image = Image.open(img_path).convert("RGB")
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
def get_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """
    Crée les DataLoaders pour l'entraînement et la validation.
    
    Args:
        data_dir: Répertoire des données
        batch_size: Taille des lots
        num_workers: Nombre de workers pour le chargement des données
        
    Returns:
        Tuple (train_loader, val_loader)
    """
    # Créer les datasets
    train_dataset = DefectDataset(data_dir / "train")
    val_dataset = DefectDataset(data_dir / "test")
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 