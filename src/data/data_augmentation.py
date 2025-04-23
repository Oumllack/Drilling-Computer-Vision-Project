import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import logging

class DataAugmentation:
    def __init__(self, data_dir: str = "data"):
        """
        Initialise l'augmentation des données.
        
        Args:
            data_dir: Répertoire des données
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Transformations de base
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transformations d'augmentation
        self.aug_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def augment_dataset(self, num_augmentations: int = 2):
        """
        Augmente le dataset d'entraînement.
        
        Args:
            num_augmentations: Nombre d'augmentations par image
        """
        try:
            train_dir = self.data_dir / "NEU" / "train"
            if not train_dir.exists():
                raise FileNotFoundError(f"Le répertoire {train_dir} n'existe pas")
                
            self.logger.info(f"Augmentation du dataset avec {num_augmentations} variations par image")
            
            # Parcourir toutes les classes
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    self.logger.info(f"Traitement de la classe {class_dir.name}")
                    
                    # Parcourir toutes les images
                    for img_path in class_dir.glob("*.jpg"):
                        # Charger l'image
                        image = cv2.imread(str(img_path))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Créer des versions augmentées
                        for i in range(num_augmentations):
                            # Appliquer l'augmentation
                            augmented = self.aug_transform(image=image)
                            aug_image = augmented["image"]
                            
                            # Sauvegarder l'image augmentée
                            aug_path = class_dir / f"{img_path.stem}_aug_{i}.jpg"
                            aug_image = aug_image.permute(1, 2, 0).numpy()
                            aug_image = (aug_image * 255).astype(np.uint8)
                            cv2.imwrite(str(aug_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                            
            self.logger.info("Augmentation du dataset terminée")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'augmentation du dataset: {e}")
            raise
            
    def get_transforms(self, is_training: bool = True):
        """
        Retourne les transformations appropriées.
        
        Args:
            is_training: Si True, retourne les transformations d'entraînement
            
        Returns:
            Transformations
        """
        if is_training:
            return self.aug_transform
        return self.base_transform

if __name__ == "__main__":
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    
    # Augmenter le dataset
    augmenter = DataAugmentation()
    augmenter.augment_dataset() 