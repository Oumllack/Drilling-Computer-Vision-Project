import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import cv2
import numpy as np

class DatasetPreparator:
    def __init__(self, archive_dir: str = "archive", output_dir: str = "data"):
        """
        Initialise le préparateur de datasets.
        
        Args:
            archive_dir: Répertoire contenant les datasets
            output_dir: Répertoire de sortie pour les données organisées
        """
        self.archive_dir = Path(archive_dir)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Créer les répertoires de sortie
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_magnetic_tile(self):
        """
        Prépare le dataset Magnetic-Tile-Defect.
        """
        try:
            self.logger.info("Préparation du dataset Magnetic-Tile-Defect...")
            source_dir = self.archive_dir / "Magnetic-Tile-Defect"
            
            # Créer les répertoires pour chaque type de défaut
            defect_types = ["MT_Free", "MT_Fray", "MT_Crack", "MT_Break", "MT_Blowhole"]
            for defect_type in defect_types:
                train_defect_dir = self.train_dir / defect_type
                test_defect_dir = self.test_dir / defect_type
                train_defect_dir.mkdir(exist_ok=True)
                test_defect_dir.mkdir(exist_ok=True)
                
                # Copier les images
                defect_dir = source_dir / defect_type
                if defect_dir.exists():
                    images = list(defect_dir.glob("*.jpg"))
                    for i, img in enumerate(images):
                        if i < len(images) * 0.8:  # 80% pour l'entraînement
                            shutil.copy2(img, train_defect_dir / img.name)
                        else:  # 20% pour le test
                            shutil.copy2(img, test_defect_dir / img.name)
                            
            self.logger.info("Dataset Magnetic-Tile-Defect préparé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation du dataset Magnetic-Tile-Defect: {e}")
            raise
            
    def prepare_crackforest(self):
        """
        Prépare le dataset CrackForest.
        """
        try:
            self.logger.info("Préparation du dataset CrackForest...")
            source_dir = self.archive_dir / "CrackForest"
            
            # Créer les répertoires
            train_crack_dir = self.train_dir / "Crack"
            test_crack_dir = self.test_dir / "Crack"
            train_crack_dir.mkdir(exist_ok=True)
            test_crack_dir.mkdir(exist_ok=True)
            
            # Copier les images et les masques
            image_dir = source_dir / "image"
            mask_dir = source_dir / "groundTruth"
            
            if image_dir.exists() and mask_dir.exists():
                images = list(image_dir.glob("*.jpg"))
                for i, img in enumerate(images):
                    mask = mask_dir / f"{img.stem}.png"
                    if mask.exists():
                        if i < len(images) * 0.8:  # 80% pour l'entraînement
                            shutil.copy2(img, train_crack_dir / img.name)
                            shutil.copy2(mask, train_crack_dir / mask.name)
                        else:  # 20% pour le test
                            shutil.copy2(img, test_crack_dir / img.name)
                            shutil.copy2(mask, test_crack_dir / mask.name)
                            
            self.logger.info("Dataset CrackForest préparé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation du dataset CrackForest: {e}")
            raise
            
    def prepare_bridge_crack(self):
        """
        Prépare le dataset Bridge_Crack_Image.
        """
        try:
            self.logger.info("Préparation du dataset Bridge_Crack_Image...")
            source_dir = self.archive_dir / "Bridge_Crack_Image" / "DBCC_Training_Data_Set"
            
            # Créer les répertoires
            train_bridge_dir = self.train_dir / "Bridge_Crack"
            test_bridge_dir = self.test_dir / "Bridge_Crack"
            train_bridge_dir.mkdir(exist_ok=True)
            test_bridge_dir.mkdir(exist_ok=True)
            
            # Copier les images
            if source_dir.exists():
                images = list(source_dir.glob("*.jpg"))
                for i, img in enumerate(images):
                    if i < len(images) * 0.8:  # 80% pour l'entraînement
                        shutil.copy2(img, train_bridge_dir / img.name)
                    else:  # 20% pour le test
                        shutil.copy2(img, test_bridge_dir / img.name)
                        
            self.logger.info("Dataset Bridge_Crack_Image préparé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation du dataset Bridge_Crack_Image: {e}")
            raise
            
    def prepare_all_datasets(self):
        """
        Prépare tous les datasets.
        """
        self.prepare_magnetic_tile()
        self.prepare_crackforest()
        self.prepare_bridge_crack()
        
        # Créer un fichier README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("""# Datasets pour l'analyse des défauts d'outils de forage

## Structure
```
data/
├── train/
│   ├── MT_Free/
│   ├── MT_Fray/
│   ├── MT_Crack/
│   ├── MT_Break/
│   ├── MT_Blowhole/
│   ├── Crack/
│   └── Bridge_Crack/
└── test/
    ├── MT_Free/
    ├── MT_Fray/
    ├── MT_Crack/
    ├── MT_Break/
    ├── MT_Blowhole/
    ├── Crack/
    └── Bridge_Crack/
```

## Sources
- Magnetic-Tile-Defect: Défauts de surface sur tuiles magnétiques
- CrackForest: Détection de fissures
- Bridge_Crack_Image: Fissures structurelles
""")
            
        self.logger.info("Tous les datasets ont été préparés avec succès")

if __name__ == "__main__":
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    
    # Préparer les datasets
    preparator = DatasetPreparator()
    preparator.prepare_all_datasets() 