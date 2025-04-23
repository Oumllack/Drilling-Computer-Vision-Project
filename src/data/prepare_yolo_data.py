import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import cv2
import numpy as np
import json

class YOLODataPreparator:
    def __init__(self, data_dir: str = "data", output_dir: str = "yolo_data"):
        """
        Initialise le préparateur de données YOLO.
        
        Args:
            data_dir: Répertoire contenant les données brutes
            output_dir: Répertoire de sortie pour les données YOLO
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Créer les répertoires de sortie
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        # Définir les classes
        self.classes = {
            "MT_Free": 0,
            "MT_Fray": 1,
            "MT_Crack": 2,
            "MT_Break": 3,
            "MT_Blowhole": 4,
            "Crack": 5,
            "Bridge_Crack": 6
        }
        
    def prepare_magnetic_tile(self):
        """
        Prépare les données Magnetic-Tile-Defect au format YOLO.
        """
        try:
            self.logger.info("Préparation des données Magnetic-Tile-Defect...")
            source_dir = self.data_dir / "train"
            
            for defect_type in ["MT_Free", "MT_Fray", "MT_Crack", "MT_Break", "MT_Blowhole"]:
                defect_dir = source_dir / defect_type
                if defect_dir.exists():
                    images = list(defect_dir.glob("*.jpg"))
                    for img in tqdm(images, desc=f"Traitement {defect_type}"):
                        # Copier l'image
                        shutil.copy2(img, self.train_dir / img.name)
                        
                        # Créer l'annotation YOLO
                        img_cv = cv2.imread(str(img))
                        height, width = img_cv.shape[:2]
                        
                        # Pour les images sans défaut, créer une annotation vide
                        if defect_type == "MT_Free":
                            continue
                            
                        # Pour les images avec défaut, créer une annotation
                        # Ici, nous supposons que le défaut occupe toute l'image
                        # Dans un cas réel, vous devriez avoir les coordonnées exactes
                        x_center = width / 2
                        y_center = height / 2
                        w = width
                        h = height
                        
                        # Normaliser les coordonnées
                        x_center /= width
                        y_center /= height
                        w /= width
                        h /= height
                        
                        # Créer le fichier d'annotation
                        ann_path = self.train_dir / f"{img.stem}.txt"
                        with open(ann_path, "w") as f:
                            f.write(f"{self.classes[defect_type]} {x_center} {y_center} {w} {h}\n")
                            
            self.logger.info("Données Magnetic-Tile-Defect préparées avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des données Magnetic-Tile-Defect: {e}")
            raise
            
    def prepare_crackforest(self):
        """
        Prépare les données CrackForest au format YOLO.
        """
        try:
            self.logger.info("Préparation des données CrackForest...")
            source_dir = self.data_dir / "train" / "Crack"
            
            if source_dir.exists():
                images = list(source_dir.glob("*.jpg"))
                for img in tqdm(images, desc="Traitement CrackForest"):
                    # Copier l'image
                    shutil.copy2(img, self.train_dir / img.name)
                    
                    # Lire le masque
                    mask_path = source_dir / f"{img.stem}.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        height, width = mask.shape
                        
                        # Trouver les contours du défaut
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Créer le fichier d'annotation
                        ann_path = self.train_dir / f"{img.stem}.txt"
                        with open(ann_path, "w") as f:
                            for contour in contours:
                                x, y, w, h = cv2.boundingRect(contour)
                                
                                # Normaliser les coordonnées
                                x_center = (x + w/2) / width
                                y_center = (y + h/2) / height
                                w = w / width
                                h = h / height
                                
                                f.write(f"{self.classes['Crack']} {x_center} {y_center} {w} {h}\n")
                                
            self.logger.info("Données CrackForest préparées avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des données CrackForest: {e}")
            raise
            
    def prepare_bridge_crack(self):
        """
        Prépare les données Bridge_Crack_Image au format YOLO.
        """
        try:
            self.logger.info("Préparation des données Bridge_Crack_Image...")
            source_dir = self.data_dir / "train" / "Bridge_Crack"
            
            if source_dir.exists():
                images = list(source_dir.glob("*.jpg"))
                for img in tqdm(images, desc="Traitement Bridge_Crack"):
                    # Copier l'image
                    shutil.copy2(img, self.train_dir / img.name)
                    
                    # Pour les images de pont, nous supposons que le défaut occupe toute l'image
                    img_cv = cv2.imread(str(img))
                    height, width = img_cv.shape[:2]
                    
                    # Normaliser les coordonnées
                    x_center = 0.5
                    y_center = 0.5
                    w = 1.0
                    h = 1.0
                    
                    # Créer le fichier d'annotation
                    ann_path = self.train_dir / f"{img.stem}.txt"
                    with open(ann_path, "w") as f:
                        f.write(f"{self.classes['Bridge_Crack']} {x_center} {y_center} {w} {h}\n")
                        
            self.logger.info("Données Bridge_Crack_Image préparées avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des données Bridge_Crack_Image: {e}")
            raise
            
    def prepare_all_data(self):
        """
        Prépare toutes les données au format YOLO.
        """
        self.prepare_magnetic_tile()
        self.prepare_crackforest()
        self.prepare_bridge_crack()
        
        # Créer le fichier de configuration YOLO
        config = {
            "path": str(self.output_dir),
            "train": "train",
            "val": "val",
            "names": {str(v): k for k, v in self.classes.items()}
        }
        
        with open(self.output_dir / "data.yaml", "w") as f:
            json.dump(config, f, indent=4)
            
        self.logger.info("Toutes les données ont été préparées avec succès")
        
    def create_zip(self):
        """
        Crée un fichier ZIP des données préparées.
        """
        try:
            self.logger.info("Création du fichier ZIP...")
            shutil.make_archive("yolo_data", "zip", self.output_dir)
            self.logger.info("Fichier ZIP créé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du fichier ZIP: {e}")
            raise

if __name__ == "__main__":
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    
    # Préparer les données
    preparator = YOLODataPreparator()
    preparator.prepare_all_data()
    preparator.create_zip() 