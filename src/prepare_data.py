import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

def setup_logging():
    """Configure le logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def prepare_data(archive_dir: str = "archive", data_dir: str = "data", train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Prépare les données pour l'entraînement.
    
    Args:
        archive_dir: Répertoire source des données
        data_dir: Répertoire de destination
        train_ratio: Ratio des données pour l'entraînement
        val_ratio: Ratio des données pour la validation
    """
    logger = setup_logging()
    
    # Créer les répertoires de destination
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les données Magnetic-Tile-Defect
    logger.info("Préparation des données Magnetic-Tile-Defect...")
    mt_dir = Path(archive_dir) / "Magnetic-Tile-Defect"
    defect_types = ["MT_Free", "MT_Fray", "MT_Crack", "MT_Break", "MT_Blowhole"]
    
    for defect_type in defect_types:
        logger.info(f"Traitement de {defect_type}...")
        src_dir = mt_dir / defect_type / "Imgs"
        
        if not src_dir.exists():
            logger.warning(f"Répertoire {src_dir} non trouvé")
            continue
            
        # Créer les répertoires de destination
        train_defect_dir = train_dir / defect_type
        val_defect_dir = val_dir / defect_type
        test_defect_dir = test_dir / defect_type
        train_defect_dir.mkdir(exist_ok=True)
        val_defect_dir.mkdir(exist_ok=True)
        test_defect_dir.mkdir(exist_ok=True)
        
        # Copier les images
        images = list(src_dir.glob("*.jpg"))
        if not images:  # Si pas d'images jpg, essayer png
            images = list(src_dir.glob("*.png"))
            
        if not images:
            logger.warning(f"Aucune image trouvée dans {src_dir}")
            continue
            
        train_size = int(len(images) * train_ratio)
        val_size = int(len(images) * val_ratio)
        
        for i, img in enumerate(tqdm(images)):
            if i < train_size:
                shutil.copy2(img, train_defect_dir / img.name)
            elif i < train_size + val_size:
                shutil.copy2(img, val_defect_dir / img.name)
            else:
                shutil.copy2(img, test_defect_dir / img.name)
                
    logger.info("Préparation des données terminée")

if __name__ == "__main__":
    prepare_data() 