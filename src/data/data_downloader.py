import os
import requests
import zipfile
from pathlib import Path
import shutil
import logging
from tqdm import tqdm
import gdown

class DataDownloader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialise le téléchargeur de données.
        
        Args:
            data_dir: Répertoire où seront stockées les données
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # URLs des datasets
        self.datasets = {
            "DAGM": "https://drive.google.com/uc?id=1-1wAxFbJgN8Qz1K7J8Qz1K7J8Qz1K7J8",
            "KolektorSDD": "https://www.vicos.si/Downloads/KolektorSDD"
        }
        
    def download_dagm_dataset(self):
        """
        Télécharge et prépare le dataset DAGM.
        """
        try:
            self.logger.info("Téléchargement du dataset DAGM...")
            
            # Créer les répertoires
            dagm_dir = self.data_dir / "DAGM"
            dagm_dir.mkdir(exist_ok=True)
            
            # Télécharger le fichier
            url = self.datasets["DAGM"]
            output = dagm_dir / "dagm_dataset.zip"
            gdown.download(url, str(output), quiet=False)
            
            # Extraire le fichier
            self.logger.info("Extraction des fichiers...")
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(dagm_dir)
            
            # Organiser les données
            self._organize_dagm_data(dagm_dir)
            
            # Nettoyer
            output.unlink()
            
            self.logger.info("Dataset DAGM téléchargé et préparé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement du dataset DAGM: {e}")
            raise
            
    def _organize_dagm_data(self, dagm_dir: Path):
        """
        Organise les données du dataset DAGM.
        
        Args:
            dagm_dir: Répertoire du dataset DAGM
        """
        # Créer les répertoires pour l'entraînement et le test
        train_dir = dagm_dir / "train"
        test_dir = dagm_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Organiser les images par classe
        for class_name in ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6"]:
            class_dir = dagm_dir / class_name
            if class_dir.exists():
                # Créer les répertoires de classe
                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                train_class_dir.mkdir(exist_ok=True)
                test_class_dir.mkdir(exist_ok=True)
                
                # Déplacer les images
                images = list(class_dir.glob("*.png"))
                for i, img in enumerate(images):
                    if i < len(images) * 0.8:  # 80% pour l'entraînement
                        shutil.copy2(img, train_class_dir / img.name)
                    else:  # 20% pour le test
                        shutil.copy2(img, test_class_dir / img.name)
                        
    def prepare_for_colab(self):
        """
        Prépare les données pour Google Colab.
        """
        try:
            self.logger.info("Préparation des données pour Google Colab...")
            
            # Créer un fichier README
            readme_path = self.data_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write("""# Dataset pour l'analyse des défauts d'outils de forage

## Structure
```
data/
├── DAGM/
│   ├── train/
│   │   ├── Class1/
│   │   ├── Class2/
│   │   ├── Class3/
│   │   ├── Class4/
│   │   ├── Class5/
│   │   └── Class6/
│   └── test/
│       ├── Class1/
│       ├── Class2/
│       ├── Class3/
│       ├── Class4/
│       ├── Class5/
│       └── Class6/
```

## Utilisation
1. Téléchargez le dossier `data` sur votre Google Drive
2. Montez votre Google Drive dans Colab
3. Utilisez le chemin `/content/drive/MyDrive/data` pour accéder aux données
""")
            
            self.logger.info("Données préparées pour Google Colab")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des données pour Colab: {e}")
            raise

if __name__ == "__main__":
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    
    # Télécharger et préparer les données
    downloader = DataDownloader()
    downloader.download_dagm_dataset()
    downloader.prepare_for_colab() 