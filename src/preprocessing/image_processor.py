import cv2
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialise le processeur d'images.
        
        Args:
            target_size: Taille cible pour le redimensionnement des images
        """
        self.target_size = target_size

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraite une image pour l'analyse.
        
        Args:
            image: Image d'entrée au format numpy array
            
        Returns:
            Image prétraitée
        """
        # Conversion en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Redimensionnement
        image = cv2.resize(image, self.target_size)
        
        # Normalisation
        image = image.astype(np.float32) / 255.0
        
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste de l'image.
        
        Args:
            image: Image d'entrée
            
        Returns:
            Image avec contraste amélioré
        """
        # Application de l'égalisation d'histogramme
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        return enhanced

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Détecte les bords dans l'image.
        
        Args:
            image: Image d'entrée
            
        Returns:
            Image avec les bords détectés
        """
        edges = cv2.Canny(image, 100, 200)
        return edges 