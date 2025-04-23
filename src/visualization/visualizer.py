import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from src.detection.defect_detector import Defect
from src.classification.state_classifier import ClassificationResult

class Visualizer:
    def __init__(self):
        """Initialise le visualiseur"""
        self.colors = {
            "normal": (0, 255, 0),        # Vert
            "usure_moderee": (255, 255, 0), # Jaune
            "usure_severe": (255, 165, 0),  # Orange
            "defaut_critique": (255, 0, 0)  # Rouge
        }
        
    def visualize_results(self, 
                         image: np.ndarray,
                         defects: List[Defect],
                         classification: ClassificationResult) -> np.ndarray:
        """
        Visualise les résultats de l'analyse sur l'image.
        
        Args:
            image: Image originale
            defects: Liste des défauts détectés
            classification: Résultat de la classification
            
        Returns:
            Image avec les annotations
        """
        # Copier l'image pour ne pas modifier l'originale
        vis_image = image.copy()
        
        # Dessiner les défauts
        vis_image = self._draw_defects(vis_image, defects)
        
        # Ajouter les informations de classification
        vis_image = self._add_classification_info(vis_image, classification)
        
        return vis_image
    
    def _draw_defects(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """
        Dessine les défauts sur l'image.
        
        Args:
            image: Image à annoter
            defects: Liste des défauts
            
        Returns:
            Image annotée
        """
        for defect in defects:
            x, y, w, h = defect.bbox
            color = self.colors.get(defect.severity, (255, 255, 255))
            
            # Dessiner le rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Ajouter le texte
            text = f"{defect.defect_type} ({defect.confidence:.2f})"
            cv2.putText(image, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return image
    
    def _add_classification_info(self, 
                               image: np.ndarray,
                               classification: ClassificationResult) -> np.ndarray:
        """
        Ajoute les informations de classification sur l'image.
        
        Args:
            image: Image à annoter
            classification: Résultat de la classification
            
        Returns:
            Image annotée
        """
        # Créer un fond semi-transparent pour le texte
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Ajouter les informations
        color = self.colors.get(classification.state, (255, 255, 255))
        cv2.putText(image, f"État: {classification.state}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Confiance: {classification.confidence:.2f}",
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
    
    def plot_features(self, features: Dict[str, float]):
        """
        Trace un graphique des caractéristiques extraites.
        
        Args:
            features: Dictionnaire des caractéristiques
        """
        plt.figure(figsize=(10, 6))
        plt.bar(features.keys(), features.values())
        plt.title("Caractéristiques extraites")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def save_visualization(self, 
                          image: np.ndarray,
                          output_path: str):
        """
        Sauvegarde l'image visualisée.
        
        Args:
            image: Image à sauvegarder
            output_path: Chemin de sauvegarde
        """
        cv2.imwrite(output_path, image) 