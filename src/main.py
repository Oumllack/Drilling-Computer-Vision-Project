import cv2
import argparse
from pathlib import Path
import logging
from src.detection.defect_detector import DefectDetector
from src.classification.state_classifier import StateClassifier
from src.visualization.visualizer import Visualizer
from src.models.model_manager import ModelManager
from src.utils.logging_config import setup_logging

def main():
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Analyse d'outils de forage par vision par ordinateur")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image à analyser")
    parser.add_argument("--output", type=str, default="output.jpg", help="Chemin de sauvegarde du résultat")
    parser.add_argument("--model-detection", type=str, help="Chemin vers le modèle de détection")
    parser.add_argument("--model-classification", type=str, help="Chemin vers le modèle de classification")
    parser.add_argument("--log-file", type=str, help="Chemin vers le fichier de log")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Niveau de logging")
    args = parser.parse_args()
    
    # Configurer le logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(args.log_file, log_level)
    
    try:
        # Vérifier que l'image existe
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"L'image {args.image} n'existe pas")
        
        # Charger l'image
        logger.info(f"Chargement de l'image {args.image}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image {args.image}")
        
        # Initialiser les composants
        logger.info("Initialisation des composants")
        model_manager = ModelManager()
        detector = DefectDetector(model_path=args.model_detection)
        classifier = StateClassifier(model_path=args.model_classification)
        visualizer = Visualizer()
        
        # Exécuter le pipeline
        logger.info("Début de l'analyse")
        defects = detector.detect_defects(image)
        logger.info(f"{len(defects)} défauts détectés")
        
        classification = classifier.classify_state(image)
        logger.info(f"État détecté: {classification.state} (confiance: {classification.confidence:.2f})")
        
        # Visualiser les résultats
        logger.info("Visualisation des résultats")
        result = visualizer.visualize_results(image, defects, classification)
        
        # Sauvegarder le résultat
        output_path = Path(args.output)
        visualizer.save_visualization(result, str(output_path))
        logger.info(f"Résultat sauvegardé dans {output_path}")
        
        # Afficher les caractéristiques
        visualizer.plot_features(classification.features)
        
        logger.info("Analyse terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 