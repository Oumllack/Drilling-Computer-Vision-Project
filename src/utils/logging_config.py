import logging
import sys
from pathlib import Path

def setup_logging(log_file: str = None, log_level: int = logging.INFO):
    """
    Configure le système de logging.
    
    Args:
        log_file: Chemin vers le fichier de log (optionnel)
        log_level: Niveau de logging
        
    Returns:
        Logger configuré
    """
    # Créer le logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Formatter pour les logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier si spécifié
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 