import argparse
import torch
from training.model import DefectModel
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def save_model(model_path: str, output_path: str):
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le modèle
    model = DefectModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), output_path)
    logger.info(f"Modèle sauvegardé dans {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sauvegarder le modèle entraîné")
    parser.add_argument("--model-path", type=str, required=True, help="Chemin du modèle à sauvegarder")
    parser.add_argument("--output-path", type=str, required=True, help="Chemin de sauvegarde")
    args = parser.parse_args()
    
    save_model(args.model_path, args.output_path) 