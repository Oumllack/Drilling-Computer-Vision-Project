import argparse
import torch
from torchvision import transforms
from PIL import Image
import logging
from training.model import DefectModel

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def predict_image(model_path: str, image_path: str):
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le modèle
    model = DefectModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Préparer les transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Charger et transformer l'image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Faire la prédiction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
    
    # Classes
    classes = ["MT_Free", "MT_Fray", "MT_Crack", "MT_Break", "MT_Blowhole"]
    
    # Afficher les résultats
    logger.info(f"Image: {image_path}")
    logger.info(f"Classe prédite: {classes[predicted_class]}")
    logger.info("\nProbabilités par classe:")
    for i, prob in enumerate(probabilities[0]):
        logger.info(f"{classes[i]}: {prob.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faire une prédiction avec le modèle entraîné")
    parser.add_argument("--model-path", type=str, required=True, help="Chemin du modèle")
    parser.add_argument("--image-path", type=str, required=True, help="Chemin de l'image à prédire")
    args = parser.parse_args()
    
    predict_image(args.model_path, args.image_path) 