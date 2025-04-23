import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from training.model import DefectModel
from training.dataset import DefectDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def test_model(model_path: str, data_dir: str):
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
    
    # Charger les données de test
    test_dataset = DefectDataset(data_dir + "/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Évaluer le modèle
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculer la précision par classe
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Afficher les résultats
    logger.info(f"Précision globale: {100. * correct / total:.2f}%")
    logger.info("\nPrécision par classe:")
    classes = ["MT_Free", "MT_Fray", "MT_Crack", "MT_Break", "MT_Blowhole"]
    for i in range(5):
        if class_total[i] > 0:
            logger.info(f"{classes[i]}: {100. * class_correct[i] / class_total[i]:.2f}%")

if __name__ == "__main__":
    test_model("models/best_model.pth", "data") 