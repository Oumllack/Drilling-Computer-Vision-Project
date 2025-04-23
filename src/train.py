import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import logging
from tqdm import tqdm

from training.dataset import DefectDataset
from training.model import DefectModel

def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle de détection de défauts")
    parser.add_argument("--data-dir", type=str, required=True, help="Répertoire des données")
    parser.add_argument("--model-dir", type=str, required=True, help="Répertoire pour sauvegarder les modèles")
    parser.add_argument("--num-epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--save-interval", type=int, default=5, help="Intervalle de sauvegarde du modèle")
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Créer le répertoire de sauvegarde des modèles
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Définir les transformations pour l'entraînement et la validation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Créer les datasets et dataloaders
    train_dataset = DefectDataset(args.data_dir + "/train", transform=train_transform)
    val_dataset = DefectDataset(args.data_dir + "/val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialiser le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectModel().to(device)
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Entraînement
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"Époque {epoch + 1}/{args.num_epochs}")
        
        # Mode entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Entraînement"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Mode validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / "best_model.pth")
            logger.info("Meilleur modèle sauvegardé!")
        
        # Sauvegarder le modèle périodiquement
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_dir / f"checkpoint_epoch_{epoch + 1}.pth")
            logger.info(f"Checkpoint sauvegardé pour l'époque {epoch + 1}")

if __name__ == "__main__":
    main() 