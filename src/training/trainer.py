import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional
from src.training.dataset import get_data_loaders
from dataset import DefectDataset

class ModelTrainer:
    def __init__(self, model: nn.Module, data_dir: str = "data", model_dir: str = "models"):
        """
        Initialise l'entraîneur de modèle.
        
        Args:
            model: Modèle à entraîner
            data_dir: Répertoire des données
            model_dir: Répertoire de sauvegarde des modèles
        """
        self.model = model
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Configuration de l'entraînement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Transformations des images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Ajouter du dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, model.fc.out_features)
        )
        
    def train(self, 
              num_epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.0001,
              weight_decay: float = 1e-4,
              save_interval: int = 5) -> Dict[str, Any]:
        """
        Entraîne le modèle.
        
        Args:
            num_epochs: Nombre d'époques d'entraînement
            batch_size: Taille des lots
            learning_rate: Taux d'apprentissage
            weight_decay: Décroissance des poids
            save_interval: Intervalle de sauvegarde du modèle
            
        Returns:
            Historique d'entraînement
        """
        try:
            # Préparer les données
            train_dataset = DefectDataset(self.data_dir / "train", is_train=True)
            val_dataset = DefectDataset(self.data_dir / "test", is_train=False)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # Optimiseur et fonction de perte
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # Historique d'entraînement
            history = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": []
            }
            
            # Ajuster le taux d'apprentissage
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)
            
            best_val_acc = 0.0
            
            # Boucle d'entraînement
            for epoch in range(num_epochs):
                self.logger.info(f"Époque {epoch+1}/{num_epochs}")
                
                # Entraînement
                train_loss, train_acc = self._train_epoch(train_loader, self.optimizer, criterion)
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                
                # Validation
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Ajuster le taux d'apprentissage
                scheduler.step(val_acc)
                
                # Sauvegarde du modèle
                if (epoch + 1) % save_interval == 0:
                    self._save_model(epoch + 1)
                    
                # Sauvegarder le meilleur modèle
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_model(epoch + 1, best=True)
                    self.logger.info(f"Nouveau meilleur modèle sauvegardé avec {val_acc:.2f}% de précision")
                    
            return history
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            raise
            
    def _prepare_data(self, batch_size: int) -> tuple:
        """
        Prépare les données d'entraînement et de validation.
        
        Args:
            batch_size: Taille des lots
            
        Returns:
            Tuple (train_loader, val_loader)
        """
        return get_data_loaders(self.data_dir, batch_size=batch_size)
        
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> tuple:
        """
        Entraîne le modèle pour une époque.
        
        Args:
            train_loader: DataLoader d'entraînement
            optimizer: Optimiseur
            criterion: Fonction de perte
            
        Returns:
            Tuple (loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return total_loss / len(train_loader), correct / total
        
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> tuple:
        """
        Valide le modèle pour une époque.
        
        Args:
            val_loader: DataLoader de validation
            criterion: Fonction de perte
            
        Returns:
            Tuple (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return total_loss / len(val_loader), correct / total
        
    def _save_model(self, epoch: int, best: bool = False):
        """
        Sauvegarde le modèle.
        
        Args:
            epoch: Numéro de l'époque
            best: Indique si le modèle est le meilleur
        """
        save_path = self.model_dir / f"model_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, save_path)
        self.logger.info(f"Modèle sauvegardé dans {save_path}") 