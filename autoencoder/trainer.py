from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import numpy as np
from torchvision import transforms
import logging
from datasets import load_dataset
from PIL import Image
import io
from torch.utils.data import Dataset

from autoencoder import ResNet50Autoencoder


class ImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if isinstance(item['image'], dict):
            image = item['image']['bytes']
            image = Image.open(io.BytesIO(image))
        elif isinstance(item['image'], bytes):
            image = Image.open(io.BytesIO(item['image']))
        else:
            image = item['image']
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)

        return image, 0


class AutoencoderTrainer:
    def __init__(self, model_class, dataset, config):
        self.model_class = model_class
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_dataloaders(self, batch_size):
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader

def log_images(self, experiment, original_images, reconstructed_images, epoch, prefix="train"):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def denormalize(images):
        return images * std + mean
    
    num_images = min(5, original_images.size(0))
    indices = torch.randperm(original_images.size(0))[:num_images]
    
    for idx in range(num_images):
        i = indices[idx]
        original = denormalize(original_images[i]).cpu().numpy()
        reconstructed = denormalize(reconstructed_images[i]).cpu().numpy()
        
        original = np.transpose(original, (1, 2, 0)).clip(0, 1)
        reconstructed = np.transpose(reconstructed, (1, 2, 0)).clip(0, 1)
        
        height, width = original.shape[:2]
        
        combined_image = np.zeros((height, width * 2, 3))
        combined_image[:, :width] = original
        combined_image[:, width:] = reconstructed
        
        experiment.log_image(
            combined_image * 255,
            name=f"{prefix}_comparison_{idx}",
            step=epoch
        )

    def train_epoch(self, model, train_loader, criterion, optimizer, experiment):
        model.train()
        total_loss = 0
    
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                output = model(data)
                loss = criterion(output, data)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def objective(self, trial):
        experiment = Experiment(
            api_key=self.config["comet_api_key"],
            project_name=self.config["project_name"]
        )

        experiment.log_code(folder="autoencoder")

        hyperparameters = {
            "latent_dim": trial.suggest_categorical("latent_dim", [128, 192, 256, 384, 512]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "freeze_percentage": trial.suggest_float("freeze_percentage", 0.5, 0.9),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "loss_function": trial.suggest_categorical("loss_function", ["mse", "l1"])
        }
        experiment.log_parameters(hyperparameters)

        model = self.model_class(
            latent_dim=hyperparameters["latent_dim"],
            freeze_percentage=hyperparameters["freeze_percentage"]
        ).to(self.device)

        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = hyperparameters["dropout_rate"]

        train_loader, val_loader = self.create_dataloaders(hyperparameters["batch_size"])
        
        criterion = nn.MSELoss() if hyperparameters["loss_function"] == "mse" else nn.L1Loss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"]
        )
        
        best_val_loss = float('inf')
        patience = self.config["patience"]
        patience_counter = 0
        
        for epoch in range(self.config["max_epochs"]):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, experiment)
            val_loss = self.validate(model, val_loader, criterion)

            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)

            with experiment.test() as test, torch.no_grad() as nograd:
                for data, _ in val_loader:
                    data = data.to(self.device)
                    out_data = model(data)
                    self.log_images(experiment, data, out_data, epoch)
                    break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(model.state_dict(), f"{self.config['model_save_path']}/best_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        log_model(experiment, model, model_name="AutoEncoder")
        experiment.log_metric("best_val_loss", best_val_loss)
        experiment.end()

        return best_val_loss

    def optimize(self):
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=self.config["n_trials"],
            timeout=self.config["timeout"],
        )
        
        return study.best_trial

config = {
    "comet_api_key": "oCeR8CKfuIxFIvfKv4N0B9mpV",
    "project_name": "autoencoder",
    "model_save_path": "./models",
    "max_epochs": 100,
    "patience": 10,
    "n_trials": 50,
    "timeout": 72000
}

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    hf_dataset = load_dataset("Artificio/WikiArt_Full", split="train")

    dataset = ImageDataset(hf_dataset, transform)
    
    trainer = AutoencoderTrainer(ResNet50Autoencoder, dataset, config)
    best_trial = trainer.optimize()
    print("Best trial:", best_trial)
