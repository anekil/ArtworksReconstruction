import comet_ml
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

from vqvae import VQVAE


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

        print(self.device)

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
        num_images = min(10, original_images.size(0))
        indices = torch.randperm(original_images.size(0))[:num_images]
        
        for idx in range(num_images):
            i = indices[idx]
            original = original_images[i].cpu().numpy()
            reconstructed = reconstructed_images[i].cpu().numpy()
            
            # Transpose from CHW to HWC format
            original = np.transpose(original, (1, 2, 0)).clip(0, 1)
            reconstructed = np.transpose(reconstructed, (1, 2, 0)).clip(0, 1)
            
            height, width = original.shape[:2]
            combined_image = np.zeros((height, width * 2, 3))
            combined_image[:, :width] = original
            combined_image[:, width:] = reconstructed
            
            experiment.log_image(
                combined_image * 255,  # Scale back to [0,255] for visualization
                name=f"{prefix}_comparison_{idx}",
                step=epoch
            )

    def train_epoch(self, model, train_loader, criterion, optimizer, latent_loss_weight = 1.):
        model.train()
        total_loss = 0
        total_q_loss = 0

        # latent_loss_weight = 0.25
    
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(self.device)

            output, latent_loss = model(data)
            recon_loss = criterion(output, data)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss * latent_loss_weight

            loss.backward()
            optimizer.step()

            total_loss += recon_loss.item()
            total_q_loss += latent_loss.item()
            
        return total_loss / len(train_loader), total_q_loss / len(train_loader)

    def validate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        total_q_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                output, latent_loss = model(data)
                recon_loss = criterion(output, data)
                latent_loss = latent_loss.mean()
                # loss = recon_loss + latent_loss

                total_loss += recon_loss.item()
                total_q_loss += latent_loss.item()

        return total_loss / len(val_loader), total_q_loss / len(val_loader)

    def objective(self, trial):
        hyperparameters = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "h_dim": trial.suggest_categorical("h_dim", [64, 128, 196]),
            "beta": trial.suggest_float("beta", 0.01, 1.0),
            "n_embeddings": trial.suggest_categorical("n_embeddings", [64, 128, 256, 512, 1024]),
        }
        
        return self.run(hyperparameters)

    def run(self, hyperparameters):
        experiment = comet_ml.Experiment(
            project_name=self.config["project_name"]
        )

        experiment.log_code(folder="autoencoder")
        experiment.log_parameters(hyperparameters)

        # h_dim = hyperparameters["h_dim"]
        # beta = hyperparameters["beta"]
        # n_embeddings = hyperparameters["n_embeddings"]

        model = self.model_class().to(self.device)
        train_loader, val_loader = self.create_dataloaders(hyperparameters["batch_size"])
        criterion = nn.MSELoss()
        

        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"]
        )
        
        best_val_loss = float('inf')
        patience = self.config["patience"]
        patience_counter = 0

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        try:
            for epoch in range(self.config["max_epochs"]):
                train_loss, train_q_loss = self.train_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_q_loss = self.validate(model, val_loader, criterion)

                scheduler.step(val_loss)

                experiment.log_metric("train_loss", train_loss, step=epoch)
                experiment.log_metric("val_loss", val_loss, step=epoch)
                experiment.log_metric("train_q_loss", train_q_loss, step=epoch)
                experiment.log_metric("val_q_loss", val_q_loss, step=epoch)

                with experiment.test() as test, torch.no_grad() as nograd:
                    for data, _ in val_loader:
                        data = data.to(self.device)
                        out_data, _ = model(data)
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

        except KeyboardInterrupt:
            print("Experiment interrupted")
            pass

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
            n_trials=self.config["n_trials"]
        )
        
        return study.best_trial

config = {
    "project_name": "autoencoder",
    "model_save_path": "./autoencoder/models/",
    "max_epochs": 300,
    "patience": 20,
    "n_trials": 20,
    "timeout": 72000
}

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    hyperparameters = {
        "learning_rate": 3e-4,
        "batch_size": 32,
        "weight_decay": 0,
        "beta": 0.33,
        "h_dim": 128,
        "n_embeddings": 512,
    }

    hf_dataset = load_dataset("Artificio/WikiArt_Full", split="train")

    dataset = ImageDataset(hf_dataset, transform)
    
    trainer = AutoencoderTrainer(VQVAE, dataset, config)
    # best_trial = trainer.optimize()
    best_trial = trainer.run(hyperparameters)
    # trials = []
    # for hyperparameters in hyperparameter_set:
    #     trainer.run(hyperparameters)
    # best_trial = max(trials)
    print("Trial loss:", best_trial)