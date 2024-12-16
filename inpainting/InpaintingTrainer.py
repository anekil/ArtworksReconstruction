import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import io
import random
import logging
from comet_ml import Experiment
import optuna
import time

class InpaintingTrainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("InpaintingTrainer")
        logging.basicConfig(level=logging.INFO)
        self.experiment = None

    def initialize_model(self):
        base_model = models.resnet18(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 226 * 226),
            nn.Sigmoid()
        )
        return base_model.to(self.device)
    def create_dataloaders(self, batch_size):
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
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

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def train(self):
        self.experiment = Experiment(
            api_key=self.config["comet_api_key"],
            project_name=self.config["project_name"]
        )

        self.experiment.log_parameters(self.config)

        model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        train_loader, val_loader, test_loader = self.create_dataloaders(self.config["batch_size"])

        best_val_loss = float('inf')
        patience_counter = 0

        total_epochs = self.config["max_epochs"]
        total_steps = len(train_loader) * total_epochs

        global_step = 0

        for epoch in range(total_epochs):
            model.train()
            total_train_loss = 0
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            for batch_idx, (damaged_images, original_images) in enumerate(train_loader):
                batch_start_time = time.time()

                damaged_images = damaged_images.to(self.device)
                original_images = original_images.to(self.device)

                optimizer.zero_grad()
                outputs = model(damaged_images)
                loss = criterion(outputs, original_images)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                global_step += 1

                batch_time = time.time() - batch_start_time
                current = batch_idx * self.config["batch_size"] + len(damaged_images)
                total = len(train_loader.dataset)
                percent = 100. * batch_idx / len(train_loader)
                print(f"Train Epoch: {epoch + 1} [{current}/{total} ({percent:.0f}%)]\tLoss: {loss.item():.6f}\tTime: {batch_time:.2f}s")

                self.experiment.log_metric("batch_train_loss", loss.item(), step=global_step)
                self.experiment.log_metric("batch_time", batch_time, step=global_step)

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for damaged_images, original_images in val_loader:
                    damaged_images = damaged_images.to(self.device)
                    original_images = original_images.to(self.device)
                    outputs = model(damaged_images)
                    loss = criterion(outputs, original_images)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

            self.experiment.log_metric("epoch_train_loss", avg_train_loss, epoch=epoch + 1)
            self.experiment.log_metric("epoch_val_loss", avg_val_loss, epoch=epoch + 1)

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s - Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")

            self.logger.info(f"Epoch {epoch + 1}: Avg Train Loss = {avg_train_loss:.6f}, Avg Val Loss = {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                #torch.save(model.state_dict(), f"{self.config['model_save_path']}/best_inpainting_model.pth")
                print(f"Validation loss decreased ({best_val_loss:.6f}). Saving model...")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epochs.")

            if patience_counter >= self.config["patience"]:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                print("Early stopping triggered.")
                break

            model.eval()
            with torch.no_grad():
                self.log_images(self.experiment, val_loader, epoch + 1)

            self.experiment.log_text(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s")

        self.test(model, test_loader, criterion)
        self.experiment.end()

    def test(self, model, test_loader, criterion):
        print("\nStarting testing phase...")
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for damaged_images, original_images in test_loader:
                damaged_images = damaged_images.to(self.device)
                original_images = original_images.to(self.device)
                outputs = model(damaged_images)
                loss = criterion(outputs, original_images)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.6f}")
        self.logger.info(f"Test Loss: {avg_test_loss:.6f}")
        self.experiment.log_metric("test_loss", avg_test_loss)
        self.log_images(self.experiment, test_loader, 'test')

    def log_images(self, experiment, data_loader, epoch):
        for damaged_images, original_images in data_loader:
            damaged_images = damaged_images.to(self.device)
            original_images = original_images.to(self.device)
            outputs = self.model(damaged_images)
            break

        num_images = min(5, damaged_images.size(0))
        indices = random.sample(range(damaged_images.size(0)), num_images)

        for idx in indices:
            di = damaged_images[idx].cpu()
            ri = outputs[idx].cpu()
            oi = original_images[idx].cpu()
            di_np = np.transpose(di.detach().numpy(), (1, 2, 0))
            ri_np = np.transpose(ri.detach().numpy(), (1, 2, 0))
            oi_np = np.transpose(oi.detach().numpy(), (1, 2, 0))
            di_np = (di_np - di_np.min()) / (di_np.max() - di_np.min())
            ri_np = (ri_np - ri_np.min()) / (ri_np.max() - ri_np.min())
            oi_np = (oi_np - oi_np.min()) / (oi_np.max() - oi_np.min())
            #Damaged | Reconstructed | Original
            combined_image = np.hstack((di_np, ri_np, oi_np))
            combined_image = (combined_image * 255).astype(np.uint8)
            experiment.log_image(
                combined_image,
                name=f"inpainting_epoch_{epoch}_idx_{idx}",
                step=epoch
            )
        print(f"Logged {num_images} images for Epoch {epoch}")
