import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import logging
from comet_ml import Experiment
import time
from torchvision import models

class SuperResolutionTrainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("SuperResolutionTrainer")
        logging.basicConfig(level=logging.INFO)
        self.experiment = None

    def initialize_model(self):

        class ResNet50Autoencoder(nn.Module):
            def __init__(self):
                super(ResNet50Autoencoder, self).__init__()

                # Load pre-trained ResNet50
                backbone = models.resnet50(pretrained=True)

                # Modify the encoder to retain only 1/6th of the channels
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool,
                    nn.Conv2d(64, 128 // 6, kernel_size=1),
                    nn.BatchNorm2d(128 // 6),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512 // 6, kernel_size=1),
                    nn.BatchNorm2d(512 // 6),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 1024 // 6, kernel_size=1),
                    nn.BatchNorm2d(1024 // 6),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, 2048 // 6, kernel_size=1),
                    nn.BatchNorm2d(2048 // 6),
                    nn.ReLU(inplace=True),
                )

                # Decoder (larger than encoder with a cone-like structure)
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(2048 // 6, 2048, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(64, 3, kernel_size=3, padding=1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        model = ResNet50Autoencoder()
        return model.to(self.device)

    def create_dataloaders(self, batch_size):
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader

    def train(self):
        self.experiment = Experiment(api_key=self.config["comet_api_key"], project_name=self.config["project_name"])
        self.experiment.log_parameters(self.config)
        if self.model is None:
            self.model = self.initialize_model()
        model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        train_loader, val_loader, test_loader = self.create_dataloaders(self.config["batch_size"])
        best_val_loss = float('inf')
        patience_counter = 0
        total_epochs = self.config["max_epochs"]
        global_step = 0
        for epoch in range(total_epochs):
            model.train()
            total_train_loss = 0.0
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            for batch_idx, (low_res_images, high_res_images) in enumerate(train_loader):
                batch_start_time = time.time()
                low_res_images = low_res_images.to(self.device)
                high_res_images = high_res_images.to(self.device)
                optimizer.zero_grad()
                outputs = model(low_res_images)
                loss = criterion(outputs, high_res_images)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                global_step += 1
                batch_time = time.time() - batch_start_time
                current = batch_idx * self.config["batch_size"] + len(low_res_images)
                total = len(train_loader.dataset)
                percent = 100.0 * batch_idx / len(train_loader)
                print(f"Train Epoch: {epoch + 1} [{current}/{total} ({percent:.0f}%)] Loss: {loss.item():.6f} Time: {batch_time:.2f}s")
                self.experiment.log_metric("batch_train_loss", loss.item(), step=global_step)
                self.experiment.log_metric("batch_time", batch_time, step=global_step)
            avg_train_loss = total_train_loss / len(train_loader)
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for low_res_images, high_res_images in val_loader:
                    low_res_images = low_res_images.to(self.device)
                    high_res_images = high_res_images.to(self.device)
                    outputs = model(low_res_images)
                    loss = criterion(outputs, high_res_images)
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
        total_test_loss = 0.0
        with torch.no_grad():
            for low_res_images, high_res_images in test_loader:
                low_res_images = low_res_images.to(self.device)
                high_res_images = high_res_images.to(self.device)
                outputs = model(low_res_images)
                loss = criterion(outputs, high_res_images)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.6f}")
        self.logger.info(f"Test Loss: {avg_test_loss:.6f}")
        self.experiment.log_metric("test_loss", avg_test_loss)
        self.log_images(self.experiment, test_loader, 'test')

    def log_images(self, experiment, data_loader, epoch):
        for low_res_images, high_res_images in data_loader:
            low_res_images = low_res_images.to(self.device)
            high_res_images = high_res_images.to(self.device)
            outputs = self.model(low_res_images)
            break
        num_images = min(5, low_res_images.size(0))
        indices = random.sample(range(low_res_images.size(0)), num_images)
        for idx in indices:
            lri = low_res_images[idx].cpu()
            ri = outputs[idx].cpu()
            hri = high_res_images[idx].cpu()
            lri_np = np.transpose(lri.detach().numpy(), (1, 2, 0))
            ri_np = np.transpose(ri.detach().numpy(), (1, 2, 0))
            hri_np = np.transpose(hri.detach().numpy(), (1, 2, 0))
            lri_np = (lri_np - lri_np.min()) / (lri_np.max() - lri_np.min() + 1e-8)
            ri_np = (ri_np - ri_np.min()) / (ri_np.max() - ri_np.min() + 1e-8)
            hri_np = (hri_np - hri_np.min()) / (hri_np.max() - hri_np.min() + 1e-8)
            combined = np.hstack([lri_np, ri_np, hri_np]) * 255
            combined = combined.astype(np.uint8)
            experiment.log_image(combined, name=f"superres_epoch_{epoch}_idx_{idx}", step=epoch)
        print(f"Logged {num_images} images for Epoch {epoch}")
