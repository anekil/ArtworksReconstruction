import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import logging
from comet_ml import Experiment
import time
from torchvision import models, transforms
def initialize_model(self):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity  # Skip connection
            return self.relu(out)
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
        class ResidualBlock(nn.Module):
            def __init__(self):
                super(ResidualBlock, self).__init__()

                # Encoder: Pretrained ResNet backbone
                resnet = models.resnet34(pretrained=True)
                self.encoder_layers = nn.ModuleList([
                    nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),  # Layer 1
                    resnet.layer1,  # Layer 2
                    resnet.layer2,  # Layer 3
                    resnet.layer3,  # Layer 4
                    resnet.layer4  # Layer 5
                ])

                # Decoder: Deconvolutional layers with skip connections
                self.decoder_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Sequential(
                        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ),
                    nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Final reconstruction
                ])

                self.final_activation = nn.Sigmoid()

            def forward(self, x):
                # Encoder forward pass
                encoder_features = []
                for layer in self.encoder_layers:
                    x = layer(x)
                    encoder_features.append(x)

                # Decoder forward pass with skip connections
                for idx, layer in enumerate(self.decoder_layers):
                    x = layer(x)
                    if idx < len(encoder_features) - 1:  # Apply skip connection
                        skip_feature = encoder_features[-(idx + 2)]  # Corresponding encoder feature
                        x += nn.functional.interpolate(skip_feature, size=x.shape[2:], mode='bilinear',
                                                       align_corners=False)

                x = self.final_activation(x)
                return x

        class EncoderDecoder(nn.Module):
            def __init__(self):
                super(EncoderDecoder, self).__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualBlock(128, 128),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualBlock(256, 256),
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    ResidualBlock(512, 512),
                )
                # Decoder
                self.decoder = nn.Sequential(
                    ResidualBlock(512, 512),
                    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    ResidualBlock(256, 256),
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    ResidualBlock(128, 128),
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        model = EncoderDecoder()
        return model.to(self.device)

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
                percent = 100. * batch_idx / len(train_loader)
                print(f"Train Epoch: {epoch + 1} [{current}/{total} ({percent:.0f}%)]\tLoss: {loss.item():.6f}\tTime: {batch_time:.2f}s")
                self.experiment.log_metric("batch_train_loss", loss.item(), step=global_step)
                self.experiment.log_metric("batch_time", batch_time, step=global_step)

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
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
                #torch.save(model.state_dict(), f"{self.config['model_save_path']}/best_superres_model.pth")
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
            lri_np = (lri_np - lri_np.min()) / (lri_np.max() - lri_np.min())
            ri_np = (ri_np - ri_np.min()) / (ri_np.max() - ri_np.min())
            hri_np = (hri_np - hri_np.min()) / (hri_np.max() - hri_np.min())

                # Low-res | Reconstructed | High-res
            combined_image = np.hstack((lri_np, ri_np, hri_np))
            combined_image = (combined_image * 255).astype(np.uint8)
            experiment.log_image(
                combined_image,
                name=f"superres_epoch_{epoch}_idx_{idx}",
                step=epoch
            )
        print(f"Logged {num_images} images for Epoch {epoch}")
