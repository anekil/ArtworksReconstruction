import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models
import pytorch_lightning as pl
import torchvision.utils as vutils


class SharedFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vgg16(weights="IMAGENET1K_V1").features
        self.layers = nn.ModuleList([
            model[0:2],
            model[2:4],
            model[4:7],
        ])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, layer_ids):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in layer_ids:
                features.append(x)
        return features


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_ids = [0, 1, 2]

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred, self.layer_ids)
        target_features = self.feature_extractor(target, self.layer_ids)
        return sum(F.l1_loss(p, t) for p, t in zip(pred_features, target_features))


class ResNet50Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class SuperResolutionLightningModule(pl.LightningModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = ResNet50Autoencoder()
        self.perceptual_loss = PerceptualLoss(SharedFeaturesExtractor())
        self.mse_loss = nn.MSELoss()
        self.dataset = dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        low_res, high_res = batch
        outputs = self(low_res)
        p_loss = self.perceptual_loss(outputs, high_res)
        m_loss = self.mse_loss(outputs, high_res)
        loss = p_loss + 10 * m_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_p_loss", p_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_m_loss", m_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        low_res, high_res = batch
        outputs = self(low_res)
        p_loss = self.perceptual_loss(outputs, high_res)
        m_loss = self.mse_loss(outputs, high_res)
        loss = p_loss + + 10 * m_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_p_loss", p_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_m_loss", m_loss, on_step=False, on_epoch=True, prog_bar=False)
        if batch_idx < 5:
            self.log_images(low_res, outputs, high_res, epoch=self.current_epoch, batch_idx=batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        low_res, high_res = batch
        outputs = self(low_res)
        p_loss = self.perceptual_loss(outputs, high_res)
        m_loss = self.mse_loss(outputs, high_res)
        loss = p_loss + + 10 * m_loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_p_loss", p_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test_m_loss", m_loss, on_step=False, on_epoch=True, prog_bar=False)
        if batch_idx < 5:
            self.log_images(low_res, outputs, high_res, epoch=self.current_epoch, batch_idx=batch_idx, stage="test")

    def log_images(self, low_res, outputs, high_res, epoch, batch_idx, stage):
        triplets = torch.cat([low_res, outputs, high_res], dim=-1)
        grid = vutils.make_grid(triplets, nrow=1, normalize=True, value_range=(0, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        filename = f"superres_{stage}_epoch_{str(epoch).zfill(3)}_batch_{batch_idx}.png"
        plt.imsave(filename, grid_np)
        self.logger.experiment.log_image(filename, name=filename)
        print(f"Logged image: {filename}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def on_train_end(self):
        experiment_name = self.logger.experiment.get_name()
        filename = f"superres_{experiment_name}_model.ckpt"
        path = os.path.join(self.hparams.get("save_dir", "."), filename)
        self.trainer.save_checkpoint(path)
        self.logger.experiment.log_asset(path)
        print(f"Model saved as: {path}")

    def setup(self, stage=None):
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)

