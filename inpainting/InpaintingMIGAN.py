import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision


class DepthwiseSeparableConv(nn.Module):
    """Lightweight convolutional block with depthwise separable convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.LeakyReLU(0.2) if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(x)


class UNetGenerator(nn.Module):
    """Lightweight U-Net generator with skip connections."""
    def __init__(self, in_channels=4, out_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(DepthwiseSeparableConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv(features[-1], features[-1])

        # Decoder
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(in_channels * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DepthwiseSeparableConv(feature, feature))
            in_channels = feature

        # Final output layer
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            print('encoder', x.shape)
            x = layer(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat((x, skip_connections[idx]), dim=1)

        return self.final_conv(x)


class Discriminator(nn.Module):
    """Patch-based discriminator for adversarial loss."""
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256]
        layers = []
        for feature in features:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = feature
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class InpaintingMIGAN(pl.LightningModule):
    """MI-GAN-inspired lightweight inpainting model."""
    def __init__(self, lr_g=1e-4, lr_d=4e-4, lambda_adv=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.generator = UNetGenerator()
        self.discriminator = Discriminator()

        self.lambda_adv = lambda_adv
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = nn.L1Loss()

        # Enable manual optimization
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_pred, is_real):
        target = torch.ones_like(y_pred) if is_real else torch.zeros_like(y_pred)
        return self.adv_loss(y_pred, target)

    def training_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        opt_g, opt_d = self.optimizers()

        # Generator step
        fake_image = self.generator(damaged_image)
        fake_pred = self.discriminator(fake_image)
        g_loss = self.recon_loss(fake_image, original_image) + \
                 self.lambda_adv * self.adversarial_loss(fake_pred, True)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()

        # Discriminator step
        fake_image = fake_image.detach()
        real_pred = self.discriminator(original_image)
        fake_pred = self.discriminator(fake_image)
        d_loss = 0.5 * (self.adversarial_loss(real_pred, True) +
                        self.adversarial_loss(fake_pred, False))
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()

    def validation_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        reconstructed_image = self.generator(damaged_image)
        val_loss = self.recon_loss(reconstructed_image, original_image)
        self.log("val_loss", val_loss, prog_bar=True)

        if batch_idx == 0:
            grid = self._create_image_grid(damaged_image, reconstructed_image, original_image)
            self.logger.experiment.log_image(grid, name=f"val_epoch_{self.current_epoch:04}.png")

    def test_step(self, batch, batch_idx):
        original_image, mask = batch
        masked_image = original_image * (1 - mask)
        reconstructed_image = self.generator(masked_image)
        val_loss = self.recon_loss(reconstructed_image, original_image)
        self.log("val_loss", val_loss, prog_bar=True)

        if batch_idx == 0:
            grid = self._create_image_grid(masked_image, reconstructed_image, original_image)
            self.logger.experiment.log_image(grid, name=f"test.png")


    def _create_image_grid(self, damaged_image, inpainted_image, original_image):
        """
        Create a grid of images for visualization:
        Row 1: Damaged images
        Row 2: Inpainted images
        Row 3: Original images
        """
        # Denormalize images for visualization
        damaged_image = damaged_image[:,: 3, :, :].cpu()
        inpainted_image = inpainted_image[:,: 3, :, :].cpu()
        original_image = original_image[:,: 3, :, :].cpu()

        grid = torchvision.utils.make_grid(
            torch.cat([damaged_image, inpainted_image, original_image], dim=0),
            nrow=damaged_image.size(0),
        )

        grid = grid.permute(1, 2, 0).numpy()
        grid = (grid * 255).astype(np.uint8)

        return grid

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.5, 0.999))
        return [g_optimizer, d_optimizer]