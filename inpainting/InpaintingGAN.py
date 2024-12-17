import numpy as np
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
import torchvision


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),  # Match 4 input channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),  # Keep dimensions
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_extractor = timm.create_model("resnet18", pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        validity = self.classifier(features)
        return validity

class GANInpainting(pl.LightningModule):
    def __init__(self, generator, discriminator, lr=1e-4):
        super(GANInpainting, self).__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.lr = lr

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, damaged_image):
        return self.generator(damaged_image)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        damaged_image, original_image = batch

        # -----------------------
        # Train Generator
        # -----------------------
        # Generate inpainted images
        inpainted_image = self(damaged_image)

        # Adversarial loss (fool the discriminator)
        fake_preds = self.discriminator(inpainted_image)
        valid_labels = torch.ones_like(fake_preds, device=self.device)
        g_adv_loss = self.adversarial_loss(fake_preds, valid_labels)

        # Reconstruction loss (difference with original image)
        g_rec_loss = self.reconstruction_loss(inpainted_image, original_image)

        # Total generator loss
        g_loss = g_adv_loss + 100 * g_rec_loss
        self.log("g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Backpropagation and optimization step for the generator
        self.manual_backward(g_loss)
        g_opt.step()
        g_opt.zero_grad()

        # -----------------------
        # Train Discriminator
        # -----------------------
        # Real images
        real_preds = self.discriminator(original_image)
        valid_labels = torch.ones_like(real_preds, device=self.device)
        d_real_loss = self.adversarial_loss(real_preds, valid_labels)

        # Fake images
        fake_images = self(damaged_image).detach()  # Detach to prevent gradients
        fake_preds = self.discriminator(fake_images)
        fake_labels = torch.zeros_like(fake_preds, device=self.device)
        d_fake_loss = self.adversarial_loss(fake_preds, fake_labels)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Backpropagation and optimization step for the discriminator
        self.manual_backward(d_loss)
        d_opt.step()
        d_opt.zero_grad()

        return {"g_loss": g_loss, "d_loss": d_loss}

    def validation_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        inpainted_image = self(damaged_image)

        g_rec_loss = self.reconstruction_loss(inpainted_image, original_image)
        self.log("val_loss", g_rec_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log images every epoch
        if batch_idx == 0:  # Log only for the first batch
            grid = self._create_image_grid(damaged_image, inpainted_image, original_image)
            self.logger.experiment.log_image(grid, name=f"val_epoch_{self.current_epoch:04}.png")

        return g_rec_loss

    def test_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        inpainted_image = self(damaged_image)

        # grid = self._create_image_grid(damaged_image, inpainted_image, original_image)
        # output_dir = "test_images"
        # os.makedirs(output_dir, exist_ok=True)
        # save_path = os.path.join(output_dir, f"test_batch_{batch_idx}.png")
        # torchvision.utils.save_image(grid, save_path)

        # print(f'original_image:{original_image.shape} damaged_image:{damaged_image.shape} inpainted_image:{inpainted_image.shape}')

        self.log("test_loss", self.reconstruction_loss(inpainted_image, original_image))
        if batch_idx == 0:
            grid = self._create_image_grid(damaged_image, inpainted_image, original_image)
            self.logger.experiment.log_image(grid, name=f"test.png")

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt]

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