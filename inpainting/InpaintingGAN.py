import numpy as np
import torch
import torch.nn as nn
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
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
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


class Critique(nn.Module):
    def __init__(self):
        super(Critique, self).__init__()
        self.feature_extractor = torchvision.models.squeezenet1_1(weights='DEFAULT')
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GANInpainting(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super(GANInpainting, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator()
        self.critique = Critique()
        self.lr = lr
        self.n_critic = 5
        self.lambda_gp = 10.0

        self.adversarial_loss = None
        self.reconstruction_loss = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, damaged_image):
        return self.generator(damaged_image)

    def training_step(self, batch, batch_idx):
        g_opt, c_opt = self.optimizers()
        damaged_image, original_image = batch

        # -----------------------
        # Train Critique
        # -----------------------
        for _ in range(self.n_critic):
            real_preds = self.critique(original_image)
            fake_images = self(damaged_image).detach()
            fake_preds = self.critique(fake_images)

            c_loss = -(real_preds.mean() - fake_preds.mean())

            gp = self.gradient_penalty(original_image, fake_images)
            c_loss += self.lambda_gp * gp

            self.log("critique_loss", c_loss, on_step=True, on_epoch=True, prog_bar=True)

            self.manual_backward(c_loss)
            c_opt.step()
            c_opt.zero_grad()

        # -----------------------
        # Train Generator
        # -----------------------
        inpainted_image = self(damaged_image)

        fake_preds = self.critique(inpainted_image)
        g_adv_loss = -fake_preds.mean()

        g_rec_loss = self.reconstruction_loss(inpainted_image, original_image)

        g_loss = g_adv_loss + 100 * g_rec_loss
        self.log("generator_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.manual_backward(g_loss)
        g_opt.step()
        g_opt.zero_grad()

        return {"generator_loss": g_loss, "critique_loss": c_loss}

    def gradient_penalty(self, real_images, fake_images):
        batch_size, c, h, w = real_images.size()
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device).expand_as(real_images)
        interpolated = epsilon * real_images + (1 - epsilon) * fake_images
        interpolated.requires_grad_(True)

        interpolated_preds = self.critique(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_preds,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_preds, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gp = ((gradient_norm - 1) ** 2).mean()
        return gp

    def validation_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        inpainted_image = self(damaged_image)

        g_rec_loss = self.reconstruction_loss(inpainted_image, original_image)
        self.log("val_loss", g_rec_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            grid = self._create_image_grid(damaged_image, inpainted_image, original_image)
            self.logger.experiment.log_image(grid, name=f"val_epoch_{self.current_epoch:04}.png")

        return g_rec_loss

    def test_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        inpainted_image = self(damaged_image)

        self.log("test_loss", self.reconstruction_loss(inpainted_image, original_image))
        if batch_idx == 0:
            grid = self._create_image_grid(damaged_image, inpainted_image, original_image)
            self.logger.experiment.log_image(grid, name=f"test.png")

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.critique.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt]

    def _create_image_grid(self, damaged_image, inpainted_image, original_image):
        damaged_image = damaged_image[:, :3, :, :].cpu()
        inpainted_image = inpainted_image[:, :3, :, :].cpu()
        original_image = original_image[:, :3, :, :].cpu()
        mask = damaged_image[:, 3:4, :, :].cpu()
        reconstructed_image = mask * inpainted_image + (1 - mask) * original_image

        grid = torchvision.utils.make_grid(
            torch.cat([damaged_image, inpainted_image, reconstructed_image, original_image], dim=0),
            nrow=damaged_image.size(0),
        )

        grid = grid.permute(1, 2, 0).numpy()
        grid = (grid * 255).astype(np.uint8)

        return grid
