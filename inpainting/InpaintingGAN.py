import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, num_residual_blocks=6, base_features=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        self.down = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_features * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_features * 4),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_features * 4) for _ in range(num_residual_blocks)]
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(base_features, out_channels, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.residual_blocks(x)
        x = self.up(x)
        return self.final(x)


class PatchCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)


class SharedFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.vgg11(weights='IMAGENET1K_V1').features
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
        loss = sum(F.l1_loss(p, t) for p, t in zip(pred_features, target_features))
        return loss



class TextureLoss(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_ids = [0, 1]

    @staticmethod
    def gram_matrix(features):
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.permute(0, 2, 1))
        return gram / (c * h * w)

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred, self.layer_ids)
        target_features = self.feature_extractor(target, self.layer_ids)
        loss = sum(F.l1_loss(self.gram_matrix(p), self.gram_matrix(t))
                   for p, t in zip(pred_features, target_features))
        return loss

class DiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated_images):
        batch_size = generated_images.size(0)
        diversity_loss = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                diversity_loss += F.mse_loss(generated_images[i], generated_images[j])
        return -diversity_loss / (batch_size * (batch_size - 1))

class GANInpainting(pl.LightningModule):
    def __init__(self, lr=1e-4, lambda_gp=20,
                 g_adv_weight=1.0, g_rec_weight=2.0, g_perc_weight=3.0, g_tex_weight=2.0, g_div_weight=3.0,):
        super(GANInpainting, self).__init__()
        self.save_hyperparameters()

        self.generator = Generator()
        self.critique = PatchCritic()
        self.lr = lr
        self.n_critic = 10

        self.feature_extractor = SharedFeaturesExtractor().eval()
        self.perceptual_loss = PerceptualLoss(self.feature_extractor)
        self.reconstruction_loss = nn.L1Loss()
        self.texture_loss = TextureLoss(self.feature_extractor)
        self.diversity_loss = DiversityLoss()
        self.automatic_optimization = False

        self.lambda_gp = lambda_gp
        self.g_adv_weight = g_adv_weight
        self.g_rec_weight = g_rec_weight
        self.g_perc_weight = g_perc_weight
        self.g_tex_weight = g_tex_weight
        self.g_div_weight = g_div_weight

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
            c_loss = torch.mean(fake_preds) - torch.mean(real_preds)
            gp = self.gradient_penalty(original_image, fake_images)
            c_loss += self.lambda_gp * gp
            self.log("critique_loss", c_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.manual_backward(c_loss)
            c_opt.step()
            c_opt.zero_grad()
            self.clip_critic_weights()

        # -----------------------
        # Train Generator
        # -----------------------
        inpainted_image = self(damaged_image)
        fake_preds = self.critique(inpainted_image)
        g_adv_loss = -torch.mean(fake_preds)
        g_perc_loss = self.perceptual_loss(inpainted_image, original_image)
        g_rec_loss = self.reconstruction_loss(inpainted_image, original_image)
        g_tex_loss = self.texture_loss(inpainted_image, original_image)
        g_div_loss = self.diversity_loss(inpainted_image)

        g_loss = (
            self.g_adv_weight * g_adv_loss +
            self.g_rec_weight * g_rec_loss +
            self.g_perc_weight * g_perc_loss +
            self.g_tex_weight * g_tex_loss +
            self.g_div_weight * g_div_loss
        )

        self.log("generator_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("g_adv_loss", g_adv_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("g_rec_loss", g_rec_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("g_perc_loss", g_perc_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("g_tex_loss", g_tex_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("g_div_loss", g_div_loss, on_step=False, on_epoch=True, prog_bar=False)

        self.manual_backward(g_loss)
        g_opt.step()
        g_opt.zero_grad()

        return {"generator_loss": g_loss, "critique_loss": c_loss}

    def clip_critic_weights(self):
        for p in self.critique.parameters():
            p.data.clamp_(-0.01, 0.01)

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
        gradient_norm = gradients.norm(2, dim=1) + 1e-8
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
        d_opt = torch.optim.Adam(self.critique.parameters(), lr=self.lr * 0.5, betas=(0.5, 0.999))
        return [g_opt, d_opt]

    def _create_image_grid(self, damaged_image, inpainted_image, original_image):
        damaged_image = damaged_image[:, :3, :, :].cpu()
        inpainted_image = inpainted_image[:, :3, :, :].cpu()
        original_image = original_image[:, :3, :, :].cpu()

        grid = torchvision.utils.make_grid(
            torch.cat([damaged_image, inpainted_image, original_image], dim=0),
            nrow=damaged_image.size(0),
        )

        grid = grid.permute(1, 2, 0).numpy()
        grid = (grid * 255).astype(np.uint8)

        return grid
