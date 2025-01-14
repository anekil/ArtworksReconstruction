import numpy as np
import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.LeakyReLU(0.2) if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x

class ResizeConv(DepthwiseSeparableConv):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=True, scale=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding, activation)
        self.scale = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = super().forward(x)
        x = self.scale(x)
        return x


class PaintingBranch(nn.Module):
    def __init__(self, features, out_channels=3):
        super().__init__()

        self.first_layer = nn.Sequential(
            ResizeConv(out_channels, out_channels, scale=2)
        )
        self.intermediate_layer = nn.Sequential(
            ResizeConv(out_channels * 2, out_channels, scale=2)
        )
        self.last_layer = ResizeConv(out_channels * 2, out_channels, scale=1)

        self.to_rgb = nn.ModuleList([
            nn.Conv2d(feature, out_channels, kernel_size=1) for feature in features
        ])

    def forward(self, skip_connections):
        x = None
        for to_rgb, skip in zip(self.to_rgb[:-1], skip_connections[:-1]):
            skip = to_rgb(skip)
            if x is None:
                x = self.first_layer(skip)
            else:
                x = torch.cat((x, skip), dim=1)
                x = self.intermediate_layer(x)
                x = F.leaky_relu(x, 0.2)
        last_skip = self.to_rgb[-1](skip_connections[-1])
        x = torch.cat((x, last_skip), dim=1)
        x = self.last_layer(x)
        return x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in features:
            self.encoder.append(ResizeConv(in_channels, feature, scale=0.5))
            in_channels = feature

        self.bottleneck = DepthwiseSeparableConv(features[-1], features[-1])

        features = features[::-1]
        for feature in features:
            self.decoder.append(ResizeConv(in_channels + feature, feature, scale=2))
            in_channels = feature

        features.insert(0, features[0])
        self.last_layer = nn.Sequential(
            ResizeConv(in_channels, in_channels, scale=1),
            ResizeConv(in_channels, out_channels, scale=1)
        )
        self.painting_branch = PaintingBranch(features, out_channels)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        painting_connections = []
        x = self.bottleneck(x)
        painting_connections.append(x)

        skip_connections = skip_connections[::-1]

        for layer, skip in zip(self.decoder, skip_connections):
            x = torch.cat((x, skip), dim=1)
            x = layer(x)
            painting_connections.append(x)

        x = self.painting_branch(painting_connections)
        return torch.sigmoid(x)

class Critique(nn.Module):
    def __init__(self):
        super(Critique, self).__init__()
        self.feature_extractor = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:6])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        validity = self.classifier(features)
        return validity

class InpaintingMIGAN(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super(InpaintingMIGAN, self).__init__()
        self.save_hyperparameters()

        self.generator = UNetGenerator()
        self.critique = Critique()
        self.lr = lr
        self.n_critic = 5
        self.lambda_gp = 10.0

        self.reconstruction_loss = nn.L1Loss()
        self.automatic_optimization = False

    def forward(self, damaged_image):
        return self.generator(damaged_image)

    def training_step(self, batch, batch_idx):
        g_opt, c_opt = self.optimizers()
        damaged_image, original_image = batch

        mask = damaged_image[:, 3:4, :, :]
        damaged_img = damaged_image[:, :3, :, :]

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

        inpainted_image = self(damaged_image)

        fake_preds = self.critique(inpainted_image)
        g_adv_loss = -fake_preds.mean()
        g_rec_loss = self.reconstruction_loss(inpainted_image * mask, original_image * mask)

        g_loss = g_adv_loss + 50 * g_rec_loss
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
        mask = damaged_image[:, :3, :, :]

        g_rec_loss = self.reconstruction_loss(inpainted_image * mask, original_image * mask)
        self.log("val_loss", g_rec_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            grid = self._create_image_grid(damaged_image, inpainted_image, original_image)
            self.logger.experiment.log_image(grid, name=f"val_epoch_{self.current_epoch:04}.png")

        return g_rec_loss

    def test_step(self, batch, batch_idx):
        damaged_image, original_image = batch
        inpainted_image = self(damaged_image)

        mask = damaged_image[:, 3, :, :]
        damaged_img = damaged_image[:, :3, :, :]

        mask = mask.unsqueeze(1)
        mask_3d = mask.expand_as(damaged_img)
        reconstructed_image = mask_3d * damaged_img + (1 - mask_3d) * inpainted_image

        self.log("test_loss", self.reconstruction_loss(reconstructed_image, original_image))
        if batch_idx == 0:
            grid = self._create_image_grid(damaged_image, reconstructed_image, original_image)
            self.logger.experiment.log_image(grid, name=f"test.png")

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(self.critique.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt]

    def _create_image_grid(self, damaged_image, inpainted_image, original_image):
        damaged_image = damaged_image.cpu()
        inpainted_image = inpainted_image.cpu()
        original_image = original_image.cpu()
        mask = damaged_image[:, 3, :, :]
        damaged_image = damaged_image[:, :3, :, :]

        mask = mask.unsqueeze(1)
        mask_3d = mask.expand_as(damaged_image)
        reconstructed_image = mask_3d * damaged_image + (1 - mask_3d) * inpainted_image

        grid = torchvision.utils.make_grid(
            torch.cat([damaged_image, inpainted_image, reconstructed_image, original_image], dim=0),
            nrow=damaged_image.size(0),
        )

        grid = grid.permute(1, 2, 0).numpy()
        grid = (grid * 255).astype(np.uint8)

        return grid
