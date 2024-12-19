import numpy as np
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
        return self.activation(x)

class PaintingBranch(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.ModuleList()
        for feature in features[::-1]:
            self.layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                DepthwiseSeparableConv(feature, feature),
                nn.Conv2d(feature, 3, kernel_size=1)
            ))

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: Downsampling path
        for feature in features:
            self.encoder.append(DepthwiseSeparableConv(in_channels, feature))
            self.encoder.append(nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv(features[-1], features[-1] * 2)

        # Decoder: Upsampling path
        for feature in reversed(features):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature * 2, feature, kernel_size=1),
                DepthwiseSeparableConv(feature, feature)
            ))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Main Branch
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            # x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            x = torch.cat((x, skip_connections[idx]), dim=1)

        # Painting Branch
        painting_outputs = self.painting_branch(x)

        # Final Composition
        composite_output = painting_outputs[-1]  # Use final painting branch output for simplicity
        return composite_output, painting_outputs

    # def forward(self, x):
    #     """
    #     Args:
    #         x: masked input image [batch_size, 3, H, W]
    #         mask: binary mask [batch_size, 1, H, W]
    #     Returns:
    #         Output: inpainted regions combined with input
    #     """
    #     # x = torch.cat((x, mask), dim=1)  # Combine image and mask as 4 channels
    #     skip_connections = []
    #
    #     # Encoder
    #     for layer in self.encoder:
    #         x = layer(x)
    #         skip_connections.append(x)
    #         x = self.pool(x)
    #
    #     # Bottleneck
    #     x = self.bottleneck(x)
    #     skip_connections = skip_connections[::-1]
    #
    #     # Decoder
    #     for idx, layer in enumerate(self.decoder):
    #         x = layer(x)
    #         x = torch.cat((x, skip_connections[idx]), dim=1)
    #
    #     output = self.final_conv(x)
    #     return torch.sigmoid(output)

class Discriminator(nn.Module):
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

# class Critique(nn.Module):
#     def __init__(self):
#         super(Critique, self).__init__()
#         self.feature_extractor = timm.create_model("resnet18", pretrained=True, num_classes=0)
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, x):
#         features = self.feature_extractor(x)
#         validity = self.classifier(features)
#         return validity

class InpaintingMOGAN(pl.LightningModule):
    def __init__(self, lr_g=1e-4, lr_d=4e-4):
        super().__init__()
        self.save_hyperparameters()

        # Generator and Discriminator
        self.generator = UNetGenerator()
        self.discriminator = Discriminator()

        # Loss Functions
        # self.adv_loss = nn.BCEWithLogitsLoss()
        # self.recon_loss = nn.L1Loss()
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_pred, is_real):
        """Adversarial loss to distinguish real and fake images."""
        target = torch.ones_like(y_pred) if is_real else torch.zeros_like(y_pred)
        return self.adv_loss(y_pred, target)

    def loss_function(self, composite_output, painting_outputs, real_image, mask):
        loss_composite = F.mse_loss(composite_output, real_image)
        loss_painting = sum(F.mse_loss(out, real_image) for out in painting_outputs) / len(painting_outputs)
        return loss_composite + loss_painting

    def training_step(self, batch, batch_idx):
        real_image, mask = batch  # Ground truth image and mask
        masked_input = real_image * (1 - mask)  # Create corrupted input by applying the mask

        # Fetch optimizers
        opt_gen, opt_disc = self.optimizers()

        # -------------------------
        # 1. Generator Step
        # -------------------------
        composite_output, painting_outputs = self.generator(masked_input)  # Generator output

        # Compute generator loss
        gen_loss = self.loss_function(composite_output, painting_outputs, real_image, mask)

        # Manual backward for the generator
        self.manual_backward(gen_loss, opt_gen)
        opt_gen.step()
        opt_gen.zero_grad()

        # Log generator loss
        self.log("gen_loss", gen_loss, on_epoch=True, prog_bar=True)

        # -------------------------
        # 2. Discriminator Step
        # -------------------------
        composite_output = composite_output.detach()  # Detach to avoid gradients from generator
        real_pred = self.discriminator(real_image)  # Discriminator on real image
        fake_pred = self.discriminator(composite_output)  # Discriminator on fake image

        # Discriminator loss
        real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        disc_loss = (real_loss + fake_loss) / 2

        # Manual backward for the discriminator
        self.manual_backward(disc_loss, opt_disc)
        opt_disc.step()
        opt_disc.zero_grad()

        # Log discriminator loss
        self.log("disc_loss", disc_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        return g_optimizer, d_optimizer

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch: Tuple (masked_image, mask, original_image)
            batch_idx: Index of the current batch
        Returns:
            Dictionary containing validation losses and metrics
        """
        masked_image, original_image = batch

        # Generate inpainted regions
        inpainted_image = self.generator(masked_image)

        # Combine inpainted regions with unmasked parts of the input
        reconstructed_image = inpainted_image
        # reconstructed_image = masked_image * mask + inpainted_image * (1 - mask)

        # Compute losses
        reconstruction_loss = nn.functional.l1_loss(reconstructed_image, original_image)  # L1 loss
        fake_pred = self.discriminator(reconstructed_image)
        adversarial_loss = nn.functional.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

        total_loss = reconstruction_loss + 0.01 * adversarial_loss  # Weighted loss combination

        # Compute metrics (e.g., PSNR, SSIM)
        psnr = self.compute_psnr(reconstructed_image, original_image)
        ssim = self.compute_ssim(reconstructed_image, original_image)

        # Log losses and metrics
        self.log("val_reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("val_adversarial_loss", adversarial_loss, prog_bar=True)
        self.log("val_total_loss", total_loss, prog_bar=True)
        self.log("val_psnr", psnr, prog_bar=True)
        self.log("val_ssim", ssim, prog_bar=True)

        if batch_idx == 0:
            grid = self._create_image_grid(masked_image, reconstructed_image, original_image)
            self.logger.experiment.log_image(grid, name=f"val_epoch_{self.current_epoch:04}.png")

        return {"loss": total_loss, "psnr": psnr, "ssim": ssim}

    @staticmethod
    def compute_psnr(predicted, target, max_pixel_value=1.0):
        """Computes Peak Signal-to-Noise Ratio (PSNR)."""
        mse = nn.functional.mse_loss(predicted, target)
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        return psnr

    @staticmethod
    def compute_ssim(predicted, target):
        """Computes Structural Similarity Index (SSIM)."""
        import pytorch_msssim
        ssim = pytorch_msssim.ssim(predicted, target, data_range=1.0)
        return ssim

    def test_step(self, batch, batch_idx):
        """
        Args:
            batch: Tuple (masked_image, mask, original_image)
            batch_idx: Index of the current batch
        Returns:
            Dictionary containing test losses and metrics
        """
        masked_image, original_image = batch

        # Generate inpainted regions
        inpainted_image = self.generator(masked_image)

        # Combine inpainted regions with unmasked parts
        reconstructed_image = inpainted_image
        # reconstructed_image = masked_image * mask + inpainted_image * (1 - mask)

        # Compute losses
        reconstruction_loss = nn.functional.l1_loss(reconstructed_image, original_image)
        fake_pred = self.discriminator(reconstructed_image)
        adversarial_loss = nn.functional.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

        total_loss = reconstruction_loss + 0.01 * adversarial_loss

        # Compute metrics (PSNR, SSIM)
        psnr = self.compute_psnr(reconstructed_image, original_image)
        ssim = self.compute_ssim(reconstructed_image, original_image)

        if batch_idx == 0:
            grid = self._create_image_grid(masked_image, reconstructed_image, original_image)
            self.logger.experiment.log_image(grid, name=f"test.png")

        # Log metrics
        self.log("test_reconstruction_loss", reconstruction_loss)
        self.log("test_adversarial_loss", adversarial_loss)
        self.log("test_total_loss", total_loss)
        self.log("test_psnr", psnr)
        self.log("test_ssim", ssim)

        return {"loss": total_loss, "psnr": psnr, "ssim": ssim}


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