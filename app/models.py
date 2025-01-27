import pickle
from pathlib import Path
import streamlit as st
import numpy as np
import torch
from PIL import Image as Img
from torchvision.transforms import v2

from app.utils import Artwork
from autoencoder.vqvae import VQVAE
from inpainting.InpaintingGAN import Generator
from superresolution.SuperResolutionTrainer import SuperResAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = Path("app/models")

class ReconstructionModule:
    def __init__(self):
        self.resolution_model = self._load_model(SuperResAutoencoder, "super_hiper_resolutioner_state_dict.pth")
        self.inpainting_model = self._load_model(Generator, "super_hiper_inpainter_state_dict.pth")
        self.classification_model, self.pca, self.kmeans = self.load_classification_model()

        self.preprocess = v2.Compose([
            v2.ToTensor(),
        ])

    def _load_model(self, model_class, filename):
        path = base_path / filename
        model = model_class()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to(device)
        model.eval()
        return model

    def load_classification_model(self):
        model = self._load_model(VQVAE, "super_hiper_classifier_state_dict.pth")

        with open(base_path / "pca.pkl", "rb") as f:
            pca = pickle.load(f)

        with open(base_path / "kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)

        return model, pca, kmeans

    @staticmethod
    def tensor_to_pil(image_tensor: torch.Tensor) -> Img:
        image_tensor = image_tensor.clamp(0, 1)
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return Img.fromarray(image)

    def classification(self, image_tensor: torch.Tensor) -> int:
        image_tensor = image_tensor.unsqueeze(0).to(device)
        latents = self.classification_model.encode(image_tensor)
        latents0 = torch.flatten(latents[0], start_dim=1).cpu()
        latents1 = torch.flatten(latents[1], start_dim=1).cpu()
        latents = torch.cat([latents0, latents1], dim=1).detach().numpy()
        embedding = self.pca.transform(latents)
        return self.kmeans.predict(embedding)[0]

    def resolution(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output_tensor = self.resolution_model(image_tensor).detach().squeeze(0)
        return output_tensor

    def inpainting(self, image_tensor: torch.Tensor, cluster_id: int, mask_tensor: torch.Tensor) -> torch.Tensor:
        cluster_tensor = torch.tensor([cluster_id], dtype=torch.int).to(device)
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=0).unsqueeze(0).to(device)
        output_tensor = self.inpainting_model(input_tensor, cluster_tensor).detach().squeeze(0)
        return output_tensor

    def pipeline(self, artwork: Artwork, is_inpainted: bool, is_super: bool) -> Artwork:
        image_tensor = self.preprocess(artwork.image)

        if is_inpainted:
            mask_tensor = self.preprocess(artwork.mask)
            cluster_id = self.classification(image_tensor)
            image_tensor = self.inpainting(image_tensor, cluster_id, mask_tensor)

        if is_super:
            image_tensor = self.resolution(image_tensor)

        artwork.result = self.tensor_to_pil(image_tensor)
        return artwork
