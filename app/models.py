import pickle
from pathlib import Path

import streamlit
import torch
from PIL.Image import Image
from torchvision.transforms import v2

from app.utils import Artwork
from autoencoder.vqvae import VQVAE
from inpainting.InpaintingGAN import Generator
from superresolution.SuperResolutionTrainer import SuperResAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = Path("app/models")

class ReconstructionModule:
    def __init__(self):
        self.resolution_model = self.load_resolution_model()
        self.inpainting_model = self.load_inpainting_model()
        self.classification_model, self.pca, self.kmeans = self.load_classification_model()

        self.to_input = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def load_classification_model(self):
        path = base_path / "super_hiper_classifier_state_dict.pth"
        model = VQVAE()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.to(device)
        model.eval()

        path = base_path / "pca.pkl"
        with open(path, "rb") as f:
            pca = pickle.load(f)

        path = base_path / "kmeans.pkl"
        with open(path, "rb") as f:
            kmeans = pickle.load(f)

        return model, pca, kmeans

    def classification(self, image : Image):
        image = self.to_input(image)
        image = image.unsqueeze(0).to(device)
        embedding = self.classification_model(image)
        embedding = self.pca.transform(embedding)
        cluster_id = self.kmeans.predict(embedding)
        return cluster_id

    def load_resolution_model(self):
        path = base_path / "super_hiper_resolutioner_state_dict.pth"
        model = SuperResAutoencoder()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.to(device)
        model.eval()
        return model

    def resolution(self, image : Image):
        image = self.to_input(image)
        image = image.unsqueeze(0).to(device)
        new_image = self.resolution_model(image)
        new_image = new_image.detach().squeeze(0).cpu().permute(1,2,0).numpy()
        return new_image

    def load_inpainting_model(self):
        path = base_path / "super_hiper_inpainter_state_dict.pth"
        model = Generator()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.to(device)
        model.eval()
        return model

    def inpainting(self, image : Image, cluster_id : int, mask : Image):
        image = self.to_input(image)
        cluster = torch.tensor(cluster_id).unsqueeze(0).to(device)
        mask = self.to_input(mask)
        damaged_image = torch.cat([image, mask], dim=0)
        damaged_image = damaged_image.unsqueeze(0).to(device)
        new_image = self.inpainting_model(damaged_image, cluster)
        new_image = new_image.detach().squeeze(0).cpu().permute(1,2,0).numpy()
        return new_image

    def pipeline(self, artwork: Artwork, is_inpainted: bool, is_super: bool):
        image = artwork.image
        cluster_id = self.classification(image)
        if is_inpainted:
            image = self.inpainting(image, cluster_id, artwork.mask)
        if is_super:
            image = self.resolution(image)
        artwork.result = image
        return artwork
