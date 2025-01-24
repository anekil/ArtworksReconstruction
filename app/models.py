import torch

from inpainting.InpaintingGAN import GANInpainting
from superresolution.SuperResolutionTrainer import SuperResAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReconstructionModule:
    def __init__(self):
        self.resolution_model = self.load_resolution_model()
        self.inpainting_model = self.load_inpainting_model()

    def load_resolution_model(self):
        path = "super_hiper_resolutioner.pth"
        model = SuperResAutoencoder()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.to(device)
        model.eval()
        return model


    def resolution(self, image):
        return self.resolution_model(image)

    def load_inpainting_model(self):
        path = "super_hiper_inpainter.pth"
        model = GANInpainting()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.generator.to(device)
        model.eval()
        return model

    def inpainting(self, image, cluster_id=0):
        return self.inpainting_model(image, cluster_id)

