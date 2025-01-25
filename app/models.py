import torch
from torchvision.transforms import v2

from app.utils import Artwork
from inpainting.InpaintingGAN import GANInpainting
from superresolution.SuperResolutionTrainer import SuperResAutoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReconstructionModule:
    def __init__(self):
        self.resolution_model = self.load_resolution_model()
        self.inpainting_model = self.load_inpainting_model()
        self.to_input = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def load_resolution_model(self):
        # path = "models/super_hiper_resolutioner.pth"
        path = "/home/aneta/Documents/UNN/app/models/super_hiper_resolutioner.pth"
        model = SuperResAutoencoder()
        torch_path = torch.load(path, weights_only=True)
        model.load_state_dict(torch_path)
        model = model.to(device)
        model.eval()
        return model

    def resolution(self, artwork : Artwork):
        image = self.to_input(artwork.image)
        image = image.unsqueeze(0).to(device)
        new_image = self.resolution_model(image)
        new_image = new_image.detach().squeeze(0).cpu().permute(1,2,0).numpy()
        artwork.image = new_image
        return artwork

    def load_inpainting_model(self):
        # path = "models/super_hiper_inpainter.pth"
        path = "/home/aneta/Documents/UNN/app/models/super_hiper_inpainter.pth"
        model = GANInpainting()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.generator.to(device)
        model.eval()
        return model

    def inpainting(self, artwork : Artwork):
        image = self.to_input(artwork.image)
        cluster = torch.tensor(artwork.cluster_id).unsqueeze(0).to(device)
        mask = self.to_input(artwork.mask)
        damaged_image = torch.cat([image, mask], dim=0)
        damaged_image = damaged_image.unsqueeze(0).to(device)
        new_image = self.inpainting_model(damaged_image, cluster)
        new_image = new_image.detach().squeeze(0).cpu().permute(1,2,0).numpy()
        artwork.image = new_image
        return artwork

