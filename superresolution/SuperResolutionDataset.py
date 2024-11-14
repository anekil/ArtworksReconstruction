from torchvision import models, transforms
from UNN.dataloading.WikiArtDataset import WikiArtDataset


class SuperResolutionDataset(WikiArtDataset):
    def __init__(self, df, transform=None, upscale_factor=2):
        super().__init__(df, transform)
        self.upscale_factor = upscale_factor

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        low_res_image = transforms.functional.resize(
            image, [image.size(1) // self.upscale_factor, image.size(2) // self.upscale_factor],
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        low_res_upsampled = transforms.functional.resize(
            low_res_image, [image.size(1), image.size(2)],
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        return low_res_upsampled, image
