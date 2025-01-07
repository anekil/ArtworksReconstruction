from PIL import Image
from torchvision import models, transforms
from UNN.dataloading.WikiArtDataset import WikiArtDataset


class SuperResolutionDataset(WikiArtDataset):
    def __init__(self, image_paths, transform=None, input_size=(224,224), output_size=(448,448)):
        self.image_paths = image_paths
        self.input_size = input_size
        self.transform = transform
        self.original_transform = transforms.Compose([
            transforms.CenterCrop(output_size),
            transforms.ToTensor()
        ])
        self.low_res_transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        original_image = self.original_transform(image)
        low_res_image = transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(original_image))
        low_res_image = transforms.ToTensor()(low_res_image)

        return low_res_image, original_image
