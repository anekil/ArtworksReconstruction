import io

import matplotlib.pyplot as plt
import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2


from superresolution.SuperResolutionDataset import SuperResolutionDataset
from superresolution.SuperResolutionTrainer import SuperResolutionLightningModule, ResNet50Autoencoder

device = 'cuda'

checkpoint_path = "superres_explicit_jigsaw_6396_model.ckpt"
model = SuperResolutionLightningModule.load_from_checkpoint(
    checkpoint_path,
    config=None,
    dataset=None
)

model = model.model.to(device)
model.eval()

df = pd.read_parquet('local_wikiart.parquet', columns=['image']).head(8000)
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

dataset = SuperResolutionDataset(df, transform=transform, upscale_factor=2)

for j in range(10):
    low_res_upsampled, image = dataset[j]
    input_tensor = low_res_upsampled.unsqueeze(0).to(device)

    for i in range(3):
        with torch.no_grad():
            output = model(input_tensor)
        input_tensor = output
        output_image = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        plt.imshow(output_image)
        plt.title(f'{i+1} iteration')
        plt.show()

    plt.imshow(image.cpu().permute(1, 2, 0).numpy())
    plt.title('original image')
    plt.show()