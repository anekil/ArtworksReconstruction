import os

import comet_ml
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CometLogger
from torchvision.transforms import v2

from superresolution.SuperResolutionDataset import SuperResolutionDataset
from superresolution.SuperResolutionTrainer import SuperResolutionLightningModule

torch.set_float32_matmul_precision('medium')
#%%
config = {
    "comet_api_key": "lliw2sljNWoBqtmqb9KztyYsG",
    "project_name": "supperresolution-test-resnet50",
    #"model_save_path": "./models",
    "batch_size": 8,
    "learning_rate": 1e-4
}

transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
])


folder_path = "/WikiArt_000/"
image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]
filtered_paths = []

for path in image_paths:
    with Image.open(path) as img:
        size = img.size
        if size[0] > 448 and size[1] > 448:
            filtered_paths.append(path)


dataset = SuperResolutionDataset(filtered_paths, transform=transform)

plt.imshow(dataset[0][0].permute(1, 2, 0))
plt.title(dataset[0][0].shape)
plt.show()
plt.imshow(dataset[0][1].permute(1, 2, 0))
plt.title(dataset[0][1].shape)
plt.show()
#%%
comet_logger = CometLogger(api_key=config["comet_api_key"], project_name=config["project_name"])
model = SuperResolutionLightningModule(config, dataset)
trainer = pl.Trainer(
        # fast_dev_run=True,
        logger=comet_logger,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                save_top_k=2,
                mode="min",
                filename=f"super-resolution-{{epoch:02d}}-{{val_loss:.2f}}",
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                mode='min'
            ),
            pl.callbacks.ModelSummary(max_depth=1)
        ],
    )
torch.set_float32_matmul_precision('medium')
trainer.fit(model)
trainer.test(model)
torch.save(model.model.state_dict(), 'super_hiper_resolutioner.pth')