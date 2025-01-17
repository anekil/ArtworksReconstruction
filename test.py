import comet_ml
import pandas as pd
import pytorch_lightning as pl
import torch
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
    "max_epochs": 50,
    "patience": 15,
    "batch_size": 16,
    "learning_rate": 1e-4
}

#%%
df = pd.read_parquet('local_wikiart.parquet', columns=['title', 'artist', 'date', 'genre', 'style', 'image']).head(8000)

#%%
transform = v2.Compose([
    v2.RandomCrop((224, 224)),
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

dataset = SuperResolutionDataset(df, transform=transform, upscale_factor=2)
comet_logger = CometLogger(api_key=config["comet_api_key"], project_name=config["project_name"])
model = SuperResolutionLightningModule(config, dataset)
trainer = pl.Trainer(logger=comet_logger, max_epochs=config["max_epochs"], log_every_n_steps=10)
trainer.fit(model)
trainer.test(model)