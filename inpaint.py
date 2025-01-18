#%%
import comet_ml
import numpy as np
import torch
import os

from comet_ml import Optimizer
from pytorch_lightning.loggers import CometLogger
from torchvision.transforms import v2

torch.cuda.empty_cache()
import pandas as pd
from inpainting.InpaintingDataset import *
from torch.utils.data import DataLoader

df = pd.read_parquet('local_wikiart.parquet', columns=['title', 'artist', 'date', 'genre', 'style', 'image'])
cluster_csv_path = "image_label_dict.csv"
cluster_df = pd.read_csv(cluster_csv_path)
cluster_df['idx'] = cluster_df['idx'].astype(int)
merged_df = df.merge(cluster_df, left_index=True, right_on="idx", how="left")
df = merged_df[['image', 'cluster']].head(8000)

total_len = len(df)
train_len = int(0.8 * total_len)
val_len = int(0.5 * (total_len - train_len))
train_df = df[:train_len].copy(deep=True)
val_df = df[train_len:train_len+val_len].copy(deep=True)
test_df = df[train_len+val_len:].copy(deep=True)

transform = v2.Compose([
    v2.RandomCrop((224, 224)),
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
])

to_input = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
])

train_dataset = InpaintingDataset(train_df, transform=transform, to_input=to_input)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10)
val_dataset = InpaintingDataset(val_df, transform=transform, to_input=to_input)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=10)
test_dataset = InpaintingDataset(test_df, transform=transform, to_input=to_input)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=10)

import matplotlib.pyplot as plt

damaged_img, original_img = train_dataset[0]
print(damaged_img.shape)
mask = damaged_img[3, :, :]
damaged_img = damaged_img[:3, :, :]

damaged_img = damaged_img.permute(1, 2, 0)
original_img = original_img.permute(1, 2, 0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("Original Image")
ax1.imshow(original_img)
ax2.set_title("Damaged Image")
ax2.imshow(damaged_img)
ax3.set_title("Mask")
ax3.imshow(mask)
plt.show()
#%%
from inpainting.InpaintingGAN import GANInpainting
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

comet_logger = CometLogger(
    api_key=os.environ.get("COMET_API_KEY"),
    project_name="image_inpainting"
)

model = GANInpainting()

trainer = Trainer(
        logger=comet_logger,
        max_epochs=30,
        accelerator="gpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                filename=f"gan-inpainting-{{epoch:02d}}-{{val_loss:.2f}}",
            ),
            EarlyStopping(monitor="val_loss", patience=20),
        ],
    )

torch.set_float32_matmul_precision('medium')
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader)

# config = {
#     "algorithm": "bayes",
#
#     "parameters": {
#         "lambda_gp": {"type": "float", "min": 0.1, "max": 50.0},  # Gradient penalty weight
#         "g_adv_weight": {"type": "float", "min": 0.1, "max": 5.0},  # Adversarial loss weight
#         "g_rec_weight": {"type": "float", "min": 0.1, "max": 5.0},  # Reconstruction loss weight
#         "g_perc_weight": {"type": "float", "min": 0.1, "max": 5.0},  # Perceptual loss weight
#         "g_tex_weight": {"type": "float", "min": 0.1, "max": 5.0},  # Texture loss weight
#         "g_div_weight": {"type": "float", "min": 0.1, "max": 5.0},  # Diversity loss weight
#     },
#
#     "spec": {
#         "metric": "val_loss",
#         "objective": "minimize",
#     },
# }

# optimizer = Optimizer(config)
#
# for experiment in optimizer.get_experiments(project_name="gan-inpainting"):
#     hyperparams = {
#         "lambda_gp": experiment.get_parameter("lambda_gp"),
#         "g_adv_weight": experiment.get_parameter("g_adv_weight"),
#         "g_rec_weight": experiment.get_parameter("g_rec_weight"),
#         "g_perc_weight": experiment.get_parameter("g_perc_weight"),
#         "g_tex_weight": experiment.get_parameter("g_tex_weight"),
#         "g_div_weight": experiment.get_parameter("g_div_weight"),
#     }
#
#     model = GANInpainting(**hyperparams)
#
#     experiment.set_name(f"gan-inpainting-{experiment.id}")
#     trainer = Trainer(
#         logger=comet_logger,
#         max_epochs=50,
#         accelerator="gpu",
#         devices=1,
#         callbacks=[
#             ModelCheckpoint(
#                 monitor="val_loss",
#                 save_top_k=1,
#                 mode="min",
#                 filename=f"gan-inpainting-{experiment.id}-{{epoch:02d}}-{{val_loss:.2f}}",
#             ),
#             EarlyStopping(monitor="val_loss", patience=20),
#         ],
#     )
#
#     torch.set_float32_matmul_precision('medium')
#     experiment.set_name(f"gan-inpainting-{experiment.id}")
#     trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
#     experiment.log_metric("val_loss", trainer.callback_metrics["val_loss"].item())
#     trainer.test(model, dataloaders=test_dataloader)
#     experiment.end()

# damaged_img, original_img = train_dataset[0]
# damaged_img = damaged_img.unsqueeze(0)
#
# model.eval()
# reconstructed = model(damaged_img)
# reconstructed_img = reconstructed.detach().squeeze()
# plt.imshow(reconstructed_img.permute(1, 2, 0))
# plt.show()