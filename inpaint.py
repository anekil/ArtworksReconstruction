#%%
import comet_ml
import torch
import os
from datasets import load_dataset
from pytorch_lightning.loggers import CometLogger
from torchvision.transforms import v2
from inpainting.InpaintingDataset import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

torch.cuda.empty_cache()

dataset = load_dataset("Artificio/WikiArt_Full", split="train")
df = dataset.to_pandas()
cluster_csv_path = "image_label_dict.csv"
cluster_df = pd.read_csv(cluster_csv_path)
cluster_df['idx'] = cluster_df['idx'].astype(int)
merged_df = df.merge(cluster_df, left_index=True, right_on="idx", how="left")
df = merged_df[['image', 'cluster']]

total_size = len(df)
new_size = total_size // 2
df = df.sample(n=new_size, random_state=42)
grouped = df.groupby('cluster')
min_samples_per_cluster = grouped.size().min()
balanced_df = grouped.apply(lambda x: x.sample(n=min_samples_per_cluster, random_state=42)).reset_index(drop=True)

train_val_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,
    stratify=balanced_df['cluster'],
    random_state=42
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    stratify=train_val_df['cluster'],
    random_state=42
)

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
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
val_dataset = InpaintingDataset(val_df, transform=transform, to_input=to_input)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10)
test_dataset = InpaintingDataset(test_df, transform=transform, to_input=to_input)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

import matplotlib.pyplot as plt

damaged_img, original_img, _ = train_dataset[0]
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
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                filename=f"gan-inpainting-{{epoch:02d}}-{{val_loss:.2f}}",
            ),
            EarlyStopping(monitor="val_loss", patience=10),
        ],
        default_root_dir="inpaint_checkpoints"
    )

torch.set_float32_matmul_precision('medium')
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader)

torch.save(model.generator.state_dict(), 'app/models/super_hiper_inpainter_state_dict.pth')
torch.save(model.generator, 'app/models/super_hiper_inpainter_full.pth')
