from torch import nn
from torchvision import models


class ResNet50Autoencoder(nn.Module):
    def __init__(self, latent_dim=512, freeze_percentage=0.9):
        super(ResNet50Autoencoder, self).__init__()
        
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        encoder_layers = list(resnet50.children())[:-1]

        total_layers = len(encoder_layers)
        layers_to_freeze = int(total_layers * freeze_percentage)

        self.encoder = nn.Sequential(*encoder_layers)
        
        for i, layer in enumerate(encoder_layers):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.encoder_latent = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(768, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.decoder_latent = nn.Sequential(
            nn.Linear(latent_dim, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2, padding=0),
            nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_latent(x)
        return x
    
    def decode(self, x):
        x = self.decoder_latent(x)
        x = x.view(x.size(0), 2048, 1, 1)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def get_latent(self, x):
        return self.encode(x)
