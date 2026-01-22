import torch

def gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return torch.nn.GroupNorm(g, c)
    return torch.nn.GroupNorm(1, c)

class Autoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # 1x28x28 -> 32x14x14
        self.gn1 = gn(32)
        self.relu1 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x14x14 -> 64x7x7
        self.gn2 = gn(64)
        self.relu2 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.flatten = torch.nn.Flatten()                     # 64x7x7 -> 3136
        self.linear = torch.nn.Linear(64 * 7 * 7, latent_dim) # 3136 -> latent dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        self.linear = torch.nn.Linear(latent_dim, 64 * 7 * 7)                 #latent dim -> 64x7x7
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.gn1 = gn(32)
        self.relu1 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.deconv2 = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.output_activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)             # latent dim -> 64x7x7
        x = x.view(x.size(0), 64, 7, 7)       # reshape to 64x7x7
        x = self.up(x)                  # 64x7x7 -> 64x14x14
        x = self.deconv1(x)             # 64x14x14 -> 32x14x14
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.up(x)                  # 32x14x14 -> 32x28x28
        x = self.deconv2(x)             # 32x28x28 -> 1x28x28
        x = self.output_activation(x)
        return x
    

def build_autoencoder(latent_dim=256):
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    autoencoder = Autoencoder(encoder, decoder)
    return autoencoder