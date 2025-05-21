import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4, 
                stride=stride, 
                padding=1,
                bias=True, #in cycleGAN paper bias is often False when using InstanceNorm
                padding_mode="reflect",
            )),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        #spectral norm for the initial Conv2d layer
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
                bias=True 
            )),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        current_in_channels = features[0] 
        for feature in features[1:]:
            layers.append(
                Block(current_in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            current_in_channels = feature

        #spectral norm for the final Conv2d layer
        layers.append(
            spectral_norm(nn.Conv2d(
                current_in_channels, #use the updated in_channels from the loop
                1, #output 1 channel for PatchGAN
                kernel_size=4,
                stride=1, #stride 1 for the last conv layer
                padding=1,
                padding_mode="reflect",
                bias=True 
            ))
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        #return torch.sigmoid(self.model(x))
        return self.model(x)  #no sigmoid


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
