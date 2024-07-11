import torch
import torch.nn as nn

"""
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return x-out
"""

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
            )

        self.down1 = CBR(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2, output_padding=1)
        self.upconv3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = self.center_crop(u4, x4.shape)
        x4 = self.upconv4(torch.cat([x4, u4], dim=1))

        u3 = self.up3(x4)
        u3 = self.center_crop(u3, x3.shape)
        x3 = self.upconv3(torch.cat([x3, u3], dim=1))

        u2 = self.up2(x3)
        u2 = self.center_crop(u2, x2.shape)
        x2 = self.upconv2(torch.cat([x2, u2], dim=1))

        u1 = self.up1(x2)
        u1 = self.center_crop(u1, x1.shape)
        x1 = self.upconv1(torch.cat([x1, u1], dim=1))

        out = self.final(x1)
        return out

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[2]) // 2
        diff_x = (layer_width - target_size[3]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[2]), diff_x:(diff_x + target_size[3])]
