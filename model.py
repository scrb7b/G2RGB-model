import torch
import torch.nn as nn
import torch.optim as optim


def concatconv(x1, x2, f1, f2):
    conv_reduce = nn.Conv2d(f1, f2, kernel_size=1)
    x = torch.cat(x1, x2)
    x = conv_reduce(x)
    return x

class ColorModel(nn.Module):

    def __init__(self):
        super(ColorModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, padding=1, stride=2, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, padding=1, stride=2, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, padding=1, stride=2, kernel_size=3)

        self.conv_4 = nn.Conv2d(256, 512, kernel_size=1)

        self.tpose1 = nn.ConvTranspose2d(512, 256, padding=1, stride=2, kernel_size=4)
        self.tpose2 = nn.ConvTranspose2d(128, 64, padding=1, stride=2, kernel_size=4)
        self.tpose3 = nn.ConvTranspose2d(64, 3, padding=1, stride=2, kernel_size=4)

        self.relu = nn.ReLU()

        self.conv_reduce1 = nn.Conv2d(768, 512, kernel_size=1)
        self.conv_reduce2 = nn.Conv2d(384, 128, kernel_size=1)
        self.conv_reduce3 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)

        x4 = self.conv_4(x3)

        x4 = torch.cat((x4, x3), dim=1)
        x4 = self.conv_reduce1(x4)

        x5 = self.tpose1(x4)
        x5 = self.relu(x5)

        x5 = torch.cat((x5, x2), dim=1)
        x5 = self.conv_reduce2(x5)

        x6 = self.tpose2(x5)
        x6 = self.relu(x6)

        x6 = torch.cat((x6, x1), dim=1)
        x6 = self.conv_reduce3(x6)

        x7 = self.tpose3(x6)
        x7 = self.relu(x7)

        return x7

model = ColorModel()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)