import torch
import torch.nn as nn
import torch.optim as optim

class ColorModel(nn.Module):
    def __init__(self):
        super(ColorModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, padding=1, stride=2, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, padding=1, stride=2, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, padding=1, stride=2, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=3)

        self.tpose1 = nn.ConvTranspose2d(64, 64, padding=1, stride=2, kernel_size=4)
        self.tpose2 = nn.ConvTranspose2d(96, 32, padding=1, stride=2, kernel_size=4)
        self.tpose3 = nn.ConvTranspose2d(48, 3, padding=1, stride=2, kernel_size=4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.relu(x1)
        #print(x1.size(),'x1')
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        #print(x2.size(),'x2')
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        #print(x3.size(),'x3')
        x4 = self.tpose1(x3)
        x4 = self.relu(x4)
        #print(x4.size(),'x4')
        x4 = torch.cat((x4, x2), dim=1)
        #print(x4.size(),'tx4')
        x5 = self.tpose2(x4)
        x5 = self.relu(x5)
        #print(x5.size(),'x5')
        x5 = torch.cat((x5, x1), dim=1)

        x6 = self.tpose3(x5)
        x6 = self.relu(x6)
        #print(x6.size(),'x6')
        return x6

model = ColorModel()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)