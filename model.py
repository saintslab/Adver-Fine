import torch
import torch.nn as nn
import torch.nn.functional as F
import detectors
import timm


def get_resnet18():
    '''
    Used for Cirfar10
    Load with:
    model = get_resnet18().to(device)
    '''
    return timm.create_model("resnet18_cifar10", pretrained=True)


class EightLayerConv(nn.Module):
    def __init__(self):
        super(EightLayerConv, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.relu6 = nn.ReLU()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.flatten(x)
        x = self.relu7(self.fc1(x))
        x = self.fc2(x)
        return x


    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    


    






    
