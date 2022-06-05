import torch
import torch.nn as nn

class Mnist_MIL_Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.linear = nn.Sequential(
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.linear(x)
        return x