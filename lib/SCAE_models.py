import torch.nn as nn
import torch.nn.functional as F

### Define the SCAE as per the paper, although imo, it is very basic
class StackedConvAutoencoder(nn.Module):
    def __init__(self):
        super(StackedConvAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)  # Output: 32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Output: 16x16
        
        # Decoder
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)  # Output: 16x16
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 32x32
        
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x, indices = self.pool(x)
        
        # Decoder
        x = self.unpool(x, indices)
        x = F.relu(self.conv2(x))
        return x


class ArtifactCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ArtifactCNN, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)  # Output: 32x32
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x16
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # Output: 16x16
        
        # Flatten layer for transitioning from conv layers to linear layers
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened features after conv and pooling layers
        self.feature_size = 32 * 16 * 16
        
        # First dense (linear) layer
        self.fc1 = nn.Linear(self.feature_size, 128)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.25)
        
        # Second dense (linear) layer for classification
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
