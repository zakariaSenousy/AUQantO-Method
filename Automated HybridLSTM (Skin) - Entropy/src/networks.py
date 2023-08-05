import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BaseNetwork(nn.Module):
    def __init__(self, name, channels=1):
        super(BaseNetwork, self).__init__()
        self._name = name
        self._channels = channels

    def name(self):
        return self._name

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PatchWiseNetwork(BaseNetwork):
    def __init__(self, channels=1):
        super(PatchWiseNetwork, self).__init__('pw' + str(channels), channels)
        
        
        print('ResNet-18 Network')
        resnet = torchvision.models.resnet18(pretrained=True)

               
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 
        self.pool = nn.AvgPool2d(3,stride=1)
        self.classifier = nn.Sequential(nn.Linear(2048, 2))        
        
    def forward(self, x):
        x = self.features(x)
        
        ct = 0
        for child in self.features.children():
            ct += 1
            if ct < 8:
                for param in child.parameters():
                    param.requires_grad = False
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class ImageWiseNetwork(BaseNetwork):
    def __init__(self, channels=1):
        super(ImageWiseNetwork, self).__init__('iw' + str(channels), channels)
        
        self.lstm = nn.LSTM(input_size=2*4,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        
        self.body = nn.Sequential(
          nn.Linear(2*128, 128),
          nn.ReLU(),
          nn.Linear(128, 2)) 
        
        
        
        self.initialize_weights()

    def forward(self, x):       
        output, _ = self.lstm(x)
        output = output[:, -1]
       
        x = self.body(output)   
        x = F.log_softmax(x, dim=1)
              
        return x
