import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNF(nn.Module):
    def __init__(self):
        super(CNNF, self).__init__()

        self.conv = torch.nn.Sequential(
            nn.Conv2d(10, 64, 1, 1, 0),  
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.ReLU(inplace=True),
            # nn.GroupNorm(8, 8)
            nn.BatchNorm2d(32)
        )
        
        self.classifier = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(32*1*1, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(64, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.numFeatures(x))  
        x = self.classifier(x)

        return x

    def numFeatures(self, x):
        size = x.size()[1:]  
        num = 1
        for s in size:
            num *= s
        return num

    def init_weights(self):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
