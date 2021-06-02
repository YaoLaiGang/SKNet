import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEUnit(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, stride=1, groups=32):
        super(SEUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, mid_features, 3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features),
            SELayer(out_features, 16)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SENet(nn.Module):
    def __init__(self, class_num):
        super(SENet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 32x32
        self.stage_1 = nn.Sequential(
            SEUnit(64, 256, mid_features=128),
            nn.ReLU(),
            SEUnit(256, 256),
            nn.ReLU(),
            SEUnit(256, 256),
            nn.ReLU()
        ) # 32x32
        self.stage_2 = nn.Sequential(
            SEUnit(256, 512, stride=2),
            nn.ReLU(),
            SEUnit(512, 512),
            nn.ReLU(),
            SEUnit(512, 512),
            nn.ReLU()
        ) # 16x16
        self.stage_3 = nn.Sequential(
            SEUnit(512, 1024, stride=2),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU()
        ) # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

class SENet50(nn.Module):
    def __init__(self):
        super(SENet50, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 25x25
        self.stage_1 = nn.Sequential(
            SEUnit(64, 256, mid_features=128),
            nn.ReLU(),
            SEUnit(256, 256),
            nn.ReLU(),
            SEUnit(256, 256),
            nn.ReLU()
        ) # 25x25
        self.stage_2 = nn.Sequential(
            SEUnit(256, 512, stride=2),
            nn.ReLU(),
            SEUnit(512, 512),
            nn.ReLU(),
            SEUnit(512, 512),
            nn.ReLU(),
            SEUnit(512, 512),
            nn.ReLU()
        ) # 12x12
        self.stage_3 = nn.Sequential(
            SEUnit(512, 1024, stride=2),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU(),
            SEUnit(1024, 1024),
            nn.ReLU()
        ) # 6x6
        self.stage_4 = nn.Sequential(
            SEUnit(1024, 2048),
            nn.ReLU(),
            SEUnit(2048, 2048),
            nn.ReLU(),
            SEUnit(2048, 2048),
            nn.ReLU()
        )# 6x6
        self.pool = nn.AvgPool2d(6)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea
    
if __name__=='__main__':
    x = torch.rand(8,18,25,25)
    net = SENet50()
    out = net(x)
    print(out.shape)