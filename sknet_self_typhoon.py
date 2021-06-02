import torch
from torch import nn
from torch.nn import init

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, nl = False, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.nl = nl
        if self.nl:
            # Nocal BLOCK
            self.theta = nn.Conv2d(features, features//2, 1) # N C W H -> N C/2 W H
            self.phi = nn.Conv2d(features, features//2, 1)
            self.gi = nn.Conv2d(features, features//2, 1)
            self.softmax_nl = nn.Softmax(dim=-1)
            self.W = nn.Sequential(
                    nn.Conv2d(features//2, features,
                            kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(self.features)
            )
        
        # SK BLOCK
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
    
    def spical_init(self):
        if self.nl:
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        # SKBlock
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        if self.nl:
            # Non Local Block
            N = fea_v.shape[0]
            g_x = self.gi(fea_v).view(N, self.features//2, -1) # N C H W -> N C/2 H*W
            g_x = g_x.permute(0,2,1) # N C/2 H*W -> N H*W C/2
            theta_x = self.theta(fea_v).view(N, self.features//2, -1)
            theta_x = theta_x.permute(0,2,1) # N H*W C/2
            phi_x = self.phi(fea_v).view(N, self.features//2, -1) # N C/2 H*W
            f = torch.matmul(theta_x, phi_x) # N H*W H*W
            f= self.softmax_nl(f)
            y = torch.matmul(f, g_x) # N H*W C/2
            y = y.permute(0, 2, 1).contiguous() # N C/2 H*W
            y = y.view(N, self.features//2, x.shape[2], x.shape[3]) # N C/2 H W
            w_y = self.W(y) # N C H W
            fea_v = w_y + fea_v
            # Non Local Block
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, nl = False, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        self.nl = nl
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, nl,stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def spical_init(self):
        if self.nl:
            self.feas[2].spical_init()

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SKNet(nn.Module):
    def __init__(self):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 25×25
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 8, 2, stride=1),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU()
        ) # 25×25
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2, nl=True),
            nn.ReLU()
        ) # 12×12
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2, nl=True),
            nn.ReLU()
        ) # 6×6
        self.pool = nn.AvgPool2d(6)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2),
            # nn.Softmax(dim=1)
        )
    
    def spical_init(self):
        self.stage_2[4].spical_init()
        self.stage_3[4].spical_init()

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


# 不使用自注意力模块的SKNet50
class SKNet50_Pure(nn.Module):
    def __init__(self):
        super(SKNet50_Pure, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 25×25
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 32, 2, stride=1),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 32, 2),
            nn.ReLU()
        ) # 25×25
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 32, 2, stride=2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 32, 2, nl=False),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 32, 2, nl=False),
            nn.ReLU()
        ) # 12×12
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 32, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2, nl=False),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2, nl=False),
            nn.ReLU()
            # SKUnit(1024, 1024, 32, 2, 32, 2),
            # nn.ReLU(),
            # SKUnit(1024, 1024, 32, 2, 32, 2, nl=True),
            # nn.ReLU()
        ) # 6×6
        self.stage_4 = nn.Sequential(
            SKUnit(1024, 2048, 32, 2, 32, 2, stride=1),
            nn.ReLU(),
            SKUnit(2048, 2048, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(2048, 2048, 32, 2, 32, 2),
            nn.ReLU()
        )# 6×6
        self.pool = nn.AvgPool2d(6)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2),
            # nn.Softmax(dim=1)
        )
    def spical_init(self):
        self.stage_2[2].spical_init()
        self.stage_2[6].spical_init()
        self.stage_3[6].spical_init()
        self.stage_3[10].spical_init()

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

# 使用自注意模块的SKNet50
class SKNet50(nn.Module):
    def __init__(self):
        super(SKNet50, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) # 25×25
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 32, 2, stride=1, nl=False),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 32, 2, nl=False),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 32, 2, nl=False),
            nn.ReLU()
        ) # 25×25
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 32, 2, stride=2, nl=False),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 32, 2, nl=False),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 32, 2, nl=False),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 32, 2, nl=False),
            nn.ReLU()
        ) # 12×12
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 32, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2, nl=True),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2, nl=True),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2, nl=True),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 32, 2, nl=True),
            nn.ReLU(),
            # SKUnit(1024, 1024, 32, 2, 32, 2),
            # nn.ReLU(),
            # SKUnit(1024, 1024, 32, 2, 32, 2, nl=True),
            # nn.ReLU()
        ) # 6×6
        self.stage_4 = nn.Sequential(
            SKUnit(1024, 2048, 32, 2, 32, 2, stride=1),
            nn.ReLU(),
            SKUnit(2048, 2048, 32, 2, 32, 2),
            nn.ReLU(),
            SKUnit(2048, 2048, 32, 2, 32, 2),
            nn.ReLU()
        )# 6×6
        self.pool = nn.AvgPool2d(6)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2),
            # nn.Softmax(dim=1)
        )
    def spical_init(self):
        # self.stage_1[0].spical_init()
        # self.stage_1[2].spical_init()
        # self.stage_1[4].spical_init()
        # self.stage_2[2].spical_init()
        # self.stage_2[4].spical_init()
        # self.stage_2[6].spical_init()
        self.stage_3[4].spical_init()
        self.stage_3[6].spical_init()
        self.stage_3[8].spical_init()
        self.stage_3[10].spical_init()

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        # mid_fea = fea
        fea = self.classifier(fea)
        return fea

if __name__=='__main__':
    x = torch.rand(8, 18, 25, 25)
    net = SKNet50_Pure()
    out = net(x)
    print(out.shape)
