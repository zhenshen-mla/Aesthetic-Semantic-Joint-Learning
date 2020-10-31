import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as m
import math


class Retainlayer(nn.Module):
    def __init__(self, channel):
        super(Retainlayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // 8, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // 8, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x)
        w = self.conv1(w)
        w = self.sigmoid(self.conv2(self.relu(w)))
        out = x * w
        return out


class Sharelayer(nn.Module):
    def __init__(self, channel):
        super(Sharelayer, self).__init__()
        self.convA = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.convB = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.convA(x1)
        x2 = self.convB(x2)
        feas = torch.cat([torch.unsqueeze(x1, dim=1), torch.unsqueeze(x2, dim=1)], dim=1)
        fea_U = torch.sum(feas, dim=1)
        avg = self.avg_pool(fea_U)
        w_channel = self.sigmoid(self.shared_MLP(avg))
        out1 = x2 * w_channel
        out2 = x1 * w_channel
        return out1, out2


class AFA_layer(nn.Module):
    def __init__(self, channels=512):
        super(AFA_layer, self).__init__()

        self.retain1 = Retainlayer(channels)
        self.retain2 = Retainlayer(channels)
        self.share = Sharelayer(channels)
        self.alpha1 = nn.Parameter(torch.Tensor(1))
        self.alpha2 = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
        # self.bnA = nn.BatchNorm2d(channels)
        # self.bnB = nn.BatchNorm2d(channels)
        self.reluA = nn.ReLU()
        self.reluB = nn.ReLU()
        # print(self.alpha1, self.alpha2)

    def reset_parameters(self):
        a = torch.Tensor(1)
        b = torch.Tensor(1)
        a = a.fill_(0.01)
        self.alpha1.data = torch.nn.Parameter(a)
        b = b.fill_(0.01)
        self.alpha2.data = torch.nn.Parameter(b)

    def forward(self, x1, x2):

        r1 = self.retain1(x1)
        r2 = self.retain2(x2)
        s1, s2 = self.share(x1, x2)
        out1 = r1 + self.alpha1 * s1
        out2 = r2 + self.alpha2 * s2
        out1 = self.reluA(out1)
        out2 = self.reluB(out2)
        return out1, out2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, A_downsample=None, B_downsample=None):
        super(Bottleneck, self).__init__()
        self.planes = planes

        self.A_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.A_bn1 = nn.BatchNorm2d(planes)
        self.A_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.A_bn2 = nn.BatchNorm2d(planes)
        self.A_conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.A_bn3 = nn.BatchNorm2d(planes * 4)
        self.A_relu = nn.ReLU(inplace=True)

        self.B_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.B_bn1 = nn.BatchNorm2d(planes)
        self.B_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.B_bn2 = nn.BatchNorm2d(planes)
        self.B_conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.B_bn3 = nn.BatchNorm2d(planes * 4)
        self.B_relu = nn.ReLU(inplace=True)

        self.A_downsample = A_downsample
        self.B_downsample = B_downsample
        self.stride = stride

    def forward(self, x):
        x1, x2 = x[0], x[1]
        residual_1 = x1
        residual_2 = x2

        out1 = self.A_conv1(x1)
        out1 = self.A_bn1(out1)
        out1 = self.A_relu(out1)

        out1 = self.A_conv2(out1)
        out1 = self.A_bn2(out1)
        out1 = self.A_relu(out1)

        out2 = self.B_conv1(x2)
        out2 = self.B_bn1(out2)
        out2 = self.B_relu(out2)

        out2 = self.B_conv2(out2)
        out2 = self.B_bn2(out2)
        out2 = self.B_relu(out2)

        out1 = self.A_conv3(out1)
        out1 = self.A_bn3(out1)
        out2 = self.B_conv3(out2)
        out2 = self.B_bn3(out2)

        if self.A_downsample is not None:
            residual_1 = self.A_downsample(x1)
        if self.B_downsample is not None:
            residual_2 = self.B_downsample(x2)
        out1 += residual_1
        out2 += residual_2
        out1 = self.A_relu(out1)
        out2 = self.B_relu(out2)

        return out1, out2


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes1=2, num_classes2=12):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.A_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.A_bn1 = nn.BatchNorm2d(64)
        self.A_relu = nn.ReLU(inplace=True)
        self.A_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.B_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.B_bn1 = nn.BatchNorm2d(64)
        self.B_relu = nn.ReLU(inplace=True)
        self.B_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.afa1 = AFA_layer(channels=64 * 4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.afa2 = AFA_layer(channels=128 * 4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.afa3 = AFA_layer(channels=256 * 4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.afa4 = AFA_layer(channels=512 * 4)

        self.A_avgpool = nn.AvgPool2d(7, stride=1)
        self.A_fc = nn.Linear(512 * block.expansion, num_classes1)
        self.B_avgpool = nn.AvgPool2d(7, stride=1)
        self.B_fc = nn.Linear(512 * block.expansion, num_classes2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        A_downsample = None
        B_downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            A_downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            B_downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, A_downsample, B_downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.A_conv1(x)
        x1 = self.A_bn1(x1)
        x1 = self.A_relu(x1)
        x1 = self.A_maxpool(x1)
        x2 = self.B_conv1(x)
        x2 = self.B_bn1(x2)
        x2 = self.B_relu(x2)
        x2 = self.B_maxpool(x2)
        x = (x1, x2)
        x1, x2 = self.layer1(x)
        x1, x2 = self.afa1(x1, x2)
        x = (x1, x2)
        x1, x2 = self.layer2(x)
        x1, x2 = self.afa2(x1, x2)
        x = (x1, x2)
        x1, x2 = self.layer3(x)
        x1, x2 = self.afa3(x1, x2)
        x = (x1, x2)
        x1, x2 = self.layer4(x)
        x1, x2 = self.afa4(x1, x2)

        x1 = self.A_avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.A_fc(x1)
        x2 = self.B_avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.B_fc(x2)

        return x1, x2

    def get_10x_lr_params(self):
        for m in self.named_modules():
            if 'afa' in m[0]:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],
                                                                                                 nn.BatchNorm1d) or isinstance(
                        m[1], nn.Linear):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_1x_lr_params(self):
        for m in self.named_modules():
            if 'afa' not in m[0]:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],
                                                                                                 nn.BatchNorm1d) or isinstance(
                        m[1], nn.Linear):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def make_network():
    resnet50 = m.resnet50(pretrained=True)
    pretrained_dict = resnet50.state_dict()

    pre_dict_1 = {'A_' + k: v for k, v in pretrained_dict.items() if k[:5] != 'layer' and k[:2] != 'fc'}
    pre_dict_2 = {k[:9] + 'A_' + k[9:]: v for k, v in pretrained_dict.items() if k[:5] == 'layer' and k[9] != '.'}
    pre_dict_4 = {'B_' + k: v for k, v in pretrained_dict.items() if k[:5] != 'layer' and k[:2] != 'fc'}
    pre_dict_5 = {k[:9] + 'B_' + k[9:]: v for k, v in pretrained_dict.items() if k[:5] == 'layer' and k[9] != '.'}

    net = ResNet(Bottleneck, [3, 4, 6, 3])
    net_dict = net.state_dict()
    net_dict.update(pre_dict_1)
    net_dict.update(pre_dict_2)
    net_dict.update(pre_dict_4)
    net_dict.update(pre_dict_5)
    net.load_state_dict(net_dict)
    return net

net = make_network()
for k, v in net.named_modules():
    print(k)
