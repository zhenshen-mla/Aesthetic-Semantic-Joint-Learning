import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as m
import math


# class SKConv(nn.Module):
#     def __init__(self, in_features):
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the radio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(SKConv, self).__init__()
#
#         self.fc_A = nn.Linear(in_features, round(in_features / 8))
#         self.fc_AA = nn.Linear(round(in_features / 8), in_features)
#         self.fc_AB = nn.Linear(round(in_features / 8), in_features)
#         self.fc_B = nn.Linear(in_features, round(in_features / 8))
#         self.fc_BA = nn.Linear(round(in_features / 8), in_features)
#         self.fc_BB = nn.Linear(round(in_features / 8), in_features)
#         self.softmax1 = nn.Softmax(dim=1)
#         self.softmax2 = nn.Softmax(dim=1)
#         self.bn_A = nn.BatchNorm2d(in_features)
#         self.bn_B = nn.BatchNorm2d(in_features)
#
#     def forward(self, x1, x2):
#         original_x1, original_x2 = x1.data, x2.data
#         feas = torch.cat([torch.unsqueeze(original_x1, dim=1), torch.unsqueeze(original_x2, dim=1)], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_A = self.fc_A(fea_s)
#         fea_B = self.fc_B(fea_s)
#
#         vectorAA = self.fc_AA(fea_A).unsqueeze_(dim=1)
#         vectorAB = self.fc_AB(fea_A).unsqueeze_(dim=1)
#         vectorBA = self.fc_BA(fea_B).unsqueeze_(dim=1)
#         vectorBB = self.fc_BB(fea_B).unsqueeze_(dim=1)
#
#         attention_vectorsA = torch.cat([vectorAA, vectorAB], dim=1)
#         attention_vectorsB = torch.cat([vectorBA, vectorBB], dim=1)
#
#         attention_vectorsA = self.softmax1(attention_vectorsA)
#         attention_vectorsB = self.softmax2(attention_vectorsB)
#
#         attention_vectorsA = attention_vectorsA.unsqueeze(-1).unsqueeze(-1)
#         attention_vectorsB = attention_vectorsB.unsqueeze(-1).unsqueeze(-1)
#
#         fea_A = self.bn_A((feas * attention_vectorsA).sum(dim=1)) + x1 - original_x1
#         fea_B = self.bn_B((feas * attention_vectorsB).sum(dim=1)) + x2 - original_x2
#
#         return fea_A, fea_B
#
#
# class AFA_layer(nn.Module):
#     def __init__(self, channels=512):
#         super(AFA_layer, self).__init__()
#         self.sklayer = SKConv(channels*4)
#
#     def forward(self, x1, x2):
#         x1, x2 = self.sklayer(x1, x2)
#         return x1, x2
#


#                   chuang
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=8):
#         super(ChannelAttentionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel * 2, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#         self.alpha1 = nn.Parameter(torch.Tensor(1))
#         self.alpha2 = nn.Parameter(torch.Tensor(1))
#         self.reset_params()
#
#     def reset_params(self):
#         a = torch.Tensor(1)
#         b = torch.Tensor(1)
#         a = a.fill_(0.1)
#         self.alpha1.data = torch.nn.Parameter(a)
#         b = b.fill_(0.1)
#         self.alpha2.data = torch.nn.Parameter(b)
#
#     def forward(self, x1, x2):  # batch*channel*H*W
#
#         w = torch.cat((self.avg_pool(x1), self.avg_pool(x2)), dim=1)  # batch*(2*channel)*1*1
#         w_AB = self.sigmoid(self.shared_MLP(w))  # batch*channel*1*1
#         w_BA = self.sigmoid(self.shared_MLP(w))
#         #w_BA = self.sigmoid(self.B_shared_MLP(w))
#         out1 = self.alpha1 * w_AB * x2 + x1
#         out2 = self.alpha2 * w_BA * x1 + x2
#         return out1, out2
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.A_conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)  # kernel_size=7 padding=3
#         self.B_conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.sigmoid = nn.Sigmoid()
#         self.alpha1 = nn.Parameter(torch.Tensor(1))
#         self.alpha2 = nn.Parameter(torch.Tensor(1))
#         self.reset_params()
#
#     def reset_params(self):
#         a = torch.Tensor(1)
#         b = torch.Tensor(1)
#         a = a.fill_(0.1)
#         self.alpha1.data = torch.nn.Parameter(a)
#         b = b.fill_(0.1)
#         self.alpha2.data = torch.nn.Parameter(b)
#
#     def forward(self, x1, x2):
#         avgout1 = torch.mean(x1, dim=1, keepdim=True)
#         avgout2 = torch.mean(x2, dim=1, keepdim=True)
#         w = torch.cat([avgout1, avgout2], dim=1)
#         w_AB = self.sigmoid(self.A_conv2d(w))
#         #w_AB = w_AB.resize(32, -1, 56, 56)
#         w_BA = self.sigmoid(self.A_conv2d(w))
#
#       #  w_BA = self.sigmoid(self.B_conv2d(w))
#         #print(w_AB.size(),x2.size())
#         out1 = self.alpha1 * w_AB * x2 + x1
#         out2 = self.alpha2 * w_BA * x1 + x2
#         return out1, out2
#
#
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
#
#     def forward(self, x1, x2):
#         x1, x2 = self.channel_attention(x1, x2)
#         x1, x2 = self.spatial_attention(x1, x2)
#         return x1, x2


#                  bing
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=8):
#         super(ChannelAttentionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.shared_MLP_A = nn.Sequential(
#             nn.Conv2d(channel * 2, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.shared_MLP_B = nn.Sequential(
#             nn.Conv2d(channel * 2, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x1, x2):  # batch*channel*H*W
#
#         w = torch.cat((self.avg_pool(x1), self.avg_pool(x2)), dim=1)  # batch*(2*channel)*1*1
#         wA = self.sigmoid(self.shared_MLP_A(w))  # batch*channel*1*1
#         wB = self.sigmoid(self.shared_MLP_B(w))  # batch*channel*1*1
#         return wA, wB
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d_A = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)  # kernel_size=7 padding=3
#         self.conv2d_B = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x1, x2):
#         avgout1 = torch.mean(x1, dim=1, keepdim=True)
#         avgout2 = torch.mean(x2, dim=1, keepdim=True)
#         w = torch.cat([avgout1, avgout2], dim=1)
#         wA = self.sigmoid(self.relu(self.conv2d_A(w)))
#         wB = self.sigmoid(self.relu(self.conv2d_B(w)))
#         return wA, wB
#
#
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
#         self.alpha = nn.Parameter(torch.Tensor(1))
#         self.beta = nn.Parameter(torch.Tensor(1))
#         self.reset_params()
#
#     def reset_params(self):
#         # cam 0.1~1.5  sam 0.01~0.3
#         alpha = torch.Tensor(1)
#         beta = torch.Tensor(1)
#         alpha = alpha.fill_(0.5)
#         self.alpha.data = torch.nn.Parameter(alpha)
#         beta = beta.fill_(0.09)
#         self.beta.data = torch.nn.Parameter(beta)
#
#     def forward(self, x1, x2):
#         x1_data, x2_data = x1.data,  x2.data
#         c1, c2 = self.channel_attention(x1_data, x2_data)
#         s1, s2 = self.spatial_attention(x1_data, x2_data)
#         share2A = self.alpha * c1 * x2_data + self.beta * s1 * x2_data
#         share2B = self.alpha * c2 * x1_data + self.beta * s2 * x1_data
#         return x1+share2A, x2+share2B
#
#     # def forward(self, x1, x2):
#     #     c1, c2 = self.channel_attention(x1, x2)
#     #     s1, s2 = self.spatial_attention(x1, x2)
#     #     share2A = self.alpha * c1 * x2 + self.beta * s1 * x2
#     #     share2B = self.alpha * c2 * x1 + self.beta * s2 * x1
#     #     return x1+share2A, x2+share2B


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
        #
        # if self.planes == 512:
        #     self.afa = AFA_layer(self.planes)

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

        out1 = self.A_conv3(out1)
        out1 = self.A_bn3(out1)

        out2 = self.B_conv1(x2)
        out2 = self.B_bn1(out2)
        out2 = self.B_relu(out2)

        out2 = self.B_conv2(out2)
        out2 = self.B_bn2(out2)
        out2 = self.B_relu(out2)

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
        #
        # if self.planes == 512:
        #     out1, out2 = self.afa(out1, out2)

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

        self.cbam1 = CBAM(channel=64 * 4)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.cbam2 = CBAM(channel=128 * 4)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.cbam3 = CBAM(channel=256 * 4)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.cbam4 = CBAM(channel=512 * 4)

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
        x1, x2 = self.cbam1(x1, x2)
        x = (x1, x2)
        x1, x2 = self.layer2(x)
        x1, x2 = self.cbam2(x1, x2)
        x = (x1, x2)
        x1, x2 = self.layer3(x)
        x1, x2 = self.cbam3(x1, x2)
        x = (x1, x2)
        x1, x2 = self.layer4(x)
        x1, x2 = self.cbam4(x1, x2)

        x1 = self.A_avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.A_fc(x1)
        x2 = self.B_avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.B_fc(x2)

        return x1, x2

    def get_10x_lr_params(self):
        for m in self.named_modules():
            if 'cbam' in m[0]:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],
                                                                                                 nn.BatchNorm1d) or isinstance(
                        m[1], nn.Linear):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_1x_lr_params(self):
        for m in self.named_modules():
            if 'cbam' not in m[0]:
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


net = ResNet(Bottleneck, [3, 4, 6, 3])
# net = m.resnet50()
for k, v in net.named_modules():
    print(k)