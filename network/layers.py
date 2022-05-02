import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConvNoRes(nn.Module):
    """cascade conv-bn-relu without residual connections"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConvNoRes, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.relu2(out2)
        return out2


class DoubleConv(nn.Module):
    """cascade conv-bn-relu with residual connections"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.identity_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            identity1 = x
        else:
            identity1 = self.identity_conv(x)
        out1 = self.conv1(x)
        out1 += identity1
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 += out1
        out2 = self.relu2(out2)
        return out2


class DoubleConvWithSE(nn.Module):
    """cascade conv-bn-relu with residual connections and channel attention module"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConvWithSE, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.se1 = SELayer(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.identity_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.se2 = SELayer(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            identity1 = x
        else:
            identity1 = self.identity_conv(x)
        out1 = self.conv1(x)
        out1 = self.se1(out1)
        out1 += identity1
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.se2(out2)
        out2 += out1
        out2 = self.relu2(out2)
        return out2


class DilatedDoubleConvWithSE(nn.Module):
    """cascade conv-bn-relu with residual connections, channel attention module and atrous conv"""
    def __init__(self, in_ch, out_ch, dilation):
        super(DilatedDoubleConvWithSE, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.se1 = SELayer(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.identity_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_ch),
        )
        self.se2 = SELayer(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            identity1 = x
        else:
            identity1 = self.identity_conv(x)
        out1 = self.conv1(x)
        out1 = self.se1(out1)
        out1 += identity1
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.se2(out2)
        out2 += out1
        out2 = self.relu2(out2)
        return out2


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.identity_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity_conv(x)
        out = self.conv(x)
        out += identity
        out = self.last_relu(out)
        return out


class SingleConvNoRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConvNoRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.last_relu(out)
        return out


class SingleConvWithSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConvWithSE, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SELayer(out_ch)
        self.identity_conv = nn.Conv2d(in_ch, out_ch, 1)
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.in_ch == self.out_ch:
            identity = x
        else:
            identity = self.identity_conv(x)
        out = self.conv(x)
        out = self.se(out)
        out += identity
        out = self.last_relu(out)
        return out


class SELayer(nn.Module):
    """channel attention module from SENet"""
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


class Resnet_Bottleneck(nn.Module):
    """the bottleneck in ResNet-101"""
    def __init__(self, inplanes, planes, expansion=4, stride=1, downsample=None):
        super(Resnet_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, size=x_size[2:], mode='bilinear', align_corners=True)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out
