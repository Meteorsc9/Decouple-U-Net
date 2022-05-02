import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers import Resnet_Bottleneck, _AtrousSpatialPyramidPoolingModule


class DeeplabV3Plus(nn.Module):
    """
    Deeplabv3+
    :param in_ch:int, the channel number of input (for RGB images is 3)
    :param out_ch:int, the channel number of output
    """

    def __init__(self, in_ch, out_ch):
        super(DeeplabV3Plus, self).__init__()
        self.inplanes = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), )
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 23, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256,
                                                       output_stride=8)
        self.bot_aspp = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True),
                                      )
        self.bot_fine = nn.Sequential(nn.Conv2d(128, 48, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(48),
                                      nn.ReLU(inplace=True), )
        self.fusion = nn.Sequential(nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, out_ch, kernel_size=1, bias=True), )

    def _make_layer(
            self,
            planes: int,
            blocks: int,
            stride: int = 1,
            downsample=None):
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(
            Resnet_Bottleneck(
                self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(
                Resnet_Bottleneck(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        low_level = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        aspp = self.aspp(x)
        aspp = self.bot_aspp(aspp)
        aspp = F.interpolate(aspp, scale_factor=4, mode='bilinear', align_corners=True)
        low_level = self.bot_fine(low_level)
        fusion = torch.cat([aspp, low_level], dim=1)
        fusion = self.fusion(fusion)
        fusion = F.interpolate(fusion, scale_factor=2, mode='bilinear', align_corners=True)
        fusion = F.sigmoid(fusion)

        return fusion

    def init_weight(self):
        """weight initialization of network"""
        for parameter in self.modules():
            if isinstance(parameter, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(parameter.weight.data, a=0, mode='fan_in')
            elif isinstance(parameter, nn.Linear):
                nn.init.kaiming_normal_(parameter.weight.data, a=0, mode='fan_in')
            elif isinstance(parameter, nn.BatchNorm2d):
                nn.init.normal_(parameter.weight.data, 1.0, 0.02)
                nn.init.constant_(parameter.bias.data, 0.0)
