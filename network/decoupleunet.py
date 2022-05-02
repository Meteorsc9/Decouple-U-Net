import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers import Resnet_Bottleneck, DoubleConvWithSE, DilatedDoubleConvWithSE, _AtrousSpatialPyramidPoolingModule


class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane):
        """
        body generation module
        :param inplane:
        adapted from https://github.com/lxtGH/DecoupleSegNets
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class DecoupleSegNet(nn.Module):
    """
    Decouplesegnet
    :param in_ch:int, the channel number of input (for RGB images is 3)
    :param out_ch:int, the channel number of output
    """

    def __init__(self, in_ch, out_ch):
        super(DecoupleSegNet, self).__init__()
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

        if out_ch == 1:
            body_out_ch = 2
        elif out_ch > 1:
            body_out_ch = out_ch

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

        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        # body generation module
        self.squeeze_body_edge = SqueezeBodyEdge(256)

        # fusion different edge part
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        # DSN for seg body part
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, body_out_ch, kernel_size=1, bias=False)
        )

        # Final segmentation part
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_ch, kernel_size=1, bias=False))

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
        # print(self.inplanes)
        # print(planes)
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

    def forward(self, x, gts=None):

        x_size = x.size()
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        fine_size = x1.size()
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        xp = self.aspp(x4)

        aspp = self.bot_aspp(xp)

        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        dec0_fine = self.bot_fine(x1)

        seg_edge = self.edge_fusion(
            torch.cat([F.interpolate(seg_edge, size=fine_size[2:], mode='bilinear', align_corners=True), dec0_fine],
                      dim=1))
        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + F.interpolate(seg_body, size=fine_size[2:], mode='bilinear', align_corners=True)
        aspp = F.interpolate(aspp, size=fine_size[2:], mode='bilinear', align_corners=True)

        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = F.interpolate(seg_edge_out, size=x_size[2:], mode='bilinear', align_corners=True)
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = F.interpolate(seg_final, size=x_size[2:], mode='bilinear', align_corners=True)
        seg_final_out = torch.sigmoid(seg_final_out)

        seg_body_out = F.interpolate(self.dsn_seg_body(seg_body), size=x_size[2:], mode='bilinear', align_corners=True)

        return [seg_final_out, seg_body_out, seg_edge_out]

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


class DecoupleUNet(nn.Module):
    """
    Decoupleunet
    :param in_ch:int, the channel number of input (for RGB images is 3)
    :param out_ch:int, the channel number of output
    """

    def __init__(self, in_ch, out_ch):
        super(DecoupleUNet, self).__init__()
        # residual block
        self.conv1 = DoubleConvWithSE(in_ch, 64)  # cascade conv-bn-relu
        self.pool1 = nn.MaxPool2d(2)  # maxpool
        self.conv2 = DoubleConvWithSE(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConvWithSE(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DilatedDoubleConvWithSE(256, 512, dilation=2)  # we use atrous conv so maxpool is not needed
        self.conv5 = DilatedDoubleConvWithSE(512, 1024, dilation=4)

        if out_ch == 1:
            body_out_ch = 2  # the relax loss function requires channel>=2
        elif out_ch > 1:
            body_out_ch = out_ch

        # body output
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, body_out_ch, kernel_size=1, bias=False),
        )
        # body generation module
        self.squeeze1 = SqueezeBodyEdge(1024)
        self.squeeze2 = SqueezeBodyEdge(512)
        self.squeeze3 = SqueezeBodyEdge(256)
        self.squeeze4 = SqueezeBodyEdge(128)
        self.squeeze5 = SqueezeBodyEdge(64)

        # segmentation module
        self.edge1_up = nn.Conv2d(1024, 512, kernel_size=1)  # upsample
        self.body1_up = nn.Conv2d(1024, 512, kernel_size=1)

        self.concat_edge2 = DoubleConvWithSE(1024, 512)  # merge feature maps from different layers
        self.concat_body2 = DoubleConvWithSE(1024, 512)

        self.concat_edge2_up = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.concat_body2_up = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.concat_edge3 = DoubleConvWithSE(512, 256)
        self.concat_body3 = DoubleConvWithSE(512, 256)

        self.concat_edge3_up = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.concat_body3_up = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.concat_edge4 = DoubleConvWithSE(256, 128)
        self.concat_body4 = DoubleConvWithSE(256, 128)

        self.concat_edge4_up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.concat_body4_up = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.concat_edge5 = DoubleConvWithSE(128, 64)
        self.concat_body5 = DoubleConvWithSE(128, 64)

        # edge output
        self.seg_edge_out = nn.Conv2d(64, 1, kernel_size=1)
        # merge body and edge information
        self.final_seg = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=1),
        )

    def forward(self, x):
        x_size = x.size()
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        c5 = self.conv5(c4)

        body1, edge1 = self.squeeze1(c5)
        body2, edge2 = self.squeeze2(c4)
        body3, edge3 = self.squeeze3(c3)
        body4, edge4 = self.squeeze4(c2)
        body5, edge5 = self.squeeze5(c1)

        concat_edge2 = self.concat_edge2(torch.cat([self.edge1_up(edge1), edge2], dim=1))
        concat_body2 = self.concat_body2(torch.cat([self.body1_up(body1), body2], dim=1))

        concat_edge3 = self.concat_edge3(torch.cat([edge3, self.concat_edge2_up(concat_edge2)], dim=1))
        concat_body3 = self.concat_body3(torch.cat([body3, self.concat_body2_up(concat_body2)], dim=1))

        concat_edge4 = self.concat_edge4(torch.cat([edge4, self.concat_edge3_up(concat_edge3)], dim=1))
        concat_body4 = self.concat_body4(torch.cat([body4, self.concat_body3_up(concat_body3)], dim=1))

        concat_edge5 = self.concat_edge5(torch.cat([edge5, self.concat_edge4_up(concat_edge4)], dim=1))
        concat_body5 = self.concat_body5(torch.cat([body5, self.concat_body4_up(concat_body4)], dim=1))

        seg_body_out = F.interpolate(self.dsn_seg_body(body1), size=x_size[2:], mode='bilinear', align_corners=True)
        seg_edge_out = torch.sigmoid(self.seg_edge_out(concat_edge5))
        seg_final_out = torch.sigmoid(self.final_seg(concat_body5 + concat_edge5))

        return [seg_final_out, seg_body_out, seg_edge_out]

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
