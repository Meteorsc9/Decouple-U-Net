import torch
import torch.nn as nn
import torch.nn.functional as F
from network.layers import DoubleConvNoRes


class VanillaUNet(nn.Module):
    """U-Net
    :param in_ch:int, the channel number of input (for RGB images is 3)
    :param out_ch:int, the channel number of output
    """

    def __init__(self, in_ch, out_ch):
        super(VanillaUNet, self).__init__()
        self.conv1 = DoubleConvNoRes(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConvNoRes(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConvNoRes(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConvNoRes(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConvNoRes(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConvNoRes(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConvNoRes(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConvNoRes(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConvNoRes(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # decoder
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)  # skip connection
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)
        return out

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


class GridAttentionBlock(nn.Module):
    """the attention gate in Attention U-Net, adapted from https://github.com/ozan-oktay/Attention-Gated-Networks"""

    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=2):
        super(GridAttentionBlock, self).__init__()

        # assert dimension in [2, 3]
        # assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap

        self.sub_sample_factor = sub_sample_factor

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        self.upsample_mode = 'bilinear'

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1,
                           bias=True)

        # Initialise weights

        # Define the operation

        self.operation_function = self._concatenation

    def forward(self, x, g):
        """
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        """

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y


class AttentionUNet(nn.Module):
    """
    Attention U-Net
    :param in_ch:int, the channel number of input (for RGB images is 3)
    :param out_ch:int, the channel number of output
    """

    def __init__(self, in_ch, out_ch):
        super(AttentionUNet, self).__init__()
        # encoder
        self.conv1 = DoubleConvNoRes(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConvNoRes(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConvNoRes(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConvNoRes(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConvNoRes(512, 1024)
        self.gate1 = GridAttentionBlock(in_channels=512, gating_channels=1024)
        # decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConvNoRes(1024, 512)
        self.gate2 = GridAttentionBlock(in_channels=256, gating_channels=512)  # attention gate
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConvNoRes(512, 256)
        self.gate3 = GridAttentionBlock(in_channels=128, gating_channels=256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConvNoRes(256, 128)
        self.gate4 = GridAttentionBlock(in_channels=64, gating_channels=128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConvNoRes(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        g1 = self.gate1(c4, c5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, g1], dim=1)
        c6 = self.conv6(merge6)
        g2 = self.gate2(c3, c6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, g2], dim=1)
        c7 = self.conv7(merge7)
        g3 = self.gate3(c2, c7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, g3], dim=1)
        c8 = self.conv8(merge8)
        g4 = self.gate4(c1, c8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, g4], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)
        return out

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


class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp_origin, self).__init__()
        self.conv = DoubleConvNoRes(in_size + (n_concat - 2) * out_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UNet_2Plus(nn.Module):
    """
    U-Net++
    :param in_ch:int, the channel number of input (for RGB images is 3)
    :param out_ch:int, the channel number of output
    adapted from https://github.com/ZJUGiveLab/UNet-Version
    """

    def __init__(self, in_ch, out_ch):
        super(UNet_2Plus, self).__init__()
        self.in_ch = in_ch

        filters = [64, 128, 256, 512, 1024]

        # encoder
        self.conv00 = DoubleConvNoRes(self.in_ch, filters[0])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = DoubleConvNoRes(filters[0], filters[1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = DoubleConvNoRes(filters[1], filters[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = DoubleConvNoRes(filters[2], filters[3])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = DoubleConvNoRes(filters[3], filters[4])

        # decoder
        self.up_concat01 = unetUp_origin(filters[1], filters[0])
        self.up_concat11 = unetUp_origin(filters[2], filters[1])
        self.up_concat21 = unetUp_origin(filters[3], filters[2])
        self.up_concat31 = unetUp_origin(filters[4], filters[3])

        self.up_concat02 = unetUp_origin(filters[1], filters[0], 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], out_ch, 1)
        self.final_2 = nn.Conv2d(filters[0], out_ch, 1)
        self.final_3 = nn.Conv2d(filters[0], out_ch, 1)
        self.final_4 = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4  # deep supervision

        return F.sigmoid(final)

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
