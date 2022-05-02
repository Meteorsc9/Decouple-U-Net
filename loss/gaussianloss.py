import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianConv(nn.Module):
    def __init__(self):
        super(GaussianConv, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        kernel = cv2.getGaussianKernel(ksize=5, sigma=1) * cv2.getGaussianKernel(ksize=5, sigma=1).T
        kernel_gaussian = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(device)
        kernel_sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
        kernel_sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_gaussian = nn.Parameter(data=kernel_gaussian, requires_grad=False)
        self.weight_sobel_x = nn.Parameter(data=kernel_sobel_x, requires_grad=False)
        self.weight_sobel_y = nn.Parameter(data=kernel_sobel_y, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight_gaussian, padding=2)
        grad_x = F.conv2d(x, self.weight_sobel_x, padding=1)
        grad_y = F.conv2d(x, self.weight_sobel_y, padding=1)
        out = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return out


class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, inputs, targets):
        gaussian_conv = GaussianConv()
        inputs = gaussian_conv(inputs) / 1.41421356237
        targets = gaussian_conv(targets) / 1.41421356237
        l1_loss = nn.L1Loss()
        out = l1_loss(inputs, targets)

        return out
