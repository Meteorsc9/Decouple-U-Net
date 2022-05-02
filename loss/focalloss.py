import torch
import torch.nn as nn
import torch.nn.functional as F

'''note: if you set adaptive_weight=True, 
   the alpha will be calculated according to the percentage of non-zero pixels in the GT semantic labels, 
   otherwise the alpha should be set manually. '''


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True, adaptive_weight=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.adaptive_weight = adaptive_weight

    def forward(self, inputs, targets):
        if self.adaptive_weight:
            target_t = targets.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
            pos_index = (target_t == 1)
            neg_index = (target_t == 0)
            pos_index = pos_index.data.cpu().numpy().astype(bool)
            neg_index = neg_index.data.cpu().numpy().astype(bool)
            pos_num = pos_index.sum()
            neg_num = neg_index.sum()
            sum_num = pos_num + neg_num
            self.alpha = 1.0 * neg_num / sum_num

        pt = (1 - inputs) * targets + inputs * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) *
                        (1 - targets)) * pt.pow(self.gamma)
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        focal_loss = bce_loss * focal_weight

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


class EdgeAttBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(EdgeAttBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input, target, edge):
        target_weight = torch.zeros_like(target)
        target_weight[edge[:, :, :, :] > 0.8] = 1.0
        loss = F.binary_cross_entropy(input, target, weight=target_weight)

        return loss


class EdgeAttBinaryFocalLoss(nn.Module):
    def __init__(self):
        super(EdgeAttBinaryFocalLoss, self).__init__()

    def forward(self, input, target, edge):
        target_weight = torch.zeros_like(target)
        target_weight[edge[:, :, :, :] > 0.8] = 1.0

        pt = (1 - input) * target + input * (1 - target)
        focal_weight = (0.5 * target + (1 - 0.5) *
                        (1 - target)) * pt.pow(2)
        loss = F.binary_cross_entropy(input, target, weight=target_weight)
        loss = loss * focal_weight
        return loss
