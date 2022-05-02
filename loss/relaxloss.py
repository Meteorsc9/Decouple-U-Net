import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''code adapted from https://github.com/lxtGH/DecoupleSegNets'''


def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )


class RelaxLoss(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(RelaxLoss, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm

    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def onehot2label(self, target):

        label = torch.argmax(target[:, :-1, :, :], dim=1).long()
        label[target[:, -1, :, :]] = self.ignore_index
        return label

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """

        loss_matrix = (-1 / border_weights *
                       (target[:, :-1, :, :].float() *
                        class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                        customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                      (1. - mask.float())

        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target_label):
        # add ohem loss for the final stage

        label_shape = target_label.shape
        target = torch.zeros(label_shape[0], 2, label_shape[2], label_shape[3]).to('cuda:0')
        target = target.scatter_(1, target_label.to(torch.int64), 1)

        weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1
        loss = 0
        target_cpu = target.data.cpu().numpy()

        for i in range(0, inputs.shape[0]):
            class_weights = self.calculate_weights(target_cpu[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss
