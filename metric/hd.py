import torch
from torchmetrics import Metric
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import cv2

''' code adapted from medpy: https://github.com/loli/medpy'''


class HD95(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_hd95", default=torch.tensor(0, dtype=float), dist_reduce_fx="sum")
        self.add_state("n_images", default=torch.tensor(0), dist_reduce_fx="sum")

    def __surface_distances(self, result, reference, voxelspacing=None):
        result = np.atleast_1d(result.astype(np.bool))
        reference = np.atleast_1d(reference.astype(np.bool))

        ''' compute average surface distance
            Note: scipys distance transform is calculated only inside the borders of the
            foreground objects, therefore the input has to be reversed'''
        dt = distance_transform_edt(~reference, sampling=voxelspacing)
        sds = dt[result]

        return sds

    def update(self, preds, targets):
        n, c, h, w = preds.shape
        ''' when batch_size=1'''
        if n == 1:
            preds = preds.squeeze().numpy() * 255
            targets = targets.squeeze().numpy() * 255
            preds = preds.astype(np.uint8)
            preds = cv2.Canny(preds, 128, 250, L2gradient=True)  # canny algorithm to calculate edge from network output
            preds = preds.astype("float")
            targets = targets.astype("float")

            value1 = self.__surface_distances(preds, targets)   # calculate unidirectional hausdorff distance
            value2 = self.__surface_distances(targets, preds)

            value = np.percentile(np.hstack((value1, value2)), 95)  # calculate 95% bidirectional hausdorff distance
            value = torch.tensor(value)
            self.sum_hd95 += value
            self.n_images += 1

        else:
            ''' when batch_size>1'''
            for i in range(0, n):
                pred = preds[i, :, :, :].squeeze().numpy() * 255
                target = targets[i, :, :, :].squeeze().numpy() * 255
                pred = pred.astype(np.uint8)
                pred = cv2.Canny(pred, 128, 250, L2gradient=True)  # canny algorithm to calculate edge from network output
                pred = pred.astype("float")
                target = target.astype("float")

                value1 = self.__surface_distances(pred, target)
                value2 = self.__surface_distances(target, pred)

                value = np.percentile(np.hstack((value1, value2)), 95)
                value = torch.tensor(value)
                self.sum_hd95 += value
            self.n_images += n

    def compute(self):
        return self.sum_hd95 / self.n_images
