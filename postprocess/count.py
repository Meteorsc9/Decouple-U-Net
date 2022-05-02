import cv2
import numpy as np


def cell_count(cell_image, nuclei_image):
    ret1, cell_image = cv2.threshold(cell_image, 128, 255,
                                     cv2.THRESH_BINARY)  # binarization of the output from segmentation networks
    ret2, nuclei_image = cv2.threshold(nuclei_image, 128, 255, cv2.THRESH_BINARY)
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(cell_image)  # calculate connected domains
    counts = ret
    for i in range(1, ret):
        if stats[i][4] <= 250:
            ''' connected domains with an area of less than 250 pixels was removed to exclude noise'''
            counts -= 1
            if np.mean(cell_image[labels == i]) == 255:
                cell_image[labels == i] = 0
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(cell_image)
    counts = ret
    nuc_counts, nuc_labels, nuc_stats, nuc_centroid = cv2.connectedComponentsWithStats(nuclei_image)
    for i in range(1, nuc_counts):
        if nuc_stats[i][4] <= 60:
            ''' connected domains (nuclei) with an area of less than 60 pixels was removed to exclude noise'''
            if np.mean(nuclei_image[nuc_labels == i]) == 255:
                nuclei_image[nuc_labels == i] = 0
    nuc_counts, nuc_labels, nuc_stats, nuc_centroid = cv2.connectedComponentsWithStats(nuclei_image)

    for i in range(1, ret):
        if stats[i][4] >= 3000:
            '''"large" cells with a quantile of 90%(3000px) or above in the first network segmentation results were subdivided using nuclei information'''
            unique_nuc = np.unique(nuc_labels[labels == i])
            if len(unique_nuc) >= 2:
                counts = counts + (len(unique_nuc) - 1)
    counts -= 1
    return counts


p_list = [21, 12, 2, 11, 18, 39, 14, 13]

