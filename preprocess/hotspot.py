from skimage.segmentation import slic
from skimage import io
from skimage.measure import regionprops
import tifffile
import numpy as np
import cv2


def subcount(elem):
    return elem[1]


ndpi_path = ''  # the path of original ndpi files
dab_path = ''  # the path of DAB channel files after stain separation.
''' to perform stain separation and color normalization, please see https://github.com/abhishekvahadane/CodeRelease_ColorNormalization
    the standard stain vector we used is DAB (0.27,0.57,0.78), Hematoxylin (0.65,0.70,0.29)'''
save_path = ''  # the path of selected hot spot files
scale_factor = 8  # we used 40x and 5x thumbnails, so 40/5=8
img_original = tifffile.imread(files=ndpi_path)
img_dab = io.imread(dab_path).astype(np.uint8)
img_dab_gray = cv2.cvtColor(img_dab, cv2.COLOR_RGB2GRAY)  # grayscale
ret, img_dab_binarized = cv2.threshold(src=img_dab_gray, thresh=200, maxval=255, type=cv2.THRESH_BINARY)  # binarization
seg = slic(img_dab, n_segments=500, compactness=25, convert2lab=True,
           start_label=1)  # slic: n_segmemts controls the number of generated hot spots, and the bigger compactness you set, the more cubic superpixels you get
superpixel_list = []
region = regionprops(seg)  # generate bounding boxes
for single_region in region:
    superpixel_mean = np.mean(
        img_dab_binarized[single_region.bbox[0]:single_region.bbox[2],
        single_region.bbox[1]:single_region.bbox[3]])  # calculate the mean pixel value in single bounding box
    superpixel_list.append(
        [single_region.label, superpixel_mean, single_region.bbox[0], single_region.bbox[2], single_region.bbox[1],
         single_region.bbox[3]])

superpixel_list.sort(key=subcount)  # sort bounding boxes according to the proportion of black pixels
hot_spot_num = 25  # set the number of selected hot spots

for i in range(0, hot_spot_num):
    io.imsave(fname=save_path,
              arr=img_original[superpixel_list[i][2] * scale_factor:superpixel_list[i][3] * scale_factor,
                  superpixel_list[i][4] * scale_factor:superpixel_list[i][5] * scale_factor, :])
