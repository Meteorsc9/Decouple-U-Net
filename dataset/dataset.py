import torch.utils.data as data
import os
import cv2
import numpy as np
import imgaug.augmenters as iaa


class ImageDataset(data.Dataset):
    def __init__(self, root, is_train, transform=None, target_transform=None):
        n = len(os.listdir(root)) // 3  # open the path of dataset and calculate the number of images
        train_and_test_list = [189, 190, 191, 192, 193, 194, 195, 196, 197, 108, 109, 110, 111, 112, 113, 114, 115,
                               116, 18, 19, 20, 21, 22, 23, 24, 25, 26, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                               162, 163, 164, 165, 166, 167, 168, 169, 170, 351, 352, 353, 354, 355, 356, 357, 358,
                               359, 126, 127, 128, 129, 130, 131, 132, 133, 134, 117, 118, 119, 120, 121, 122, 123,
                               124, 125, 216, 217, 218, 219, 220, 221, 222, 223, 224, 144, 145, 146, 147, 148, 149,
                               150, 151, 152, 252, 253, 254, 255, 256, 257, 258, 259, 260, 36, 37, 38, 39, 40, 41, 42,
                               43, 44, 333, 334, 335, 336, 337, 338, 339, 340, 341, 81, 82, 83, 84, 85, 86, 87, 88,
                               89, 0, 1, 2, 3, 4, 5, 6, 7, 8, 297, 298, 299, 300, 301, 302, 303, 304, 305, 270, 271,
                               272, 273, 274, 275, 276, 277, 278, 54, 55, 56, 57, 58, 59, 60, 61, 62, 234, 235, 236,
                               237, 238, 239, 240, 241, 242, 198, 199, 200, 201, 202, 203, 204, 205, 206, 288, 289,
                               290, 291, 292, 293, 294, 295, 296, 9, 10, 11, 12, 13, 14, 15, 16, 17, 207, 208, 209,
                               210, 211, 212, 213, 214, 215, 243, 244, 245, 246, 247, 248, 249, 250, 251, 135, 136,
                               137, 138, 139, 140, 141, 142, 143, 180, 181, 182, 183, 184, 185, 186, 187, 188, 261,
                               262, 263, 264, 265, 266, 267, 268, 269, 72, 73, 74, 75, 76, 77, 78, 79, 80, 342, 343,
                               344, 345, 346, 347, 348, 349, 350, 225, 226, 227, 228, 229, 230, 231, 232, 233, 315,
                               316, 317, 318, 319, 320, 321, 322, 323, 279, 280, 281, 282, 283, 284, 285, 286, 287,
                               171, 172, 173, 174, 175, 176, 177, 178, 179, 306, 307, 308, 309, 310, 311, 312, 313,
                               314, 153, 154, 155, 156, 157, 158, 159, 160, 161, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                               324, 325, 326, 327, 328, 329, 330, 331, 332, 45, 46, 47, 48, 49, 50, 51, 52, 53, 27,
                               28, 29, 30, 31, 32, 33, 34, 35, 90, 91, 92, 93, 94, 95, 96, 97, 98]

        imgs = []
        train_list = train_and_test_list[72:]
        test_list = train_and_test_list[:72]
        if is_train:
            cv_list = train_list
            n = int(n * 4 / 5)
        else:
            cv_list = test_list
            n = int(n / 5)
        for i in range(n):
            img = os.path.join(root, "%d.png" % cv_list[i])
            mask = os.path.join(root, "%d_mask.png" % cv_list[i])
            contour = os.path.join(root, "%d_contour.png" % cv_list[i])
            imgs.append([img, mask, contour])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train

    def __getitem__(self, index):
        x_path, mask_path, contour_path = self.imgs[index]
        img_x = cv2.imread(x_path)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_mask = cv2.imread(mask_path, 0)
        img_contour = cv2.imread(contour_path, 0)
        img_stack = np.zeros((256, 256, 5),
                             dtype='uint8')  # stack the images to ensure they are processed the same way during data augmentation
        if self.is_train:
            '''data augmentation'''
            seq = iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Rot90((0, 3), keep_size=False)])
            img_mask = np.expand_dims(img_mask, 2)
            img_contour = np.expand_dims(img_contour, 2)
            img_stack[:, :, 0:3] = img_x
            img_stack[:, :, 3:4] = img_mask
            img_stack[:, :, 4:] = img_contour
            img_stack = seq(image=img_stack)
            img_x = img_stack[:, :, 0:3]
            img_mask = img_stack[:, :, 3:4]
            img_contour = img_stack[:, :, 4:]
            img_x = np.ascontiguousarray(img_x)
            img_mask = np.ascontiguousarray(img_mask)
            img_contour = np.ascontiguousarray(img_contour)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
            img_contour = self.target_transform(img_contour)

        return img_x, img_mask, img_contour

    def __len__(self):
        return len(self.imgs)


'''dataloader for images not cropped to 256*256 sub-patches'''


class ImageDatasetNoTile(data.Dataset):
    def __init__(self, root, is_train, transform=None,
                 target_transform=None):
        n = len(os.listdir(root)) // 3  # open the path of dataset and calculate the number of images
        train_and_test_list = [21, 12, 2, 11, 18, 39, 14, 13, 24, 16, 28, 4, 37, 9, 0, 33, 30, 6, 26, 22, 32, 1, 23,
                               27, 15, 20, 29, 8, 38, 25, 35, 31, 19, 34, 17, 7, 36, 5, 3, 10]

        imgs = []
        train_list = train_and_test_list[8:]
        test_list = train_and_test_list[:8]
        if is_train:
            cv_list = train_list
            n = int(n * 4 / 5)

        else:
            cv_list = test_list
            n = int(n / 5)
        for i in range(n):
            img = os.path.join(root, "%d.png" % cv_list[i])
            mask = os.path.join(root, "%d_mask.png" % cv_list[i])
            contour = os.path.join(root, "%d_contour.png" % cv_list[i])
            imgs.append([img, mask, contour])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train

    def __getitem__(self, index):
        x_path, mask_path, contour_path = self.imgs[index]
        img_x = cv2.imread(x_path)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_mask = cv2.imread(mask_path, 0)
        img_contour = cv2.imread(contour_path, 0)
        img_stack = np.zeros((512, 512, 5), dtype='uint8')
        if self.is_train:
            seq = iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Rot90((0, 3), keep_size=False)])
            img_mask = np.expand_dims(img_mask, 2)
            img_contour = np.expand_dims(img_contour, 2)
            img_stack[:, :, 0:3] = img_x
            img_stack[:, :, 3:4] = img_mask
            img_stack[:, :, 4:] = img_contour
            img_stack = seq(image=img_stack)
            img_x = img_stack[:, :, 0:3]
            img_mask = img_stack[:, :, 3:4]
            img_contour = img_stack[:, :, 4:]
            img_x = np.ascontiguousarray(img_x)
            img_mask = np.ascontiguousarray(img_mask)
            img_contour = np.ascontiguousarray(img_contour)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
            img_contour = self.target_transform(img_contour)

        return img_x, img_mask, img_contour

    def __len__(self):
        return len(self.imgs)
