# -*- coding: utf-8 -*-
"""
data augmentation
"""

import os
import cv2
import numpy as np
import shutil
import concurrent.futures

def Crop(src_img):
    """
    crop to 2:1
    """
    # to same size
    h, w = src_img.shape
    new_img = np.zeros((886, 409))
    if h == 866:   
        tmp = (409 - w) // 2
        new_img[10:876, tmp:tmp+w] = src_img.copy()
    elif w == 399:
        tmp = (409 - w) // 2
        new_img[:, tmp:tmp+w] = src_img.copy()
    else:
        new_img = src_img.copy()
    
    return new_img

def worker(img, idx, src_path, dest_path):
    im_path = os.path.join(src_path, idx, img)
    im_mat = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    im_new = Crop(im_mat)
    # new_name = img.split('.')[0] + '_%d' % i + '.jpg'
    cv2.imwrite(os.path.join(dest_path, idx, img), im_new)

def dataset_aug(src_path, dest_path):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.mkdir(dest_path)

    for idx in os.listdir(src_path):
        # make dir for class

        if not os.path.isdir(os.path.join(src_path, idx)):
            continue
        os.mkdir(os.path.join(dest_path, idx))

        # aug
        im_list = os.listdir(os.path.join(src_path, idx))
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(worker, im_list, [idx]*len(im_list), \
                        [src_path]*len(im_list), [dest_path]*len(im_list))
        # for img in im_list:
        #     im_path = os.path.join(src_path, idx, img)
        #     im_mat = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        #     for i in range(1):
        #         im_new = Crop(im_mat)
        #         cv2.imwrite(os.path.join(dest_path, idx, img), im_new)

if __name__ == '__main__':
    src_path = './dataset/test/识别'
    dest_path = './dataset/test_aug/识别'
    dataset_aug(src_path, dest_path)
