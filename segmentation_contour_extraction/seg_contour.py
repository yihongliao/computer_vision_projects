#!/usr/bin/env python
# coding: utf-8

# In[187]:


import numpy as np 
import matplotlib.pyplot as plt
import cv2
import math
from scipy import signal


# In[188]:


def otsu(img, num_itr, invert):
    mask = np.ones(img.shape, dtype=np.uint8)

    for i in range(num_itr):
        foreground = img[mask!=0]
        hist, bins = np.histogram(foreground, bins = np.unique(foreground), density=True)
        bins = bins[:-1]

        sigmaB2 = np.zeros(len(bins))
        i = 0
        for k in bins:
            omega0 = sum(hist[bins < k])
            omega1 = sum(hist[bins >= k])
            mu0 = bins[bins < k].dot(hist[bins < k]/omega0)
            mu1 = bins[bins >= k].dot(hist[bins >= k]/omega1)
            sigmaB2[i] = omega0 * omega1 * (mu1 - mu0)**2
            i = i + 1
        k_star = bins[np.argmax(sigmaB2)]
        
        if invert == 0:
            mask = img > k_star
        else:
            mask = img < k_star

    return mask

def bgr_segmentation(img, iterations, inverts):
    mask = np.ones(img[:, :, 0].shape, dtype=np.uint8)
    for i in range(3):
#         i = 2
        mask_tmp = otsu(img[:, :, i], iterations[i], inverts[i])
        mask = mask*mask_tmp
        print(i)
    return mask

def texture_segmentation(img, iterations, inverts, Ns):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.ones(img_gray.shape, dtype=np.uint8)
    for i, N in enumerate(Ns):
        img_tmp = get_feature_map(img_gray, N)
        mask_tmp = otsu(img_tmp, iterations[i], inverts[i])
        mask = mask*mask_tmp
        print(i)
    return mask

def get_feature_map(img, N):
    img = img.astype(float)
    sum_kernel = np.ones((N, N))
    window_sum = signal.convolve2d(img, sum_kernel, boundary='symm', mode='same')
    window2_sum = signal.convolve2d(img**2, sum_kernel, boundary='symm', mode='same')
    mean = window_sum/(N**2)
    mean2 = window2_sum/(N**2)
    var = mean2 - mean**2
    return var.astype(int)

def find_contour(img, k_size, dil_itr, ero_itr):
    
    kernel = np.ones((k_size, k_size), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=dil_itr)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=ero_itr)
    
    sum_kernel = np.ones((3, 3))
    window_sum = signal.convolve2d(img_erosion, sum_kernel, boundary='symm', mode='same')
    sum_mask = window_sum < 9
    
    return sum_mask*img_erosion
    
    
        


# In[189]:


if __name__ == '__main__':
    path = r'C:\Users\yhosc\Desktop\ECE661\HW6\HW6-Images\\'
    outputPath = r'C:\Users\yhosc\Desktop\ECE661\HW6\\'
    img_cat = cv2.imread(path+'cat.jpg')
    
    mask_bgr = bgr_segmentation(img_cat, [1, 1, 1], [1, 1, 0])
    img_cat_bgr = cv2.bitwise_and(img_cat, img_cat, mask = mask_bgr)
    
#     cv2.imwrite(outputPath+'testB.jpg', img_cat[:, :, 0])
#     cv2.imwrite(outputPath+'testG.jpg', img_cat[:, :, 1])
#     cv2.imwrite(outputPath+'testR.jpg', img_cat[:, :, 2])
    
    mask_texture = texture_segmentation(img_cat, [1, 1, 1], [0, 0, 0], [6, 8, 10])
    img_cat_texture = cv2.bitwise_and(img_cat, img_cat, mask = mask_texture)
    
    img_cat_contour_bgr = find_contour(mask_bgr, 5, 1, 1)
    img_cat_contour_texture = find_contour(mask_texture, 5, 1, 1)
        
    cv2.imwrite(outputPath+'img_cat_bgr.jpg', img_cat_bgr)
    cv2.imwrite(outputPath+'img_cat_texture.jpg', img_cat_texture)
    cv2.imwrite(outputPath+'img_cat_contour_bgr.jpg', img_cat_contour_bgr*255)
    cv2.imwrite(outputPath+'img_cat_contour_texture.jpg', img_cat_contour_texture*255)


# In[190]:


img_car = cv2.imread(path+'car.jpg')

mask_bgr = bgr_segmentation(img_car, [1, 1, 1], [1, 0, 1])
img_car_bgr = cv2.bitwise_and(img_car, img_car, mask = mask_bgr)

#     cv2.imwrite(outputPath+'testB.jpg', img_car[:, :, 0])
#     cv2.imwrite(outputPath+'testG.jpg', img_car[:, :, 1])
#     cv2.imwrite(outputPath+'testR.jpg', img_car[:, :, 2])

mask_texture = texture_segmentation(img_car, [1, 1, 1], [0, 0, 0], [6, 8, 10])
img_car_texture = cv2.bitwise_and(img_car, img_car, mask = mask_texture)

img_car_contour_bgr = find_contour(mask_bgr, 5, 1, 1)
img_car_contour_texture = find_contour(mask_texture, 5, 1, 1)
    
cv2.imwrite(outputPath+'img_car_bgr.jpg', img_car_bgr)
cv2.imwrite(outputPath+'img_car_texture.jpg', img_car_texture)
cv2.imwrite(outputPath+'img_car_contour_bgr.jpg', img_car_contour_bgr*255)
cv2.imwrite(outputPath+'img_car_contour_texture.jpg', img_car_contour_texture*255)


# In[191]:


img_f16 = cv2.imread(path+'f16.jpg')

mask_bgr = bgr_segmentation(img_f16, [1, 1, 1], [0, 1, 1])
img_f16_bgr = cv2.bitwise_and(img_f16, img_f16, mask = mask_bgr)

#     cv2.imwrite(outputPath+'testB.jpg', img_f16[:, :, 0])
#     cv2.imwrite(outputPath+'testG.jpg', img_f16[:, :, 1])
#     cv2.imwrite(outputPath+'testR.jpg', img_f16[:, :, 2])

mask_texture = texture_segmentation(img_f16, [1, 1, 1], [0, 0, 0], [2, 3, 4])
img_f16_texture = cv2.bitwise_and(img_f16, img_f16, mask = mask_texture)

img_f16_contour_bgr = find_contour(mask_bgr, 5, 1, 1)
img_f16_contour_texture = find_contour(mask_texture, 5, 1, 1)
    
cv2.imwrite(outputPath+'img_f16_bgr.jpg', img_f16_bgr)
cv2.imwrite(outputPath+'img_f16_texture.jpg', img_f16_texture)
cv2.imwrite(outputPath+'img_f16_contour_bgr.jpg', img_f16_contour_bgr*255)
cv2.imwrite(outputPath+'img_f16_contour_texture.jpg', img_f16_contour_texture*255)


# In[192]:


img_dog = cv2.imread(path+'dog.jpg')

mask_bgr = bgr_segmentation(img_dog, [1, 0, 2], [0, 1, 0])
img_dog_bgr = cv2.bitwise_and(img_dog, img_dog, mask = mask_bgr)

#     cv2.imwrite(outputPath+'testB.jpg', img_dog[:, :, 0])
#     cv2.imwrite(outputPath+'testG.jpg', img_dog[:, :, 1])
#     cv2.imwrite(outputPath+'testR.jpg', img_dog[:, :, 2])

mask_texture = texture_segmentation(img_dog, [1, 1, 1], [0, 0, 0], [3, 4, 5])
img_dog_texture = cv2.bitwise_and(img_dog, img_dog, mask = mask_texture)

img_dog_contour_bgr = find_contour(mask_bgr, 5, 1, 1)
img_dog_contour_texture = find_contour(mask_texture, 5, 1, 1)
    
cv2.imwrite(outputPath+'img_dog_bgr.jpg', img_dog_bgr)
cv2.imwrite(outputPath+'img_dog_texture.jpg', img_dog_texture)
cv2.imwrite(outputPath+'img_dog_contour_bgr.jpg', img_dog_contour_bgr*255)
cv2.imwrite(outputPath+'img_dog_contour_texture.jpg', img_dog_contour_texture*255)

