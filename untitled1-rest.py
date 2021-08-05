# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:31:21 2021

@author: meslami
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage    
from skimage import measure
from skimage import io
from skimage.color import colorconv


image=io.imread('./image315-4.tif', as_gray=False)
image_prime=io.imread('./MESSIDOR/image315prime.tif', as_gray=False)

image=(255*image)/np.max(image)
image_prime=(255*image_prime)/np.max(image_prime)

# diff=abs(image_prime-image)
# diff=colorconv.rgb2gray(diff)
# diff=(255*diff)/np.max(diff)

#plt.imshow(diff)

diff=abs(image_prime-image)
plt.imshow(diff)
diff=colorconv.rgb2gray(diff)
diff=(255*diff)/np.max(diff)

#io.imsave('diff.png', diff)

#plt.figure()
#plt.imshow(diff)

# Find contours at a constant value of 0.8 
contours = measure.find_contours(diff, 50)

## Display the image and plot all contours found
#fig, ax = plt.subplots()
#ax.imshow(diff, cmap=plt.cm.gray)
#
#for contour in contours:
#    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#
#ax.axis('image')
#ax.set_xticks([])
#ax.set_yticks([])
#plt.show()


contour_lengths=np.zeros(shape=(len(contours),1),dtype=np.uint64)
for i in range(len(contours)):
    contour_lengths[i]=contours[i].shape[0]
    
sort_info=np.argsort(contour_lengths, axis=0)
sort_info=np.squeeze(sort_info)

contour=contours[sort_info[-1]]
r_mask_disc = np.zeros_like(diff, dtype='bool')
r_mask_disc[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
r_mask_disc = ndimage.binary_fill_holes(r_mask_disc)
plt.imshow(r_mask_disc)

contour=contours[sort_info[-3]]
r_mask_cup = np.zeros_like(diff, dtype='bool')
r_mask_cup[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
r_mask_cup = ndimage.binary_fill_holes(r_mask_cup)
plt.imshow(r_mask_cup)

mask=np.zeros(shape = r_mask_cup.shape, dtype=np.uint8)
mask=np.where(r_mask_disc==1,255,mask)
mask=np.where(r_mask_cup==1,128,mask)

plt.imshow(mask)
io.imsave('mask.png', mask)
