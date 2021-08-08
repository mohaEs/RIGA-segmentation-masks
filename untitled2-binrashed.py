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
from skimage import draw
from region_growing import regionGrow
from skimage.morphology import convex_hull_image
from skimage.morphology import square
from skimage.morphology import binary_opening
from skimage.morphology import binary_closing
from skimage.morphology import erosion
from skimage.morphology import dilation

img_path='./BinRushed/image29-1.jpg'
img_prime_path='./BinRushed/image29prime.jpg'
image=io.imread(img_path, as_gray=False)
image_prime=io.imread(img_prime_path, as_gray=False)

image=(image)/np.max(image)
image_prime=(image_prime)/np.max(image_prime)

diff=image_prime-image
#plt.imshow(diff)


diff=colorconv.rgb2gray(diff)
#plt.imshow(diff)
diff=(diff)/np.max(diff)
#plt.imshow(diff)

diff=np.where(diff<0.25, 0, diff)
diff=np.where(diff>0, 1, diff)
#plt.imshow(diff)


#if 
#warning



nonzero_pxls= np.nonzero(diff)
nonzero_pxls_row=nonzero_pxls[0]
nonzero_pxls_col=nonzero_pxls[1]
nonzero_pxls_ar=np.array([nonzero_pxls_row, nonzero_pxls_col])
avg_row=np.mean(nonzero_pxls_row)
avg_col=np.mean(nonzero_pxls_col)


s=np.tile([avg_row, avg_col], (len(nonzero_pxls[0]), 1))
s=np.transpose(s)

distances=nonzero_pxls_ar-s
distances=np.multiply(distances,distances)
distances=np.sum(distances, axis=0)
distances=np.sqrt(distances)

#==> remove outliers
mean = np.mean(distances)
standard_deviation = np.std(distances)
distance_from_mean = abs(distances - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation

nonzero_pxls_row=nonzero_pxls_row[not_outlier]
nonzero_pxls_col=nonzero_pxls_col[not_outlier]
nonzero_pxls_ar=np.array([nonzero_pxls_row, nonzero_pxls_col])
cleanedDiff=np.zeros(diff.shape)
cleanedDiff[nonzero_pxls_row,nonzero_pxls_col]=1

#==> compute a circle inside
chull = convex_hull_image(cleanedDiff)
label=measure.label(chull)
props=measure.regionprops(label)
center=props[0].centroid
center_row=int(center[0])
center_col=int(center[1])
radius=int(props[0].minor_axis_length/2)-15
    

#
##==> compute a circle inside
#center_row=np.mean(nonzero_pxls_row).astype(np.uint64)
#center_col=np.mean(nonzero_pxls_col).astype(np.uint64)
#
#s=np.tile([center_row, center_col], (len(nonzero_pxls_col), 1))
#s=np.transpose(s)
#
#distances=nonzero_pxls_ar-s
#distances=np.multiply(distances,distances)
#distances=np.sum(distances, axis=0)
#distances=np.sqrt(distances)

#radius=(np.mean(distances)).astype(np.int64)

arr = np.zeros(diff.shape)
rr, cc = draw.circle_perimeter(center_row, center_col, radius=radius, shape=arr.shape)
arr[rr, cc] = 1
io.imsave('img_circle.png', arr)
io.imsave('img_diff.png', diff)


#==> determine seeds
#==> seeds should not be close to edges

seed_flag=np.ones((len(rr),1), dtype=np.uint8)
for i in range(len(rr)):
    
    candidate=np.array([rr[i],cc[i]])
    s=np.tile(candidate, (len(nonzero_pxls_row), 1))
    s=np.transpose(s)
    
    distances_=nonzero_pxls_ar-s
    distances_=np.multiply(distances,distances)
    distances_=np.sum(distances, axis=0)
    distances_=np.sqrt(distances)
    
    if np.min(distances_)<5:
        seed_flag[i]=0
        
#    ind_min=np.argmin(distances_)
#    ind_max=np.argmin(distances_)
#    
#    if distances[ind_min] < radius:
#        seed_flag[i]=0
#        
#    if seed_flag[i]==0:
#        arr[rr[i], cc[i]] = 0
#    
#io.imsave('img_circle_2.png', arr)

rand=np.random.permutation(len(seed_flag))     
seed_flag_reserved=seed_flag
seed_flag[rand[50:]]=0
              
seeds=[]
for k in range(len(rr)):
    if seed_flag[k]==1:
        seed=np.array([rr[k],cc[k]])
        seeds.append(seed)

   
#==> region growing with seeds
exemple = regionGrow(img_path, 15, 30000)
segments=exemple.ApplyRegionGrow(seeds)
    
if segments=='None':
#    print('s')
    seed_flag=seed_flag_reserved
    seed_flag[rand[100:]]=0
    exemple = regionGrow(img_path, 10, 30000)
    segments=exemple.ApplyRegionGrow(seeds)
#plt.imshow(segments)    
io.imsave('img_segments.png', segments)

segments=colorconv.rgb2gray(segments)
#==> Find contours at a constant value of 0.8 
contours = measure.find_contours(segments, 0.9)


if len(contours)!=4:
    print('warning')

contour_lengths=np.zeros(shape=(len(contours),1),dtype=np.uint64)
for i in range(len(contours)):
    contour_lengths[i]=contours[i].shape[0]
    
sort_info=np.argsort(contour_lengths, axis=0)
sort_info=np.squeeze(sort_info)

contour=contours[sort_info[-1]]
r_mask_disc = np.zeros_like(diff, dtype='bool')
r_mask_disc[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
r_mask_disc = ndimage.binary_fill_holes(r_mask_disc)
#plt.imshow(r_mask_disc)
#r_mask_disc = binary_closing(r_mask_disc, square(20))
r_mask_disc = dilation(r_mask_disc, square(5))
#plt.imshow(r_mask_disc)

contour=contours[sort_info[-2]]
r_mask_cup = np.zeros_like(diff, dtype='bool')
r_mask_cup[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
r_mask_cup = ndimage.binary_fill_holes(r_mask_cup)
#r_mask_cup = binary_opening(r_mask_cup, square(5))
r_mask_cup = erosion(r_mask_cup, square(5))
#plt.imshow(r_mask_cup)

mask=np.zeros(shape = r_mask_cup.shape, dtype=np.uint8)
mask=np.where(r_mask_disc==1,255,mask)
mask=np.where(r_mask_cup==1,128,mask)

#plt.imshow(mask)
io.imsave('img_mask.png', mask)
