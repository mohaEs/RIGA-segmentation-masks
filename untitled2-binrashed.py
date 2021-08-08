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

#./BinRushed//BinRushed1-Corrected/
img_path='./BinRushed//BinRushed1-Corrected/image5-6.jpg'
img_prime_path='./BinRushed/BinRushed1-Corrected/image5prime.jpg'
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


#==> remove outliers
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

if np.max(label)>1:
    print('==> skiped: ')

props=measure.regionprops(label)
center=props[0].centroid
center_row=int(center[0])
center_col=int(center[1])
radius=int(props[0].minor_axis_length/2)-15

#            center_row=np.mean(nonzero_pxls_row).astype(np.uint64)
#            center_col=np.mean(nonzero_pxls_col).astype(np.uint64)
#            
#            s=np.tile([center_row, center_col], (len(nonzero_pxls_col), 1))
#            s=np.transpose(s)
#            
#            nonzero_pxls_ar=np.array([nonzero_pxls_row, nonzero_pxls_col])
#            distances=nonzero_pxls_ar-s
#            distances=np.multiply(distances,distances)
#            distances=np.sum(distances, axis=0)
#            distances=np.sqrt(distances)
#            
#            radius=np.mean(distances).astype(np.uint64)

arr = np.zeros(diff.shape)
rr, cc = draw.circle_perimeter(center_row, center_col, radius=radius, shape=arr.shape)     
arr[rr, cc] = 1
#            io.imsave('img_circle.png', arr)
#            io.imsave('img_diff.png', diff)

#==> determine seeds
#==> seeds should not be close to edges

seed_flag=np.ones((len(rr),1), dtype=np.uint8)
for i in range(len(rr)):
    
    candidate=np.array([rr[i],cc[i]])
    s=np.tile(candidate, (len(nonzero_pxls_row), 1))
    s=np.transpose(s)
    
    distances=nonzero_pxls_ar-s
    distances=np.multiply(distances,distances)
    distances=np.sum(distances, axis=0)
    distances=np.sqrt(distances)
    
    if np.min(distances)<10:
        seed_flag[i]=0

rand=np.random.permutation(len(seed_flag))     
seed_flag[rand[50:]]=0                      

seeds=[]
for k in range(len(rr)):
    if seed_flag[k]==1:
        seed=np.array([rr[k],cc[k]])
        seeds.append(seed)
        
    
#==> region growing with seeds

exemple = regionGrow(img_path, 15, 35000)
segments=exemple.ApplyRegionGrow(seeds)

                
if type(segments)==str:
#    print('s')
    exemple = regionGrow(img_path, 10, 35000)
    segments=exemple.ApplyRegionGrow(seeds)

del exemple
#plt.imshow(segments)    
#            io.imsave('segments.png', segments)

if type(segments)==str:
    print('==> skiped: ')

segments=colorconv.rgb2gray(segments)
#            io.imsave('img_segments.png',segments)


#==> Find contours at a constant value of 0.8 
contours = measure.find_contours(segments, 0.9)            

#            if len(contours)!=4:
#                print('warning')

contour_lengths=np.zeros(shape=(len(contours),1),dtype=np.uint64)
for i in range(len(contours)):
    contour_lengths[i]=contours[i].shape[0]
    
sort_info=np.argsort(contour_lengths, axis=0)
sort_info=np.squeeze(sort_info)

contour=contours[sort_info[-1]]
r_mask_disc = np.zeros_like(diff, dtype='bool')
r_mask_disc[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
r_mask_disc = ndimage.binary_fill_holes(r_mask_disc)
r_mask_disc = dilation(r_mask_disc, square(5))
#plt.imshow(r_mask_disc)

contour=contours[sort_info[-2]]
r_mask_cup = np.zeros_like(diff, dtype='bool')
r_mask_cup[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
r_mask_cup = ndimage.binary_fill_holes(r_mask_cup)
r_mask_cup = erosion(r_mask_cup, square(5))
#plt.imshow(r_mask_cup)

mask=np.zeros(shape = r_mask_cup.shape, dtype=np.uint8)
mask=np.where(r_mask_disc==1,255,mask)
mask=np.where(r_mask_cup==1,128,mask)
#plt.imshow(mask)
io.imsave('img_mask.png', mask)
