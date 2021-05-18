# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:04:19 2021

@author: meslami
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage    
from skimage import measure
from skimage import io
from skimage.color import colorconv
from tqdm import tqdm


#%%
def make_masks(in_dir, out_dir, filename):
    
    if (filename.find('tif') != -1) or (filename.find('jpg') != -1):
        
    
        image_prime=io.imread(os.path.join(in_dir,filename))
        imgname=filename[:-9] 
        
        for i in range(6):
            
            try:
                filename2=imgname+'-'+str(i+1)+'.tif'
                #print(filename2)
                
                image=io.imread(os.path.join(in_dir,filename2), as_gray=False)
            
            except:
                filename2=imgname+'-'+str(i+1)+'.jpg'
                #print(filename2)
                
                image=io.imread(os.path.join(in_dir,filename2), as_gray=False)                
            diff=image_prime-image
            diff=colorconv.rgb2gray(diff)
            diff=(255*diff)/np.max(diff)
            
            contours = measure.find_contours(diff, 0.8)
            
            if len(contours)!=4 :
                if len(contours)!=5 :
                    print('==> skipped:', filename2)
            else:
            
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
                
                contour=contours[sort_info[1]]
                r_mask_cup = np.zeros_like(diff, dtype='bool')
                r_mask_cup[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
                r_mask_cup = ndimage.binary_fill_holes(r_mask_cup)
                #plt.imshow(r_mask_cup)
                
                mask=np.zeros(shape = r_mask_cup.shape, dtype=np.uint8)
                mask=np.where(r_mask_disc==1,255,mask)
                mask=np.where(r_mask_cup==1,128,mask)
                
                # plt.imshow(mask)
                filename_save=os.path.join(out_dir, filename2[:-3]+'png')
                io.imsave(filename_save, mask, check_contrast=False)


#%%
            
folder_1='./MESSIDOR'
folder_out='./MESSIDOR-segments'

print('==> preparing Messidor data ...')

if not os.path.exists(folder_out):
    os.mkdir(folder_out)

filenames=[]
for root, dirs, files in os.walk(folder_1):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        make_masks(folder_1, folder_out, filename)                    
