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

def combine_masks(in_dir, folder_out, imgname):
    
    Masks=[]
    for i in range(6):            
        
        filename2=imgname+'-'+str(i+1)+'.png'        
        try:
            mask=io.imread(os.path.join(in_dir,filename2), as_gray=True)
            Masks.append(mask)
        except:
            a=0
#            print('ohoh')
        
    if len(Masks)<2:
        return 'Skipped'
    else:
        imgsize=Masks[0].shape
        np_array_cup=np.zeros(shape=[imgsize[0], imgsize[1], len(Masks)])
        np_array_disc=np.zeros(shape=[imgsize[0], imgsize[1], len(Masks)])
        
        for i in range(len(Masks)):
            Mask=Masks[i]
            Mask_disc=np.where(Mask==128, 255, Mask)
            Mask_cup=np.where(Mask==255, 0, Mask)
#            io.imsave('mask_disc.png', Mask_disc)
#            io.imsave('Mask_cup.png', Mask_cup)
            np_array_disc[:,:,i]=Mask_disc
            np_array_cup[:,:,i]=Mask_cup
            
        soft_mask_cup=np.mean(np_array_cup, 2)
        soft_mask_cup=soft_mask_cup*2-1
        soft_mask_cup=np.where(soft_mask_cup==-1, 0, soft_mask_cup)
        soft_mask_disc=np.mean(np_array_disc, 2)
        
        soft_mask_disc=soft_mask_disc.astype(np.uint8)
        soft_mask_cup=soft_mask_cup.astype(np.uint8)
 
        io.imsave(os.path.join(folder_out,'soft_disc', imgname+'prime.png'), soft_mask_disc, check_contrast=False)    
        io.imsave(os.path.join(folder_out,'soft_cup', imgname+'prime.png'), soft_mask_cup, check_contrast=False)
        
        np_array=np.zeros(shape=soft_mask_disc.shape)        
        np_array=np.where(soft_mask_disc>127,255,np_array)
        np_array=np.where(soft_mask_cup>127,128,np_array)
        np_array=np_array.astype(np.uint8)
        
        io.imsave(os.path.join(folder_out,'hards', imgname+'prime.png'), np_array, check_contrast=False)
        
        return 'Done'
    
                    
       
    
    

#%%

folder_1='./MESSIDOR'
folder_segments='./MESSIDOR-segments'
folder_out='./MESSIDOR-Masks'

print('==> preparing Messidor data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))


filenames=[]
for root, dirs, files in os.walk(folder_1):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)
            

            
#%%
        
folder='./Magrabia/MagrabiaMale'
folder_segments='./Magrabia-segments/MagrabiaMale'
folder_out='./Magrabia-Masks/MagrabiaMale'

print('==> preparing MagrabiaMale data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))

filenames=[]
for root, dirs, files in os.walk(folder):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)

#%%
folder='./Magrabia/MagrabiFemale'
folder_segments='./Magrabia-segments/MagrabiFemale'
folder_out='./Magrabia-Masks/MagrabiFemale'

print('==> preparing MagrabiaFemale data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))

filenames=[]
for root, dirs, files in os.walk(folder):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)         
        

#%%        
        

#%%
            
folder='./BinRushed/BinRushed1-Corrected'
folder_segments='./BinRushed-segments/BinRushed1-Corrected'
folder_out='./BinRushed-Masks/BinRushed1-Corrected'

print('==> preparing BinRushed1-Corrected data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))


filenames=[]
for root, dirs, files in os.walk(folder):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)       
        
#%%
        
folder='./BinRushed/BinRushed2'
folder_segments='./BinRushed-segments/BinRushed2'
folder_out='./BinRushed-Masks/BinRushed2'

print('==> preparing BinRushed2 data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))


filenames=[]
for root, dirs, files in os.walk(folder):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)   

#%%
        
folder='./BinRushed/BinRushed3'
folder_segments='./BinRushed-segments/BinRushed3'
folder_out='./BinRushed-Masks/BinRushed3'

print('==> preparing BinRushed3 data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))


filenames=[]
for root, dirs, files in os.walk(folder):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)     

#%%
        
folder='./BinRushed/BinRushed4'
folder_segments='./BinRushed-segments/BinRushed4'
folder_out='./BinRushed-Masks/BinRushed4'

print('==> preparing BinRushed4 data ...')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)
    os.mkdir(os.path.join(folder_out,'hards'))
    os.mkdir(os.path.join(folder_out,'soft_disc'))
    os.mkdir(os.path.join(folder_out,'soft_cup'))

    
filenames=[]
for root, dirs, files in os.walk(folder):
    for filename in files:
        filenames.append(filename)
                
for f in tqdm(range(len(filenames))):
    filename=filenames[f]
    if filename.find('prime') !=-1:
        imgname=filename[:-9]    
        
        status=combine_masks(folder_segments, folder_out, imgname)
        if status=='Skipped':
            print('==> Skipped; ', imgname)           

        
