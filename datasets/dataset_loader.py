'''
Dataset for training
'''
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy import ndimage
from skimage.transform import resize
import imgaug.augmenters as iaa
from sklearn.model_selection import KFold
import random
import cv2
    

class BrainDataset(Dataset):
    def __init__(self, image_set, image_size, augmentation):
        self.image_size = image_size
        self.augmentation = augmentation

        # Read image names and labels
        df = pd.read_csv(image_set)
        self.image_names = list(df.image_names)
        self.labels = list(df.labels)
    

    def __len__(self):
        return len(self.image_names)
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): index of the data in the folder/list
        Returns:
            tuple: (arrayimage, target) where arrayimage is tensor (1, slice, height, width), target is the label (categorical).
        """
        arrayimage = cv2.imread(self.image_names[index])[:,:,::-1] # read the image, convert BGR space to RGB space 
        arrayimage = arrayimage.astype("float32") 
        
        # Padding the image to a square and then resized to a fixed size
        arrayimage = self._pad_and_resize(arrayimage)

        if self.augmentation == 1:
            max_value = np.max(arrayimage)
            seq = iaa.Sequential([                
                iaa.flip.Fliplr(p=0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(rotate=(-30, 30),
                    cval=250)
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.ShearX((-10, 10))
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.01*max_value, 0.05*max_value), per_channel=0.5),
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0.2, 0.6))
                ),
                ], random_order=True)
            
            arrayimage = seq(image=arrayimage)
            
        arrayimage = arrayimage.astype("float32") 
        label = self.labels[index]
        arrayimage = np.transpose(arrayimage, (2,0,1))
        return arrayimage, label
 
            
    def _pad_and_resize(self, arrayimage):
        h = arrayimage.shape[0]
        w = arrayimage.shape[1]
        top, bottom, left, right = 0, 0, 0, 0
        if h<w:
            delta_h = w - h
            top, bottom = delta_h//2, delta_h-(delta_h//2)
        elif h>w:
            delta_w = h - w
            left, right = delta_w//2, delta_w-(delta_w//2)
        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(arrayimage, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        # Resize the image
        resized_img = resize(new_im, order=1, output_shape=self.image_size)
        return resized_img

    
if __name__ == '__main__':
    image_set = '/Users/HemingY/Documents/brain_classification/data/split/fold_0_train.csv'
    image_size = (512, 512)
    augmentation = 1
    dataset = BrainDataset(image_set, image_size, augmentation)


    
    
    