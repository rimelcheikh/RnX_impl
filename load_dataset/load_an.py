from torchvision.transforms import ToTensor, Normalize, Grayscale, CenterCrop, Compose, Resize
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

# Standard Libraries
from os import path, listdir
import sys
import csv
import re
import cv2
import random

# Modules
from load_dataset import preprocess_data

def sort_numeric_directories(dir_names):
    return sorted(dir_names, key=lambda x: (int(re.sub("\D", "", x)), x))


class AffectNetCategorical(Dataset):
    def __init__(self, idx_test_fold, idx_set=0, max_loaded_images_per_label=1000, transforms=None, is_norm_by_mean_std=True,
                 base_path_to_affectnet=None):
        """
            This class follows the experimental methodology conducted by (Mollahosseini et al., 2017).
            Refs.
            Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. Affectnet: A database for facial expression,
            valence, and arousal computing in the wild. IEEE Transactions on Affective Computing.
            :param idx_set: Labeled = 0, Unlabeled = 1, Validation = 2, Test = Not published by
                            (Mollahosseini et al., 2017)
            :param max_loaded_images_per_label: Maximum number of images per label
            :param transforms: transforms (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.idx_test_fold = idx_test_fold
        self.idx_set = idx_set
        self.max_loaded_images_per_label = max_loaded_images_per_label
        self.transforms = transforms
        self.base_path_to_affectnet = base_path_to_affectnet
        self.affectnet_sets = {'supervised': 'train_set/',
                               'unsupervised': 'Training_Unlabeled/',
                               'validation': 'val_set/'}

        # Default values
        self.num_labels = 8
        if is_norm_by_mean_std:
            self.mean = [149.35457 / 255., 117.06477 / 255., 102.67609 / 255.]
            self.std = [69.18084 / 255., 61.907074 / 255., 60.435623 / 255.]
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]

        # Load data
        self.loaded_data = self._load()
        print('Size of the loaded set: {}'.format(self.loaded_data[0].shape[0]))
        
        self.label = self.loaded_data[:][1]

    def __len__(self):
        return self.loaded_data[0].shape[0]

    def __getitem__(self, idx):
        sample = {'image': self.loaded_data[0][idx], 'emotion': self.loaded_data[1][idx]}
        sample['image'] = Image.fromarray(sample['image'])

        if not (self.transforms is None):
            sample['image'] = self.transforms(sample['image'])

        return Normalize(mean=self.mean, std=self.std)(ToTensor()(sample['image'])), sample['emotion'], idx

    def online_normalization(self, x):
        return Normalize(mean=self.mean, std=self.std)(ToTensor()(x))

    def norm_input_to_orig_input(self, x):
        x_r = torch.zeros(x.size())
        x_r[0] = (x[2] * self.std[2]) + self.mean[2]
        x_r[1] = (x[1] * self.std[1]) + self.mean[1]
        x_r[2] = (x[0] * self.std[0]) + self.mean[0]
        return x_r

    @staticmethod
    def get_class(idx):
        classes = {
            0: 'Neutral',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt'}

        return classes[idx]

    @staticmethod
    def _parse_to_label(idx):
        """
            The file name follows this structure: 'ID_s_exp_s_val_s_aro_.jpg' Ex. '0000000s7s-653s653.jpg'.
            Documentation of labels adopted by AffectNet's authors:
            Expression: expression ID of the face (0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No-Face)
            Valence: valence value of the expression in interval [-1,+1] (for Uncertain and No-face categories the value is -2)
            Arousal: arousal value of the expression in interval [-1,+1] (for Uncertain and No-face categories the value is -2)
            :param idx: File's name
            :return: label
        """
        s = idx.split("images/")
        ss = s[1].split(".")
        #print(s[0]+"annotations/"+ss[0]+"_exp.npy")
        if(path.isfile(s[0]+"annotations/"+ss[0]+"_exp.npy")):
            label = np.load(s[0]+"annotations/"+ss[0]+"_exp.npy")     
            discrete_label = np.int32(label)
            ok = True
            res = discrete_label if (discrete_label < 8) else -1
            #print(res)
        else:
            res = -1
            ok = False
        return res , ok
        

    def _load(self):
        final_data, final_labels = [], []
        
        for i in ['train_set', 'val_set']:
            data_affect_net, labels_affect_net = [], []
            counter_loaded_images_per_label = [0 for _ in range(self.num_labels)]
            
            path_folders_affect_net = path.join(self.base_path_to_affectnet, i)
    
            images_folder_affect_net = sort_numeric_directories(listdir(path_folders_affect_net+'/images'))
            labels_folder_affect_net = sort_numeric_directories(listdir(path_folders_affect_net+'/annotations'))
    
    
            cont = 0
    
            for f_af in images_folder_affect_net:
                path_to_image_affect_net = path.join(path_folders_affect_net+'/images/', f_af)
    
                #for file_name_image_affect_net in images_affect_net:
                lbl, is_exp = self._parse_to_label(path_to_image_affect_net)
                if (lbl >= 0) and (is_exp) and (counter_loaded_images_per_label[int(lbl)] < self.max_loaded_images_per_label):
                    img = np.array(preprocess_data.read(path_to_image_affect_net), np.uint8)
    
                    data_affect_net.append(img)
                    labels_affect_net.append(lbl)
                    cont += 1
                    
                    """if(lbl == 0):
                        cv2.imshow("baseball", np.array(img))
                        cv2.waitKey(0)"""
    
                    counter_loaded_images_per_label[int(lbl)] += 1
    
                has_loading_finished = (np.sum(counter_loaded_images_per_label) >= (
                            self.max_loaded_images_per_label * self.num_labels))
    
                if has_loading_finished:
                    break
            final_data += data_affect_net
            final_labels += labels_affect_net
        
        final_data = np.array(final_data)
        final_labels = np.array(final_labels)
        
        # Create StratifiedKFold object.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        
        to_return_data, to_return_labels = [], []
        for train_index, test_index in skf.split(final_data, final_labels):
            
            
            if(self.idx_set!=0):
                X_valid, X_test, y_valid, y_test = train_test_split(final_data[test_index], final_labels[test_index], test_size=0.5, random_state=1)

                if (self.idx_set==1):
                    to_return_data.append(X_valid)
                    to_return_labels.append(y_valid)
                else:
                    to_return_data.append(X_test)
                    to_return_labels.append(y_test)
                    
            else:
                to_return_data.append(final_data[train_index])
                to_return_labels.append(final_labels[train_index])
      
            
        return [np.array(to_return_data[self.idx_test_fold]), np.array(to_return_labels[self.idx_test_fold])]
        
    

# AffectNet (Categorical) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == '__main__':

    for fold in range(0, 1):
    
            
        train_data = AffectNetCategorical(fold, idx_set=0,
                                   max_loaded_images_per_label=100,
                                   transforms=None, #transforms.Compose(data_transforms),
                                   base_path_to_affectnet='C:/Users/rielcheikh/Desktop/FER/DB/AffectNet/')
    
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory = True)
        print('Train set size:', train_data.__len__())
    
        # validation set
        """val_data = AffectNetCategorical(fold, idx_set=1,
                                 max_loaded_images_per_label=1000000,
                                 transforms=None, #transforms.Grayscale(num_output_channels=3),
                                 base_path_to_affectnet='C:/Users/rielcheikh/Desktop/FER/DB/AffectNet/')
        
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory = True)
        print('Val set size:', val_data.__len__())
        
        #testing set
        test_data = AffectNetCategorical(fold, idx_set=2,
                                    max_loaded_images_per_label=1000000,
                                    transforms=None, #transforms.Grayscale(num_output_channels=3),
                                    base_path_to_affectnet='C:/Users/rielcheikh/Desktop/FER/DB/AffectNet/')
    
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory = True)
        print('Test set size:', test_data.__len__())"""
            
    
        print("---------------------------------------------------------------------------------")

