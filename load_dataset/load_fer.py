from torchvision.transforms import ToTensor, Normalize, Grayscale, CenterCrop, Compose, Resize
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms


# Standard Libraries
from os import path, listdir
import sys
import csv
import re
import cv2
import random

# Modules
from load_dataset import preprocess_data


class FERplus(Dataset):
    def __init__(self, idx_test_fold, idx_set=0, max_loaded_images_per_label=1000, transforms=None, base_path_to_FER_plus=None):
        """
            Code based on https://github.com/microsoft/FERPlus.

            :param idx_set: Labeled = 0, Validation = 1, Test = 2
            :param max_loaded_images_per_label: Maximum number of images per label
            :param transforms: transforms (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.idx_test_fold = idx_test_fold
        self.idx_set = idx_set
        self.max_loaded_images_per_label = max_loaded_images_per_label
        self.transforms = transforms
        self.base_path_to_FER_plus = base_path_to_FER_plus
        self.fer_sets = {0: 'FER2013Train/', 1: 'FER2013Valid/', 2: 'FER2013Test/'}
        
        # Default values
        self.num_labels = 8
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
        
        """import cv2
        cv2.imshow("s", sample['image'])
 
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows()"""

        
        sample['image'] = Image.fromarray(sample['image'])

        if not (self.transforms is None):
            sample['image'] = self.transforms(sample['image'])
        
        image = Normalize(mean=self.mean, std=self.std)(ToTensor()(sample['image'])) 
        label = sample['emotion']


        return image, label, idx

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
        Parse labels to make them compatible with AffectNet.
        :param idx:
        :return:
        """
        emo_to_return = np.argmax(idx)

        if emo_to_return == 2:
            emo_to_return = 3
        elif emo_to_return == 3:
            emo_to_return = 2
        elif emo_to_return == 4:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 4

        return emo_to_return

    @staticmethod
    def _process_data(emotion_raw):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal)
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size

        # find the peak value of the emo_raw list
        maxval = max(emotion_raw)
        if maxval > 0.5 * sum_list:
            emotion[np.argmax(emotion_raw)] = maxval
        else:
            emotion = emotion_unknown  # force setting as unknown

        return [float(i) / sum(emotion) for i in emotion]

    def _load(self):
        final_data, final_labels = [], []
        for i in range(3):
            csv_label = []
            data, labels = [], []
            counter_loaded_images_per_label = [0 for _ in range(self.num_labels)]
    
            path_folders_images = path.join(self.base_path_to_FER_plus, 'Images', self.fer_sets[i])  #self.idx_set
            path_folders_labels = path.join(self.base_path_to_FER_plus, 'Labels', self.fer_sets[i])
    
            with open(path_folders_labels + '/label.csv') as csvfile:
                lines = csv.reader(csvfile)
                for row in lines:
                    csv_label.append(row)
    
            # Shuffle training set
            if self.idx_set == 0:
                np.random.shuffle(csv_label)
    
            for l in csv_label:
                emotion_raw = list(map(float, l[2:len(l)]))
                emotion = self._process_data(emotion_raw)
                emotion = emotion[:-2]
    
                try:
                    emotion = [float(i) / sum(emotion) for i in emotion]
                    emotion = self._parse_to_label(emotion)
                except ZeroDivisionError:
                    emotion = 9
    
                if (emotion < self.num_labels) and (counter_loaded_images_per_label[int(emotion)] < self.max_loaded_images_per_label):
                    counter_loaded_images_per_label[int(emotion)] += 1
    
                    img = np.array(preprocess_data.read(path.join(path_folders_images, l[0])), np.uint8)
    
                    box = list(map(int, l[1][1:-1].split(',')))
    
                    if box[-1] != 48:
                        print("[INFO] Face is not centralized.")
                        print(path.join(path_folders_images, l[0]))
                        print(box)
                        exit(-1)
    
                    img = img[box[0]:box[2], box[1]:box[3], :]
                    img = preprocess_data.resize(img, (96, 96))
                    """import cv2
                    cv2.imshow("s", img)
         
                    cv2.waitKey(0) # waits until a key is pressed
                    cv2.destroyAllWindows()"""
                    #img = preprocess_data.resize(img, (224, 224))

                    data.append(img)
                    labels.append(emotion)

                has_loading_finished = (np.sum(counter_loaded_images_per_label) >= (self.max_loaded_images_per_label * self.num_labels))

            if has_loading_finished:
                break      
            
            final_data += data
            final_labels += labels
        
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
        
        
            
        #return [np.array(to_return_data), np.array(to_return_labels)]

# FER+ <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



if __name__ == '__main__':

    for fold in range(0, 1):
    
        data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=30,
                                                translate=(.1, .1),
                                                scale=(1.0, 1.25),
                                                interpolation=transforms.InterpolationMode.BILINEAR)]
            
        train_dataset = FERplus(fold, idx_set=0,
                                        max_loaded_images_per_label=1000000,
                                        transforms=transforms.Compose(data_transforms),
                                        base_path_to_FER_plus='../DB/FER+/')
        print('Train set size:', train_dataset.__len__())
            
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = 32,
                                                    num_workers = 2,
                                                    shuffle = True,  
                                                    pin_memory = True)
        
        val_dataset = FERplus(fold, idx_set=1,
                                    max_loaded_images_per_label=1000000,
                                    transforms=None,
                                    base_path_to_FER_plus='../DB/FER+/')
        print('Validation set size:', val_dataset.__len__())
            
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size = 32,
                                                num_workers = 2,
                                                shuffle = False,  
                                                pin_memory = True)
        
        test_dataset = FERplus(fold, idx_set=2,
                                    max_loaded_images_per_label=1000000,
                                    transforms=None,
                                    base_path_to_FER_plus='../DB/FER+/')
        print('Testing set size:', test_dataset.__len__())
            
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size = 32,
                                                num_workers = 2,
                                                shuffle = False,  
                                                pin_memory = True)
            
    
        print("---------------------------------------------------------------------------------")

  