# This code may not be run on GFG IDE 
# as required packages are not found. 
    
# STRATIFIES K-FOLD CROSS VALIDATION { 10-fold }
  
# Import Required Modules.
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn import datasets
  
from torchvision.transforms import ToTensor, Normalize, Grayscale, Compose, Resize
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import numpy as np
import torch

# Standard Libraries
from os import path, listdir

# Modules
from load_dataset import preprocess_data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import re
import cv2

# Standard Libraries
from os import path, makedirs
import copy

# Modules

def sort_numeric_directories(dir_names):
    return sorted(dir_names, key=lambda x: (int(re.sub("\D", "", x)), x))


class CohnKanade(Dataset):
    def __init__(self, idx_test_fold, set_id, transforms=None, base_path_to_dataset=None):
        

        self.idx_test_fold = idx_test_fold
        self.base_path_to_dataset = base_path_to_dataset

        if set_id == 'training' or set_id == 'validation' or set_id == 'testing':
            self.set = set_id
        else:
            raise RuntimeError("The 'set' variable must be 'training_labeled', 'training_unlabeled', 'validation' or 'test'.")

        self.transforms = transforms
        self.loaded_data = self._load()
        # Default values
        self.mean = 0.0
        self.std = 1.0
        
        self.label = self.loaded_data[:][1]

    def __len__(self):
        return self.loaded_data[0].shape[0]

    def __getitem__(self, idx):

        sample = {'image': self.loaded_data[0][idx], 'emotion': self.loaded_data[1][idx]}
    
        sample['image'] = np.array(sample['image'],dtype=np.uint8)
        sample['image'] = Image.fromarray(sample['image'])


        if not (self.transforms is None):
            sample['image'] = self.transforms(sample['image'])

        return Normalize(mean=[self.mean], std=[self.std])(ToTensor()(sample['image'])), sample['emotion'], idx

    def online_normalization(self, x):
        return Normalize(mean=[self.mean], std=[self.std])(ToTensor()(x))

    def norm_input_to_orig_input(self, x):
        return (x * self.std) + self.mean

    @staticmethod
    def get_class(idx):
        classes = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happiness',
            4: 'Sadness',
            5: 'Surprise',
            6: 'Contempt',
            7: 'Neutral'}

        return classes[idx]

    @staticmethod
    def _parse_to_label(idx):
        to_return = np.zeros(8, dtype=np.float32)

        if idx == 0:
            to_return[0] = 1.
        elif idx == 1:
            to_return[6] = 1.
        elif idx == 2:
            to_return[1] = 1.
        elif idx == 3:
            to_return[2] = 1.
        elif idx == 4:
            to_return[3] = 1.
        elif idx == 5:
            to_return[4] = 1.
        elif idx == 6:
            to_return[5] = 1.
        elif idx == -1:
            to_return[7] = 1.

        return np.argmax(to_return)

    def _load(self):
        # Setting the training, validation and tes indexes
        validation_fold_index = (self.idx_test_fold + 1) % 10
        training_fold_index = set(range(10))

        
        training_fold_index = list(training_fold_index)

        
        # Setting the absolute path to the dataset
        path_images = self.base_path_to_dataset + 'extended-cohn-kanade-images/cohn-kanade-images/'
        path_labels = self.base_path_to_dataset + 'Emotion_labels/Emotion/'
        # Setting general variables
        folds_data_index = [[], [], [], [], [], [], [], [], [], []]
        data, labels = [], []
        subject_step = -1
        index_image = 0
        data_file_name = []

        # Loading the dataset
        subjects = np.sort(np.array(listdir(path_labels)))
        for subject in subjects:
            subject_not_counted = True
            path_sessions_images = path_images + subject + '/'
            path_sessions_labels = path_labels + subject + '/'
            sessions = np.sort(np.array(listdir(path_sessions_labels)))

            for session in sessions:
                path_frames = path_sessions_images + session + '/'
                path_classes = path_sessions_labels + session + '/'
                file_class = np.sort(np.array(listdir(path_classes)))
                frames = np.sort(np.array(listdir(path_frames)))

                # Has expression, then load image
                if len(file_class) > 0:
                    if subject_not_counted:
                        subject_not_counted = False
                        subject_step = subject_step + 1 if subject_step < 9 else 0

                    # Adding the facial expression category
                    label_file = 0
                    with open(path_classes + file_class[-1]) as f:
                        for line in f:
                            label_file = int(np.float32(line.split()[0])) - 1
                    for i in range(3):
                        
                        image_is = io.imread(path_frames + frames[-(i + 1)])
                        image_is = Image.fromarray(image_is)
                

                        dt_tr = [Grayscale(num_output_channels=3),]
                        tr = Compose(dt_tr)
                        image_is = tr(image_is)
                        if not (np.shape(image_is) == (490, 640, 3)):
                            dt_tr = [Resize((490,640)),]
                            tr = Compose(dt_tr)
                            image_is = tr(image_is)
                        #print("---",np.shape(image_is))
                        image_is = np.array(image_is)
                    
                        folds_data_index[subject_step].append(index_image)
                        index_image += 1
                        data.append(image_is)
                        data_file_name.append(path_frames + frames[-(i + 1)])
                        labels.append(self._parse_to_label(label_file))
                        
                        """if i ==0:
                            print(self._parse_to_label(label_file))
                            cv2.imshow("baseball", np.array(image_is))
                            cv2.waitKey(0)"""

                    # Adding the neutral category example (index equals 7)
                    image_is = io.imread(path_frames + frames[0])
                    image_is = Image.fromarray(image_is)

                    dt_tr = [Grayscale(num_output_channels=3),]
                    tr = Compose(dt_tr)
                    image_is = tr(image_is)
                    if not (np.shape(image_is) == (490, 640, 3)):
                        dt_tr = [Resize((490,640)),]
                        tr = Compose(dt_tr)
                        image_is = tr(image_is)
                    image_is = np.array(image_is)                    
    
                    folds_data_index[subject_step].append(index_image)
                    index_image += 1
                    data.append(image_is)
                    data_file_name.append(path_frames + frames[0])
                    labels.append(self._parse_to_label(-1))
                            

        data = np.array(data)
        labels = np.array(labels)
        #print("!!!!!!!!!!!",np.unique(np.array(labels),return_counts=True))
        
        # Create StratifiedKFold object.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        x_train_fold, x_test_fold, y_train_fold, y_test_fold  = [], [], [], []
        
        to_return_data, to_return_labels = [], []
        for train_index, test_index in skf.split(data, labels):
            
          
            if(self.set!="training"):
                X_valid, X_test, y_valid, y_test = train_test_split(data[test_index], labels[test_index], test_size=0.5, random_state=1)

                if (self.set=="validation"):
                    to_return_data.append(X_valid)
                    to_return_labels.append(y_valid)
                else:
                    to_return_data.append(X_test)
                    to_return_labels.append(y_test)
                    
            else:
                to_return_data.append(data[train_index])
                to_return_labels.append(labels[train_index])
      
        return [np.array(to_return_data[self.idx_test_fold]), np.array(to_return_labels[self.idx_test_fold])]
        
        
    
    

    
    
if __name__ == '__main__':

    for fold in range(0, 1):
    
        data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.Grayscale(num_output_channels=3),
                         transforms.Resize(96),
                         transforms.RandomAffine(degrees=30,
                                                   translate=(.1, .1),
                                                   scale=(1.0, 1.25),
                                                   interpolation=transforms.InterpolationMode.BILINEAR)]
            
        train_dataset = CohnKanade(fold,
                                    'training',
                                     transforms=transforms.Compose(data_transforms),
                                     base_path_to_dataset='../DB/CK+/')
        print('Train set size:', train_dataset.__len__())
            
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = 32,
                                                       num_workers = 2,
                                                       shuffle = True,  
                                                       pin_memory = True)
        
        
        val_dataset = CohnKanade(fold,
                                'validation',
                                transforms=transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                                                transforms.Resize(96),]),
                                 base_path_to_dataset='../DB/CK+/')
      
        print('Validation set size:', val_dataset.__len__())
        
        
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 32,
                                               num_workers = 2,
                                               shuffle = False,  
                                               pin_memory = True)
        
        
        test_dataset = CohnKanade(fold,
                                'testing',
                                transforms=transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                                                transforms.Resize(96),]),
                                 base_path_to_dataset='../DB/CK+/')
      
        print('Validation set size:', test_dataset.__len__())
        
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = 32,
                                               num_workers = 2,
                                               shuffle = False,  
                                               pin_memory = True)
        
    
        print("---------------------------------------------------------------------------------")

  
