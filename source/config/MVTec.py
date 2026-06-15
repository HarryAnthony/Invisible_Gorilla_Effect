import torchvision.transforms as T
import torch
import numpy as np
from source.dataloaders.Dataset_class import Dataset_class
from source.util.general_utils import DefaultDict

Database_class = Dataset_class

#Training parameters
num_epochs = 600
momentum = 0.9 
weight_decay = 1e-2
lr_milestones = [int(num_epochs*0.25),int(num_epochs*0.5),int(num_epochs*0.75)]
lr_gamma = 0.2
criterion = 'CrossEntropyLoss'
initialisation_method = 'he'

# network architecture
dropout = 0.3 
depth = 28
widen_factor = 10

# data parameters
image_size = 224

# location of data
root = 'data/MVTec/'
loader_root = 'data/MVTec'
df_name = 'MVTec'

def database_specific_selections(dataset,selections={},**kwargs):
    return dataset

#If setting is not known, then will use the default transform with mean and std of the CheXpert dataset
transform_train = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), # to make the images square
            T.RandomRotation(degrees=15), #Randomly rotate the image by 5 degrees
            T.RandomCrop(224, padding=25), #Randomly crop the image by 4 pixels
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0],std=[1]),]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.RandomRotation(degrees=5),
            T.RandomCrop(224, padding=4),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.30340656638145447, 0.3038780689239502, 0.3263583481311798],
                        std=[0.30353817343711853, 0.30365899205207825, 0.2855879068374634]),
            lambda x: x.expand(3,-1,-1)]),
            
            'setting2' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), 
            T.RandomRotation(degrees=5),
            T.RandomCrop(224, padding=4),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.2142082005739212, 0.2361614853143692, 0.23782064020633698]
                        ,std=[0.15761487185955048, 0.17942944169044495, 0.15038353204727173]),
                        ]),
            
    })

transform_test = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0],std=[1]),]),


            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.30340656638145447, 0.3038780689239502, 0.3263583481311798],
                        std=[0.30353817343711853, 0.30365899205207825, 0.2855879068374634]),]),
            
            'setting2' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224), 
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.2142082005739212, 0.2361614853143692, 0.23782064020633698]
                        ,std=[0.15761487185955048, 0.17942944169044495, 0.15038353204727173]),
                        ]),
            })


#Pre-made dataset selection settings for the CheXpert dataset
dataset_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['good','contamination','crack','faulty_imprint','scratch'], 'classes_OOD': ['color']},
                                           'demographic_selections' : {'Object': ['Pill','equal']},
                                           'dataset_selections': {},
                                           'train_val_test_split_criteria': {'k_fold_split': True, 'k': 5, 'fold': 0}},
                            #Setting 1 trains a classifier on Pill images.

                            'setting2': {'class_selections' : {'classes_ID': ['good','bent','flip','scratch'], 'classes_OOD': ['color']},
                                           'demographic_selections' : {'Object': ['Metal nut','equal']},
                                           'dataset_selections': {},
                                           'train_val_test_split_criteria': {'k_fold_split': True, 'k': 5, 'fold': 0}},
                            #Setting 2 trains a classifier on Metal Nut images.
                                           
                       }
                           


OOD_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['color'], 'classes_OOD': ['']},
                                           'demographic_selections' : {'Object': ['Pill','equal']},
                                           'dataset_selections': {},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},
                            #Setting 1 trains a classifier on Pill images.

                            'setting2': {'class_selections' : {'classes_ID': ['color'], 'classes_OOD': ['']},
                                           'demographic_selections' : {'Object': ['Metal nut','equal']},
                                           'dataset_selections': {},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},
                             #Setting 2 trains a classifier on Metal Nut images.
                                           }
                            


#The classes in the MVTec-AD dataset
classes = ('good','contamination','crack','faulty_imprint','scratch', 'colour',
               'good','bent','flip','scratch')
