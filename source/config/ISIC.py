import torchvision.transforms as T
import torch
from source.dataloaders.Dataset_class import Dataset_class
from source.dataloaders.ISIC_dataloader import select_ink_annotation_images, select_no_ink_annotation_images, clear_artefacts, select_colour_chart_images
import numpy as np
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

root = 'data/ISIC/'
loader_root = 'data/ISIC'
df_name = 'ISIC'

def database_specific_selections(dataset,selections={},**kwargs):
    if 'ink_artefact_selection' in selections.keys():
        dataset = select_ink_annotation_images(dataset,criteria=selections['ink_artefact_selection'])
    if 'no_ink_artefact_selection' in selections.keys():
        dataset = select_no_ink_annotation_images(dataset,criteria=selections['no_ink_artefact_selection'])
    if 'clear_artefacts' in selections.keys():
        dataset = clear_artefacts(dataset,criteria=selections['clear_artefacts'])
    if 'colour_chart_selection' in selections.keys():
        dataset = select_colour_chart_images(dataset,criteria=selections['colour_chart_selection'])
    return dataset

#If setting is not known, then will use the default transform with mean and std of the CheXpert dataset
transform_train = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.RandomRotation(degrees=45), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.76355016,0.5461531 ,0.5705491],
                       std=[0.08832154,0.11716927,0.13153051])]),


            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.RandomRotation(degrees=45), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.7170533,0.5532407,0.5117366],
                        std=[0.11591253,0.13099958,0.13956805])
            ]),

            'setting2' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.RandomRotation(degrees=45), #
            T.RandomCrop(224, padding=25), 
            T.RandomHorizontalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.2),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.7170533,0.5532407,0.5117366],
                        std=[0.11591253,0.13099958,0.13956805])]),

            },
            )

transform_test = DefaultDict(T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.76355016,0.5461531 ,0.5705491],
                       std=[0.08832154,0.11716927,0.13153051])
            ]),

            {'setting1' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.7170533,0.5532407,0.5117366],
                        std=[0.11591253,0.13099958,0.13956805])
            ]),


            'setting2' : T.Compose([
            T.Resize((224,224)),
            T.CenterCrop(224),  
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=[0.7170533,0.5532407,0.5117366],
                        std=[0.11591253,0.13099958,0.13956805])
            ])


            }
)

#Pre-made dataset selection settings for the CheXpert dataset
dataset_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['malignant','benign'], 'classes_OOD': []}, #Colour chart
                                           'demographic_selections' : {},
                                           'dataset_selections': {'clear_artefacts':['remove all images with artefact']},
                                           'train_val_test_split_criteria': {'k_fold_split': True, 'k': 5, 'fold': 0}},

                            'setting2': {'class_selections' : {'classes_ID': ['malignant','benign'], 'classes_OOD': []}, #Colour chart
                                           'demographic_selections' : {},
                                           'dataset_selections': {'clear_artefacts':['remove all images with artefact']},
                                           'train_val_test_split_criteria': {'k_fold_split': True, 'k': 5, 'fold': 0}},

                            }


OOD_selection_settings = {'setting1': {'class_selections' : {'classes_ID': ['malignant','benign'], 'classes_OOD': [], 'replace_values_dict':{np.nan:0}},
                                           'demographic_selections' : {},
                                            'dataset_selections': {'remove_duplicates':[], 'ink_artefact_selection':['remove all images without artefact']},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},

                        'setting2': {'class_selections' : {'classes_ID': ['malignant','benign'], 'classes_OOD': [], 'replace_values_dict':{np.nan:0}},
                                           'demographic_selections' : {},
                                            'dataset_selections': {'remove_duplicates':[], 'colour_chart_selection':['remove all images without artefact']},
                                           'train_val_test_split_criteria': {'valSize': 0, 'testSize': 1}},

                                           }


#The classes in the ISIC
classes = ('AIMP','acrochordon','actinic keratosis','angiofibroma or fibrous papule','angiokeratoma','angioma','atypical melanocytic proliferation','atypical spitz tumor','basal cell carcinoma','cafe-au-lait macule','clear cell acanthoma','dermatofibroma','lentigo NOS','lentigo simplex','lichenoid keratosis','melanoma','melanoma metastasis','mucosal melanosis','neurofibroma','nevus','nevus spilus','other','pigmented benign keratosis','pyogenic granuloma','scar','sebaceous adenoma','sebaceous hyperplasia','seborrheic keratosis','solar lentigo','squamous cell carcinoma','vascular lesion','verruca','indeterminate','benign','indeterminate','malignant')
