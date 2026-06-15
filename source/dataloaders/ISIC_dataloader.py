from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import  Dataset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

#Dataset specific selection functions
def select_ink_annotation_images(dataset,criteria=['remove all images without artefact']):
    """
    A function for selecting images with an ink artefact and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a no artefact. These include:
        'remove all images with artefact': Remove all images without artefact
        'set_to_train': Put all images with artefact into the training set
        'set_to_val': Put all images with artefact into the validation set
        'set_to_test': Put all images with artefact into the test set
        'make_no_artefact_binary_classifier': Make a binary classifier with all images with artefact as class 1 and the remaining as class 0
        'make_no_artefact_class': Make a new class for artefact images without changing the other classes

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    artefact_list = np.loadtxt("data/ISIC/annotations/Ink_annotations/ink_annotation.txt", delimiter=',', dtype=str)
    artefact_data =  dataset['total_df'][dataset['total_df']['Path'].isin(artefact_list)]

    #Remove all images that contain a artefact (useful for making an ID or OOD dataset)
    if 'remove all images without artefact' in criteria:
        dataset['total_df'] = artefact_data 
    
    #Put all images with no artefact into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], artefact_data]) if 'train_df' in dataset else artefact_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], artefact_data]) if 'validation_df' in dataset else artefact_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], artefact_data]) if 'test_df' in dataset else artefact_data
        dataset['total_df'] = dataset['total_df'].drop(artefact_data.index)

    #Set all images with no artefact into class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_no_artefact_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in artefact_data else 0)
    elif 'make_no_artefact_class' in criteria: #Make a new class for no artefact
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in artefact_data})

    return dataset


def select_no_ink_annotation_images(dataset, criteria=['remove all images without artefact']):
    """
    A function for selecting images without an ink artefact and controlling how they are used in the dataset.
    Mirror of ``select_ink_annotation_images`` using the complement of ``ink_annotation.txt``.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images without ink artefacts. These include:
        'remove all images without artefact': Keep only images without ink artefacts
        'set_to_train': Put all images without ink artefacts into the training set
        'set_to_val': Put all images without ink artefacts into the validation set
        'set_to_test': Put all images without ink artefacts into the test set
        'make_no_artefact_binary_classifier': Make a binary classifier with all images without ink as class 1 and the remaining as class 0
        'make_no_artefact_class': Make a new class for images without ink without changing the other classes

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    ink_list = np.loadtxt("data/ISIC/annotations/Ink_annotations/ink_annotation.txt", delimiter=',', dtype=str)
    no_ink_data = dataset['total_df'][~dataset['total_df']['Path'].isin(ink_list)]
    no_ink_paths = set(no_ink_data['Path'])

    if 'remove all images without artefact' in criteria:
        dataset['total_df'] = no_ink_data

    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], no_ink_data]) if 'train_df' in dataset else no_ink_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], no_ink_data]) if 'validation_df' in dataset else no_ink_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], no_ink_data]) if 'test_df' in dataset else no_ink_data
        dataset['total_df'] = dataset['total_df'].drop(no_ink_data.index)

    if 'make_no_artefact_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in no_ink_paths else 0)
    elif 'make_no_artefact_class' in criteria:
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in no_ink_paths})

    return dataset


def select_colour_chart_images(dataset,criteria=['remove all images without artefact']):
    """
    A function for selecting images with a colour chart artefact and controlling how they are used in the dataset.

    Parameters
    ----------
    dataset: dict
        A dictionary containing the dataset information
    criteria: list
        A list of strings containing the criteria for selecting images with a no artefact. These include:
        'remove all images with artefact': Remove all images without artefact
        'set_to_train': Put all images with artefact into the training set
        'set_to_val': Put all images with artefact into the validation set
        'set_to_test': Put all images with artefact into the test set
        'make_no_artefact_binary_classifier': Make a binary classifier with all images with artefact as class 1 and the remaining as class 0
        'make_no_artefact_class': Make a new class for artefact images without changing the other classes

    Returns
    -------
    dataset: dict
        A dictionary containing the dataset information
    """
    artefact_list = np.loadtxt("data/ISIC/annotations/Colour_chart/colour_chart.txt", delimiter=',', dtype=str)
    #Get all images with no artefact
    artefact_data =  dataset['total_df'][dataset['total_df']['Path'].isin(artefact_list)]

    #Remove all images that contain a artefact (useful for making an ID or OOD dataset)
    if 'remove all images without artefact' in criteria:
        dataset['total_df'] = artefact_data 
    
    #Put all images with no artefact into the training, validation or test set (enables more control over the dataset images)
    if 'set_to_train' in criteria or 'set_to_val' in criteria or 'set_to_test' in criteria:
        if 'set_to_train' in criteria:
            dataset['train_df'] = pd.concat([dataset['train_df'], artefact_data]) if 'train_df' in dataset else artefact_data
        elif 'set_to_val' in criteria:
            dataset['validation_df'] = pd.concat([dataset['validation_df'], artefact_data]) if 'validation_df' in dataset else artefact_data
        else:
            dataset['test_df'] = pd.concat([dataset['test_df'], artefact_data]) if 'test_df' in dataset else artefact_data
        dataset['total_df'] = dataset['total_df'].drop(artefact_data.index)

    #Set all images with no artefact into class 1 and the remaining to class 0 (for making a binary classifier)
    if 'make_no_artefact_binary_classifier' in criteria:
        dataset['total_df']['class'] = dataset['total_df']['Path'].apply(lambda x: 1 if x in artefact_data else 0)
    elif 'make_no_artefact_class' in criteria: #Make a new class for no artefact
        new_class_int = int(max(dataset['total_df']['class']) + 1)
        dataset['total_df']['class'] = dataset['total_df']['Path'].map({x: new_class_int for x in artefact_data})

    return dataset


def clear_artefacts(dataset,criteria=['remove all images without artefact']):

    training_data_labels = np.loadtxt("data/ISIC/annotations/Training data/training_data.txt", delimiter=',', dtype=str)
    artefact_data =  dataset['total_df'][dataset['total_df']['Path'].isin(training_data_labels)]

    #Remove all images that contain a artefact (useful for making an ID or OOD dataset)
    if 'remove all images with artefact' in criteria:
        dataset['total_df'] = artefact_data 
    
    return dataset
