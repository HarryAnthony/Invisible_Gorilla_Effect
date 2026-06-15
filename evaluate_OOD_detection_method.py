import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from types import SimpleNamespace
import copy
from source.util.general_utils import select_cuda_device, try_literal_eval
from source.util.processing_data_utils import get_dataloader, get_ood_dataset, get_dataset_selections
from source.util.training_utils import get_class_weights
from source.util.evaluate_network_utils import load_net,  evaluate_accuracy, evaluate_ood_detection_method
from source.util.ood_evaluation_kwargs import METHODS_NEED_TRAINING_DATA, build_ood_evaluation_kwargs
from source.util.Select_dataset import Dataset_selection_methods
from source.util.common_args import create_parser
from make_synthetic_artefacts import RandomErasing_square, RandomErasing_triangle, RandomErasing_polygon, RandomErasing_ring, RandomErasing_text, add_Gaussian_noise, modify_transforms


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

#Check that a valid ood_type is selected
valid_ood_type_options = ['different_dataset','synthetic','different_class']
args.ood_type = try_literal_eval(args.ood_type)
if isinstance(args.ood_type,list):
    for item in args.ood_type:
        if item not in valid_ood_type_options:
            raise Exception('OOD type should be in [different_dataset,synthetic,different_class]')
elif isinstance(args.ood_type,str):
    if args.ood_type not in ['different_dataset','synthetic','different_class']:
        raise Exception('OOD type should be in [different_dataset,synthetic,different_class]')
else:
    raise Exception('OOD type should be in [different_dataset,synthetic,different_class]') 


#Select a GPU to use
use_cuda = select_cuda_device(args.cuda_device)
pin_memory = use_cuda
device_count = torch.cuda.device_count()


#Load net with its associated parameters
net, net_dict, cf = load_net(args.seed,use_cuda=use_cuda)
classes_ID = net_dict['classes_ID']
args.setting = net_dict['setting']
num_classes = net_dict['num_classes']
file_name = net_dict['file_name']
save_dir = net_dict['save_dir']
requires_split = net_dict['Requires split']


#Load parameters specific to this database
batch_size = args.batch_size 
resize = cf.image_size # image size
kwargs_method = {}
trainloader = None


#Load ID and OOD datasets
if requires_split == 0: #If dataset does not need to be split
    df_ID_test = cf.ID_dataset
    if args.method in ['mahalanobis','MBM','ReAct']:
        df_ID_train = cf.train_ID
else: #Else split dataset by classes in and classes out
    root = cf.root

    loader_root = cf.loader_root
    dataset_seed = net_dict['train_val_test_split_criteria']['dataset_seed']
    # load dataframe 
    path_train = os.path.join(cf.root, 'train.csv')
    data_temp = pd.read_csv(path_train)
    path_train_valid = os.path.join(cf.root, 'valid.csv')
    data_valid = pd.read_csv(path_train_valid)
    if os.path.exists(cf.root+'/test.csv'):
        path_train_test = os.path.join(cf.root, 'test.csv')
        data_test = pd.read_csv(path_train_test)
        data = pd.concat([data_temp,data_valid,data_test])
    else:
        data = pd.concat([data_temp,data_valid])

    ID_Dataset_selection = Dataset_selection_methods(data,cf,mode='test')
    ID_dataset = ID_Dataset_selection.apply_selections(class_selections=net_dict['class_selections'],
                                                 demographic_selections=net_dict['demographic_selections'],
                                                 dataset_selections=net_dict['dataset_selections'],
                                                 train_val_test_split_criteria=net_dict['train_val_test_split_criteria'])
    
    df_ID_test = cf.Database_class(cf.loader_root, ID_dataset['test_df'], cf.transform_test[args.setting])
    classes_ID = ID_dataset['classes_ID']
    if args.method in METHODS_NEED_TRAINING_DATA:
        # get dataloader for training data 
        args_train = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'root': loader_root, 'pin_memory': use_cuda, 'device_count': device_count}
        train_weights = get_class_weights(ID_dataset['train_df'])
        df_ID_train = cf.Database_class(cf.loader_root, ID_dataset['train_df'], cf.transform_test[args.setting])
        trainloader = get_dataloader(args=SimpleNamespace(**args_train), dataset=df_ID_train)
        kwargs_method = {'trainloader': trainloader}

OOD_dataset_split = 0
if 'different_dataset' in args.ood_type:
    df_OOD, classes_OOD, OOD_dataset_split, OOD_cf = get_ood_dataset(args.ood_dataset)
if 'different_class' in args.ood_type or OOD_dataset_split==1:
    OOD_cf = cf if 'different_dataset' not in args.ood_type else OOD_cf
    df_OOD = data if 'different_dataset' not in args.ood_type else df_OOD
    
    OOD_Dataset_selection = Dataset_selection_methods(df_OOD,OOD_cf,mode='test')
    OOD_class_selections, OOD_demographic_selections, OOD_dataset_selections, OOD_train_val_test_split_criteria = get_dataset_selections(OOD_cf,args,dataset_seed,get_ood_data=True)
    OOD_dataset = OOD_Dataset_selection.apply_selections(class_selections=OOD_class_selections,
                                                 demographic_selections=OOD_demographic_selections,
                                                 dataset_selections=OOD_dataset_selections,
                                                 train_val_test_split_criteria=OOD_train_val_test_split_criteria)
    
    classes_OOD = OOD_dataset['classes_ID']
    df_OOD = cf.Database_class(OOD_cf.loader_root, OOD_dataset['test_df'], cf.transform_test[args.setting])
if args.ood_type == 'synthetic':
        df_OOD = copy.deepcopy(df_ID_test)
        classes_OOD = classes_ID


#Generate Synthetic Artefacts
if 'synthetic' in args.ood_type: 
    transform_keys = ['scale', 'ratio', 'value', 'setting', 'noise_mean',
    'noise_std', 'coarseness', 'rotation_angle', 'foreign_texture',
    'gaussian_filter_sigma', 'make_transparent', 'transparency_power',
    'triangle_type', 'polygon_coordinates', 'ellipse_parameter',
    'ring_width', 'text', 'font_family']

    transform_kwargs = {key: getattr(args, f'synth_{key}') for key in transform_keys}
    transform_kwargs['p'] = 1

    kernel_size = tuple(np.array(args.synth_scale)*cf.image_size)

    # Dictionary to map synth_artefact values to functions
    artefact_to_transform = {
        'square': RandomErasing_square(**transform_kwargs),
        'triangle': RandomErasing_triangle(**transform_kwargs),
        'polygon': RandomErasing_polygon(**transform_kwargs),
        'ring': RandomErasing_ring(**transform_kwargs),
        'text': RandomErasing_text(**transform_kwargs),
        'Gaussian_noise': add_Gaussian_noise(**transform_kwargs),
        'Gaussian_blur': T.GaussianBlur(kernel_size=tuple(map(lambda x: round(x)+1 if round(x)%2==0 else round(x), kernel_size)), 
                                        sigma=(args.synth_gaussian_filter_sigma if args.synth_gaussian_filter_sigma != 0 else 1)),
        'invert': T.RandomInvert(p=1),
    }

    # Check if args.synth_artefact is in the dictionary, then apply the corresponding transform
    if args.synth_artefact in artefact_to_transform:
        transform_fn = artefact_to_transform[args.synth_artefact]
        df_OOD.transform = modify_transforms(transform_fn, df_OOD.transform, where_to_insert='insert_after', insert_transform=T.Normalize(mean=[0], std=[1]))
    else:
        raise(Exception(f'Artefact {args.synth_artefact} not recognised. Available options are {list(artefact_to_transform.keys())}'))
    

args_dataloader = {'resize': resize, 'batch_size': batch_size, 'shuffle': False, 'pin_memory': use_cuda, 'device_count': device_count}
ID_loader = get_dataloader(args=SimpleNamespace(**args_dataloader), dataset=df_ID_test)
OOD_loader = get_dataloader(args=SimpleNamespace(**args_dataloader), dataset=df_OOD)



if args.method not in ['iForest_seed','FPI_seed','ddpm_seed' ,'DeepSVDD_seed' ,'XOOD-M seed' ,'FPI', 'PII', 'RotPred', 'DUQ', 'DeepSVDD', 'ddpm', 'RotPred_seed', 'PixelCNN', 'PixelCNN_seed', 'CIDER', 'CIDER_mahal_seed', 'DUQ', 'DUQ_seed', 'Likelihood_ratios_seed']:
    from source.util.model_methods_util import add_methods_to_model
    add_methods_to_model(net)
    layer_names = net.list_layers()
    

#Evaluate accuracy of ID and OOD datasets on the ID classification task
if args.evaluate_OOD_accuracy == True:
    assert any(item in classes_OOD for item in classes_ID), 'There must be class overlap between ID and OOD'
    evaluate_accuracy(net,OOD_loader,use_cuda=use_cuda,save_results=args.save_results,plot_metric=args.plot_metric,save_dir='outputs/experiment_outputs',filename='OOD')
if args.evaluate_ID_accuracy == True:
    evaluate_accuracy(net,ID_loader,use_cuda=use_cuda,save_results=args.save_results,plot_metric=args.plot_metric,save_dir='outputs/experiment_outputs')

kwargs_test = build_ood_evaluation_kwargs(
    args, use_cuda, net_dict, num_classes, trainloader=trainloader
)

np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))



if args.return_metrics == True:
    AUROC, AUCPR = evaluate_ood_detection_method(args.method,net,ID_loader,OOD_loader,return_metrics=True,**kwargs_test)
    
else:
    evaluate_ood_detection_method(args.method,net,ID_loader,OOD_loader,**kwargs_test)



