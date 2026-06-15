"""
Helper functions for evaluating the network and its OOD detection performance.
"""
import pandas as pd
import torch
import numpy as np 
from torch.autograd import Variable
import sklearn.metrics as skm
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from torch.nn import functional as F
from torchvision.models import resnet18, vgg11, vgg16_bn, resnet34, resnet50, vgg16, vit_b_32, swin_t
from source.models.wide_resnet import Wide_ResNet
from source.util.training_utils import set_activation_function, add_dropout_network_architechture
from source.util.general_utils import print_progress, try_literal_eval
from torch.utils.data import DataLoader,Dataset
from source.external_methods.external_models.Wideresnet_Dec import WideResNetDec, WideResNetEncoder
from source.ad_hoc_methods.ad_hoc_models.RotPred_models import RotPredModel
from source.ad_hoc_methods.ad_hoc_utils.bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from source.external_methods.external_models.DDPM import MyDDPM, MyUNet
from source.ad_hoc_methods.ad_hoc_models.CIDER import SupCEHeadResNet
from types import SimpleNamespace



def check_net_exists(seed, verbose=True, get_output=False):
    """
    Check if the net exists in the database of nets.

    Parameters
    ----------
    seed : int
        The seed of the net to check.
    verbose : bool, optional
        Whether to print the net information. The default is True.
    get_output : bool, optional
        Whether to return the net information. The default is False.

    Raises
    ------
    Exception
        If the net does not exist in the database.

    Returns
    -------
    None if get_output is False, otherwise returns a dictionary containing the net information.
    """

    model_list = pd.read_csv('outputs/saved_models/model_list.csv', quotechar='"')#,dtype=dtype_spec)

    matching_models = model_list[model_list['Seed'] == int(seed)]
    if len(matching_models) == 0:
        raise Exception('Experiment seed is not in the list of known experiments')
    model = matching_models.iloc[0]

    if verbose:
        print('Model database: {}\nModel setting: {}\nModel type: {}\nModel widen factor x depth: {} x {}\nDropout: {}\n'.format(
            model['Database'], model['Setting'], model['Model_type'], model['Widen_factor'],
            model['Depth'], model['Dropout']))

    if get_output:
        net_info = {
            'Model pathname': model['Model_name'],
            'Model database': model['Database'],
            'Model setting': 'setting'+str(int(model['Setting'])),
            'Model architecture': model['Model_type'],
            'Model widen factor': model['Widen_factor'],
            'Model depth': model['Depth'],
            'Model activation': model['Activation_function'],
            'Dropout': model['Dropout'],
            'Act_func_Dropout': model['act_func_dropout'],
            'DUQ': model['DUQ'],
            'Requires split': model['requires_split'],
            'Dataset seed': model['Dataset_seed'],
            'class_selections': model['class_selections'],
            'demographic_selections': model['demographic_selections'],
            'dataset_selections': model['dataset_selections'],
            'train_val_test_split_criteria': model['train_val_test_split_criteria'],
            'num_classes': int(model['num_classes'])
        }

        return net_info
    return None


def load_net(seed,verbose=True,use_cuda=True):
    """
    Load the net from the database of nets.

    Parameters
    ----------
    seed : int
        The seed of the net to load.
    verbose : bool, optional
        Whether to print the net information. The default is True.
    use_cuda : bool, optional
        Whether to use cuda. The default is True.
    ensemble : bool, optional
        Whether the net is being used for an ensemble. The default is False.

    Raises
    ------
    Exception
        If the net does not exist in the database.

    Returns
    -------
    net : torch.nn.Module
        The loaded net.
    net_dict : dict
        A dictionary containing the net information.
    cf : config file
        The configuration file for the dataset.

    """
    net_info = check_net_exists(seed,verbose=verbose,get_output=True)

    net_dict = {'Requires split': net_info['Requires split'],
                'setting': net_info['Model setting']}

    #Get configuration for given datasets
    if net_info['Model database'] == 'CheXpert' or net_info['Model database'] == 'cheXpert' or net_info['Model database'] == 'chexpert':
        from source.config import chexpert as cf_chexpert
        net_info['Model database'] = 'chexpert'
        cf = cf_chexpert
    elif net_info['Model database'] == 'ISIC':
        from source.config import ISIC as cf_ISIC
        cf = cf_ISIC
    elif net_info['Model database'] == 'MVTec':
        from source.config import MVTec as cf_MVTec
        cf = cf_MVTec


    #Get the classes in and out
    if net_info['Requires split']:
        net_dict['class_selections'] = turn_str_into_dict(net_info['class_selections'])
        net_dict['demographic_selections'] = turn_str_into_dict(net_info['demographic_selections'])
        net_dict['dataset_selections'] = turn_str_into_dict(net_info['dataset_selections'])
        net_dict['train_val_test_split_criteria'] = turn_str_into_dict(net_info['train_val_test_split_criteria'])
        net_dict['classes_ID'] = net_dict['class_selections']['classes_ID']
        net_dict['classes_OOD'] = net_dict['class_selections']['classes_OOD']
    else:
        net_dict['classes_ID'] = cf.classes
        net_dict['classes_OOD'] = []

    net_dict['num_classes'] = net_info['num_classes']

    if verbose:
        print('| Preparing '+net_info['Model database']+' test with the following classes: ')
        print(f"| Classes ID: {net_dict['classes_ID']}")
        print(f"| Classes OOD: {net_dict['classes_OOD']}\n")

    #Select the network architecture
    model_architecture_dict = {
    'wide-resnet': (Wide_ResNet, ['Model depth', 'Model widen factor', 'Dropout']),
    'ResNet18': (resnet18, []),
    'ResNet34': (resnet34, []),
    'ResNet50': (resnet50, []),
    'vgg11': (vgg11, []),
    'vgg16_bn': (vgg16_bn, []),
    'vgg16': (vgg16, []),
    'vit_b_32': (vit_b_32, []),
    'swin_t': (swin_t, []),
    'WideResNetDec': (WideResNetDec, []),
    'WideResNetDec-Encoder': (WideResNetEncoder, []),
    'Rotpred-ResNet18': (resnet18, []),
    'BNN-ResNet18': (resnet18, []),
    'UNet': (MyUNet, []),
    'DDPM': (MyDDPM, ['n_steps', 'min_beta', 'max_beta', 'device', 'image_chw']),
    'CIDER-ResNet18': (SupCEHeadResNet, []),
    }
    model_architecture = net_info['Model architecture']
    model_func, model_args = model_architecture_dict.get(model_architecture, (None, None))
    if model_func is None:
        raise ValueError('Invalid model architecture')
    
    #Load the network architecture
    kwargs = {}
    if model_architecture != 'DDPM':
        kwargs['num_classes'] = int(net_dict['num_classes'])
    if model_architecture == 'WideResNetDec':
        kwargs['num_classes'] = 1

    # Special handling for DDPM
    if model_architecture == 'DDPM':
        n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02
        # Initialize UNet first
        unet = MyUNet(n_steps=n_steps, in_channels=3, out_channels=3)
        # Then initialize DDPM with the UNet
        net = model_func(unet, n_steps=n_steps, min_beta=min_beta, 
                        max_beta=max_beta, 
                        image_chw=(3,224,224))
    elif model_architecture == 'CIDER-ResNet18':
        CIDER_args = SimpleNamespace(
            num_classes = int(net_dict['num_classes']),
            head = 'linear',
            feat_dim = 128,
            w = 1,
            promo_m = 0.5,
            net_type='ResNet18'
        )

        net = model_func(args=CIDER_args)
    else:
        net = model_func(**kwargs)
    if net_info['Model architecture'] == 'Rotpred-ResNet18':
        net =  RotPredModel(net,int(net_dict['num_classes']))
    if net_info['Model architecture'] == 'BNN-ResNet18':
       const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # initialize mu/sigma from the dnn weights
        "moped_delta": 0.2,
        }
       dnn_to_bnn(net, const_bnn_prior_parameters)
 
    
    net = add_dropout_network_architechture(net,net_info)
    net_dict['file_name'] = f"{str(net_info['Model architecture'])}-{int(net_info['Model depth'])}x{net_info['Model widen factor']}_{str(net_info['Model database'])}-{int(seed)}"
    net_dict['save_dir'] = os.path.join("outputs", f"{net_info['Model database']}_{net_info['Model setting']}")
    net_dict['pathname'] = net_info['Model pathname']

    # Model setup
    assert os.path.isdir('outputs/saved_models'), 'Error: No saved_models directory found!'
    if use_cuda:
        checkpoint = torch.load('outputs/saved_models/'+net_info['Model database']+'/'+net_info['Model pathname']+'.pth')
    else:
        checkpoint = torch.load('outputs/saved_models/'+net_info['Model database']+'/'+net_info['Model pathname']+'.pth',map_location='cpu')
    #Apply parameters and activation function to the network
    params = {}
    for k_old in checkpoint.keys():
        k_new = k_old.replace('module.', '')
        params[k_new] = checkpoint[k_old]
    net.load_state_dict(params)
    net = set_activation_function(net,net_info['Model activation'])

    net_dict['act_func_dropout_rate'] = net_info['Act_func_Dropout']
    net_dict['net_type'] = net_info['Model architecture']

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    return net, net_dict, cf



def turn_str_into_dict(string):
    """
    Turn a string into a dictionary.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    convert_dict : dict
        The converted dictionary.
    """
    import re
    pattern = r"('replace_values_dict': {)([^}]*?)nan([^}]*?})"
    replacement = r'\1\2"null"\3'
    string = re.sub(pattern, replacement, string)

    convert_dict = try_literal_eval(string)

    if 'replace_values_dict' in convert_dict.keys():
        if 'null' in convert_dict['replace_values_dict'].keys():
            convert_dict['replace_values_dict'][np.nan] = convert_dict['replace_values_dict']['null']
            convert_dict['replace_values_dict'].pop('null')
 
    return convert_dict
                


def evaluate_ood_detection_method(method,net,idloader,oodloader,return_metrics=False,**kwargs):
    """
    Evaluate the OOD detection performance of a net for a given method.

    Parameters
    ----------
    method : str
        The name of the OOD detection method.
    net : torch.nn.Module
        The net to evaluate.
    idloader : torch.utils.data.DataLoader
        The dataloader for the in-distribution dataset.
    oodloader : torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    return_metrics : bool, optional
        Whether to return the AUROC and AUCPR. The default is False.

    Raises
    ------
    ValueError
        If the method is invalid.

    Returns
    -------
    AUROC : float
        The AUROC (if return_metrics is True).
    AUCPR : float
        The AUCPR (if return metrics is True).
    """
    from source.post_hoc_methods import mcp, odin, mcdp, deepensemble, mahalanobis, ReAct, GRAM, gradnorm, DICE, PCX, NuSA, WeiPer, Negative_aware_norm, SHE, ASH, LOF, KNN, KDE, Residual, ViM, GAIA, XOOD_M, TAPUUD, GradOrth, COP, CORP, FeatureNorm, NAC, Neural_Mean_Discrepancy
    from source.external_methods import evaluate_DDPM, evaluate_RealNVP, evaluate_FPI_seed as evaluate_FPI, evaluate_DeepSVDD
    from source.ad_hoc_methods import evaluate_BNN, evaluate_RotPred, evaluate_CIDER, Reject_class


    OOD_detection_dict = {'MCP': {'function': mcp.evaluate, 'name': ['MCP']},
                           'ODIN': {'function': odin.evaluate, 'name': ['ODIN']},
                           'MCDP': {'function': mcdp.evaluate, 'name': ['MCDP']},
                           'deepensemble': {'function': deepensemble.evaluate, 'name': ['Deep_ensemble']},
                            'mahalanobis' : {'function': mahalanobis.evaluate, 'name': ['Mahalanobis']},
                            'MBM':  {'function': mahalanobis.evaluate_MBM, 'name': ['MBM']},
                            'ReAct': {'function': ReAct.evaluate, 'name': ['ReAct']},
                            'GRAM': {'function': GRAM.evaluate, 'name': ['GRAM']},
                            'gradnorm': {'function': gradnorm.evaluate, 'name': ['GradNorm']},
                            'DICE': {'function': DICE.evaluate, 'name': ['DICE']},
                            'NMD': {'function': Neural_Mean_Discrepancy.evaluate, 'name': ['NMD']},
                            'PCX': {'function': PCX.evaluate, 'name': ['PCX']},
                            'NuSA': {'function': NuSA.evaluate, 'name': ['NuSA']},
                            'WeiPer': {'function': WeiPer.evaluate, 'name': ['WeiPer']},
                            'negative_aware_norm': {'function': Negative_aware_norm.evaluate, 'name': ['Negative_aware_norm']},
                            'SHE': {'function': SHE.evaluate, 'name': ['SHE']},
                            'ASH': {'function': ASH.evaluate, 'name': ['ASH']},
                            'LOF': {'function': LOF.evaluate, 'name': ['LOF']},
                            'KNN': {'function': KNN.evaluate, 'name': ['KNN']},
                            'KDE': {'function': KDE.evaluate, 'name': ['KDE']},
                            'Residual': {'function': Residual.evaluate, 'name': ['Residual']},
                            'ViM': {'function': ViM.evaluate, 'name': ['ViM']},
                            'GAIA': {'function': GAIA.evaluate, 'name': ['GAIA']},
                            'XOOD-M': {'function': XOOD_M.evaluate, 'name': ['XOOD_M']}, 
                            'TAPUUD':  {'function': TAPUUD.evaluate, 'name': ['TAPUUD']}, 
                            'GradOrth':  {'function': GradOrth.evaluate, 'name': ['GradOrth']}, 
                            'COP': {'function': COP.evaluate, 'name': ['CoP']}, 
                            'CORP': {'function': CORP.evaluate, 'name': ['CoRP']},
                            'FeatureNorm': {'function': FeatureNorm.evaluate, 'name': ['FeatureNorm']}, #1
                            'NAC': {'function': NAC.evaluate, 'name': ['NAC']}, #4
                            'ddpm': {'function': evaluate_DDPM.evaluate, 'name': ['ddpm']},
                            'Reject_class': {'function': Reject_class.evaluate, 'name': ['Reject_class']},
                            'Norm_flow': {'function': evaluate_RealNVP.evaluate, 'name': ['Norm_flow']},
                            'BNN': {'function': evaluate_BNN.evaluate, 'name': ['BNN_flipout_seed']},
                            'RotPred_seed': {'function': evaluate_RotPred.evaluate, 'name': ['RotPred_seed']},
                            'CIDER': {'function': evaluate_CIDER.evaluate, 'name': ['CIDER']},
                            'FPI': {'function': evaluate_FPI.evaluate, 'name': ['FPI']},
                            'DeepSVDD': {'function': evaluate_DeepSVDD.evaluate, 'name': ['DeepSVDD']},

    }
    
    if method not in OOD_detection_dict.keys():
        raise ValueError(f'Invalid OOD detection method: {method}')
    kwargs['OOD_dict'] = OOD_detection_dict[method]
    
    if return_metrics == True:
        OOD_detection_method_scores = ood_evaluation(OOD_detection_dict[method], net, idloader, oodloader, return_metrics=True,**kwargs)
        return OOD_detection_method_scores

    ood_evaluation(OOD_detection_dict[method], net, idloader, oodloader, **kwargs)




def ood_evaluation(ood_detection_method, net, idloader, oodloader, verbose=True, save_results=False, save_results_micro=False, first_edition=False, save_dir=None, return_metrics=False, plot_metric=False, combine_metrics=True, use_cuda=True, filename='', **kwargs):
    """
    Evaluate the OOD detection performance of a net for a given method.

    Parameters
    ----------
    ood_detection_method : dict
        The dictionary of the OOD detection method and the methods name.
    net : torch.nn.Module
        The net to evaluate.
    idloader : torch.utils.data.DataLoader
        The dataloader for the in-distribution dataset.
    oodloader : torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    verbose : bool, optional
        Whether to print the AUROC and AUCPR. The default is True.
    save_results : bool, optional
        Whether to save the results in textfiles. The default is False.
    save_dir : str, optional
        The directory to save the results, requires save_results to be True. The default is None.
    return_metrics : bool, optional
        Whether to return the AUROC and AUCPR. The default is False.
    plot_metric : bool, optional
        Whether to plot a visualisation of the OOD detection metric, requires save_results to be True. The default is False.
    filename : str, optional
        The filename to save the results as, requires save_results to be True. The default is ''.

    Returns
    -------
    AUROC : float
        The AUROC (if return_metrics is True).
    AUCPR : float
        The AUCPR (if return metrics is True).
    """

    confidence_id_ood = ood_detection_method['function'](net, idloader, oodloader, **kwargs)
    
    ood_detection_method['name'] = kwargs['OOD_dict']['name'] if ood_detection_method['name'] != kwargs['OOD_dict']['name'] else ood_detection_method['name']
    confidence_id_ood = [confidence_id_ood] if isinstance(confidence_id_ood[0][0],(float,int,np.float32)) == True else confidence_id_ood
    OOD_detection_method_scores = []
    

    if ood_detection_method['name'][0] == 'RotPred':
        net.use_rotation_head = False


    if save_results_micro:
        if save_dir is None:
            raise ValueError('save_dir must be set when save_results_micro is True (use --save_results True).')
        idloader_f, oodloader_f = loader_with_paths(idloader,oodloader)
        net.eval()    
        [id_names,id_logits_list,id_labels_list,id_correct_list,id_predicted_list],[ood_names,ood_logits_list,ood_labels_list,ood_correct_list,ood_predicted_list] = get_image_micro_results(idloader_f,oodloader_f,net,verbose=False,use_cuda=use_cuda)
        
        metrics_filename_ID = "Micro_metrics_ID%s.txt" % (filename) if (len(ood_detection_method['name'])!=1 or combine_metrics==True or first_edition==True) else "Micro_metrics_ID_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f1_path = os.path.join(str(save_dir), str(metrics_filename_ID))
        if combine_metrics == False:
            with open(f1_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['OOD detection method', 'Image', 'Metric', 'Correct', 'Target', 'Predicted'])
        metrics_filename_OOD = "Micro_metrics_OOD%s.txt" % (filename) if (len(ood_detection_method['name'])!=1 or combine_metrics==True or first_edition==True) else "Micro_metrics_OOD_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f2_path = os.path.join(str(save_dir), str(metrics_filename_OOD))
        if combine_metrics == False:
            with open(f2_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['OOD detection method', 'Image', 'Metric', 'Correct','Target', 'Predicted'])


    for idx, (id, ood) in enumerate(confidence_id_ood):
        AUROC, AUCPR = get_AUROC_AUCPR(id, ood)
        OOD_detection_name = ood_detection_method['name'][idx]
        OOD_detection_method_scores.append([OOD_detection_name,AUROC, AUCPR])

        if verbose and not save_results_micro:
            print(OOD_detection_name, 'AUROC:', AUROC, 'AUCPR:', AUCPR)

        if save_results_micro:
            with open(f1_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                for image_name,metric,correct_bool,target,predicted in zip(id_names,id,id_correct_list,id_labels_list,id_predicted_list):
                    writer.writerow([OOD_detection_name,image_name,metric,correct_bool,target,predicted])
            with open(f2_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                for image_name,metric,correct_bool,target,predicted in zip(ood_names,ood,ood_correct_list,ood_labels_list,ood_predicted_list):
                    writer.writerow([OOD_detection_name,image_name,metric,correct_bool,target,predicted])
            if plot_metric:
                plot_images_wrt_metric(ood,ood_names,oodloader_f,net,save_fig=True,plot_fig=False,save_dir=save_dir,filename='OOD_%s' % (OOD_detection_name))
                plot_images_wrt_metric(id,id_names,idloader_f,net,save_fig=True,plot_fig=False,save_dir=save_dir,filename='ID_%s' % (OOD_detection_name))
            if verbose:
                print(OOD_detection_name, 'AUROC:', AUROC, 'AUCPR:', AUCPR)

    if save_results:
        if save_dir is None:
            raise ValueError('save_dir must be set when save_results is True (use --save_results True).')
        metrics_filename = "Macro_metrics%s.txt" % (filename) if (len(ood_detection_method['name'])!=1 or combine_metrics==True or first_edition==True) else "Macro_metrics_%s%s.txt" % (ood_detection_method['name'][0],filename)
        f4_path = os.path.join(save_dir, metrics_filename)
        if combine_metrics == False:
            with open(f4_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['OOD detection method', 'AUROC', 'AUCPR'])
                for (OOD_method_name, AUROC, AUCPR) in OOD_detection_method_scores:
                    writer.writerow([OOD_method_name, AUROC, AUCPR])
        else:
            with open(f4_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                for (OOD_method_name, AUROC, AUCPR) in OOD_detection_method_scores:
                    writer.writerow([OOD_method_name, AUROC, AUCPR])

    if return_metrics:
        return OOD_detection_method_scores
    
    
def get_softmax_score(inputs,net,use_cuda=True,required_grad=False,softmax_only=False,temper=1,**kwargs):
    """
    Classify inputs using a given neural network and output the softmax scores.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to classify.
    use_cuda : bool
        Whether to use cuda.
    net : torch.nn.Module
        The neural network to use.
    required_grad : bool, optional
        Whether to require gradients for the inputs. The default is False.
    softmax_only : bool, optional
        Whether to only output the softmax scores. The default is False.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    outputs : torch.Tensor
        The outputs of the neural network.
    inputs : torch.Tensor
        The inputs to the neural network.
    softmax_score : torch.Tensor
        The softmax outputs of the neural network.
    """
    if use_cuda:
        inputs = inputs.cuda()
    inputs = Variable(inputs, requires_grad=required_grad)
    outputs = net(inputs)
    softmax_score = softmax(outputs,temper=temper)  #Convert outputs into softmax
    if softmax_only == True:
        return softmax_score
    return outputs, inputs, softmax_score


def get_softmax_score_report_accuracy(inputs,targets,use_cuda,net,correct,total,logits_list,labels_list,correct_list,predicted_list,required_correct_list=False,**kwargs):
    """
    Classify inputs with a given neural network, output the softmax scores and report the accuracy of the classifier.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to classify.
    targets : torch.Tensor
        The targets of the inputs.
    use_cuda : bool
        Whether to use cuda.
    net : torch.nn.Module
        The neural network to use.
    correct : int
        The number of correct predictions.
    total : int
        The total number of predictions.
    logits_list : list
        The list of logits.
    labels_list : list
        The list of labels.
    correct_list : list
        The list of correct predictions.
    predicted_list : list
        The list of predicted labels.
    required_grad : bool, optional
        Whether requireS gradients for the inputs. The default is False.
    required_correct_list : bool, optional
        Whether requireS the correct list. The default is False.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    outputs : torch.Tensor
        The outputs of the neural network.
    inputs : torch.Tensor
        The inputs to the neural network.
    nnOutputs : torch.Tensor
        The softmax outputs of the neural network.
    hidden : torch.Tensor
        The hidden layer outputs of the neural network.
    total : int
        The total number of predictions.
    """
    outputs, inputs, softmax_score = get_softmax_score(inputs,net,use_cuda=use_cuda,**kwargs)

    if use_cuda:
        targets = targets.cuda()
    targets = Variable(targets)

    with torch.no_grad():
            logits_list.append(outputs.data)
            labels_list.append(targets.data)

    if required_correct_list:
        #Compare classifier outputs to targets to get accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_list.extend(predicted.eq(targets.data).cpu().tolist())
        predicted_list.extend(predicted.cpu().tolist())
        return outputs, inputs, softmax_score, total, correct, logits_list, labels_list,correct_list,predicted_list

    return outputs, inputs, softmax_score, logits_list, labels_list


def calculate_accuracy(logits_list,labels_list,correct,total,correct_list,confidence_list,ece_criterion,verbose=True):
    """
    Calculate the accuracy and AUC of an ood_method.

    Parameters
    ----------
    logits_list : list
        The list of logits.
    labels_list : list
        The list of labels.
    correct : int
        The number of correct predictions.
    total : int
        The total number of predictions.
    correct_list : list
        The list of correct predictions.
    confidence_list : list
        The list of confidence scores.
    ece_criterion : torch.nn.Module
        The ECE criterion.
    verbose : bool, optional
        Whether to print the accuracy. The default is False.

    Returns
    -------
    acc : float
        The accuracy of the classifier.
    acc_list : float
        The accuracy of the classifier.
    auroc_classification : float
        The AUROC of the classifier.
    """
    #Calculate the accuracy
    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece = ece_criterion(logits, labels)
    acc = 100.*correct/total
    acc_list = (sum(correct_list)/len(correct_list))

    pred_probs_total = combine_arrays([confidence_list,correct_list])
    pred_probs_total_sort = np.array(pred_probs_total)[np.array(pred_probs_total)[:, 0].argsort()]
    confidence_list = np.array([pred_probs_total_sort[i][0] for i in  range(len(pred_probs_total))])
    correct_list = np.array([pred_probs_total_sort[i][1] for i in range(len(pred_probs_total))])


    from sklearn.metrics import balanced_accuracy_score
    bal_acc = balanced_accuracy_score(torch.argmax(logits,dim=1).cpu(), labels.cpu())

    # calculate AUROC for classifcation accuracy
    fpr, tpr, _ = skm.roc_curve(y_true = correct_list, y_score = confidence_list, pos_label = 1) #positive class is 1; negative class is 0
    auroc_classification = skm.auc(fpr, tpr)


    if verbose:
        print("| Test Result\tAcc@1: %.2f%%" %(acc))
        print(f'| ECE: {ece.item()}')
        print(f'| Acc list: {acc_list}')
        print(f'| AUROC classification: {auroc_classification}')
        print(f'| Balanced accuracy : {bal_acc}')


    return acc, auroc_classification, ece.item(), bal_acc


def softmax(outputs, temper=1):
    """
    Calculate the softmax using the outputs of a neural network.

    Parameters
    ----------
    outputs : torch.Tensor
        The outputs of a neural network.
    temper : float, optional
        The temperature for the softmax. The default is 1.

    Returns
    -------
    nnOutputs : torch.Tensor
        The softmax outputs of the neural network.
    """
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs/temper) / np.sum(np.exp(nnOutputs/temper), axis=1, keepdims=True)
    return nnOutputs


def evaluate_accuracy(net,loader,verbose=True,use_cuda=True,save_results=False,save_dir='',plot_metric=False,filename='ID',return_outputs=False):
    ece_criterion = ECELoss()
    if use_cuda:
        ece_criterion.cuda()
    net.eval()
    net.training = False
    correct, total = 0, 0
    total = 0
    logits_list = []
    labels_list = []
    confidence_list = np.array([])
    correct_list = []
    predicted_list = []

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    l = len(loader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (inputs, targets) in enumerate(loader):
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        with torch.no_grad():
            _, inputs, softmax_score, total, correct, logits_list, labels_list,correct_list,predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,total,logits_list,labels_list,correct_list,predicted_list,required_grad=False,required_correct_list=True)
        confidence_list = np.concatenate([confidence_list,np.max(softmax_score,axis=1)])

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    labels_list_array = [value.item() for tensor in labels_list for value in tensor.view(-1)]

    set_style(fontsize=12)

    cm = confusion_matrix(labels_list_array, predicted_list) #normalize='false'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No nevus','nevus']) #['benign_keratosis','nevus','vascular_lesion','basal_cell_carcinoma','melanoma','dermatofibroma','actinic_keratosis'])
    disp.plot()
    disp.plot(xticks_rotation=90)
    plt.title('No Devices ID, Pacemaker OOD task (OOD data)')
    plt.tight_layout()
    plt.show()

    acc, auroc, ece, bal_acc = calculate_accuracy(logits_list,labels_list,correct,total,correct_list,confidence_list,ece_criterion)


    if save_results:
        f1_path = os.path.join(save_dir,'ID_task_accuracy'+str(filename)+'.txt')
        with open(f1_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID task accuracy', 'AUROC', 'ECE'])
            for (acc_val, auroc_val, ece_val) in [(acc, auroc, ece)]:
                writer.writerow([acc_val, auroc_val, ece_val])

        true_confidences = [confidence for i, confidence in enumerate(confidence_list) if correct_list[i]]
        false_confidences = [confidence for i, confidence in enumerate(confidence_list) if not correct_list[i]]
        correct_bool_list = [True if i in correct_list else False for i in range(len(confidence_list))]

        f2_path = os.path.join(save_dir,'ID_task_confidence'+str(filename)+'.txt')
        with open(f2_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Confidence', 'Correct'])
            for (confidence_val, correct_val) in [(confidence_list, correct_bool_list)]:
                writer.writerow([confidence_val, correct_val])



    if return_outputs == True:
        return acc, bal_acc, labels_list


def get_metrics(path_id_confidence, path_ood_confidence, verbose=True, normalized=True):
    """ 
    Returns most common metrics (AUC, FPR, TPR) for comparing OOD vs ID inputs.
    Assumes that values are probabilities/confidences between 0 and 1 as default. 
    If not, please set normalized to False.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    verbose : bool, optional
        Whether to print the metrics. The default is True.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.

    Returns
    -------
    auroc : float
        The AUROC of the classifier.
    aucpr : float
        The AUCPR of the classifier.
    fpr : float
        The FPR of the classifier.
    tpr : float
        The TPR of the classifier.
    """
    id = np.loadtxt(path_id_confidence)
    ood = np.loadtxt(path_ood_confidence)
    if verbose:
        print('Mean confidence OOD: {}, Median: {}, Length: {}'.format(np.mean(ood), np.median(ood), len(ood)))
        print('Mean confidence ID: {}, Median: {}, Length: {}'.format(np.mean(id), np.median(id), len(id)))
    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
    if not normalized:
        # use unity based normalization to also catch negative values
        pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    pred_probs_total = combine_arrays([pred_probs,true_labels])
    pred_probs_total_sort = np.array(pred_probs_total)[np.array(pred_probs_total)[:, 0].argsort()]
    pred_probs = np.array([pred_probs_total_sort[i][0] for i in  range(len(pred_probs_total))])
    true_labels = np.array([pred_probs_total_sort[i][1] for i in range(len(pred_probs_total))])
    fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
    auroc = skm.auc(fpr, tpr)
    precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
    aucpr = skm.auc(recall, precision)
    if verbose:
        print('AUROC: {}'.format(auroc))
    return auroc, aucpr, fpr, tpr


def combine_arrays(input_arrays):
    """
    Combines elements of arrays into a single array

    Parameters
    ----------
    input_arrays : list
        A list of arrays to be combined.

    Returns
    -------
    output_array : list
        A list of arrays containing the combined elements of the input arrays.
    """
    if any(len(input_arrays[0])!= len(i) for i in input_arrays):
        raise Exception("Lists must all be the same length")

    output_array = []
    for i in range(0,len(input_arrays[0])):
        output_array.append([])
        for j in range(0,len(input_arrays)):
            output_array[i].append(input_arrays[j][i])
    
    return output_array


def plot_softmax_confidence(correct_predictions, incorrect_predictions, normalized=True, title='ODD metric', save_dir=0, show_plot=False):
    """
    Plots the histograms of the ood metric for comparing OOD and ID inputs.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.
    title : str, optional
        The title of the plot. The default is 'ODD metric'.
    save_dir : str, optional
        The directory to save the plot. The default is 0.
    """
    max_val = np.max([np.max(correct_predictions),np.max(incorrect_predictions)])
    min_val = np.min([np.min(correct_predictions),np.min(incorrect_predictions)])
    plt.figure()
    set_style(fontsize=12)
    for _ , (out_scores,color,name) in enumerate([[correct_predictions,'mediumseagreen','Correct prediction'],[incorrect_predictions,'darkmagenta','Incorrect prediction']]):
        vals,bins = np.histogram(out_scores,bins = 51,density=True)
        bin_centers = (bins[1:]+bins[:-1])/2.0
        plt.plot(bin_centers,vals,linewidth=2,color=color,marker="",label=name)
        plt.fill_between(bin_centers,vals,[0]*len(vals),color=color,alpha=0.3)
    plt.xlim(min_val-(bins[1]-bins[0]),max_val+(bins[1]-bins[0]))
    plt.ylim(0)
    plt.legend()
    plt.title(str(title))
    plt.xlabel('Normalised maximum class softmax probability')
    plt.ylabel('Frequency')
    plt.grid()

    if save_dir != 0:
        plt.savefig(save_dir)

    if show_plot == True:
        plt.show()
    plt.close()


def plot_metrics(id, ood, normalized=True, title='ODD metric', save_dir=0, show_plot=False):
    """
    Plots the histograms of the ood metric for comparing OOD and ID inputs.

    Parameters
    ----------
    path_id_confidence : str
        The path to the text file containing the confidence scores of the in-distribution data.
    path_ood_confidence : str
        The path to the text file containing the confidence scores of the out-of-distribution data.
    normalized : bool, optional
        Whether the confidence scores are normalized. The default is True.
    title : str, optional
        The title of the plot. The default is 'ODD metric'.
    save_dir : str, optional
        The directory to save the plot. The default is 0.
    """
    max_val = np.max([np.max(id),np.max(ood)])
    min_val = np.min([np.min(id),np.min(ood)])

    id_norm = (id - min_val)/(max_val - min_val)
    ood_norm = (ood - min_val)/(max_val - min_val)

    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
    if not normalized:
        # use unity based normalization to also catch negative values
        pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs))
    _, _, thresholds = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0

    plt.figure()
    set_style(fontsize=12)
    for _ , (out_scores,color,name) in enumerate([[id_norm,'RoyalBlue','In distribution'],[ood_norm,'orange','Out of distribution']]):
        vals,bins = np.histogram(out_scores,bins = 51,density=True)
        bin_centers = (bins[1:]+bins[:-1])/2.0
        plt.plot(bin_centers,vals,linewidth=2,color=color,marker="",label=name)
        plt.fill_between(bin_centers,vals,[0]*len(vals),color=color,alpha=0.3)
    plt.xlim(0,1)
    plt.ylim(0)
    plt.legend()
    plt.title(str(title))
    plt.xlabel('Metric')
    plt.ylabel('Frequency')

    plt.axvline(x=thresholds[-1],color='black',linestyle='--')
    plt.grid()

    if save_dir != 0:
        plt.savefig(save_dir)

    if show_plot == True:
        plt.show()
    plt.close()
    

def get_AUROC_AUCPR(id,ood,return_fpr_tpr=False):
    """
    Calculates the AUROC, AUCPR, FPR and TPR of an ID and OOD dataset.

    Parameters
    ----------
    id : numpy array
        The confidence scores of the in-distribution data.
    ood : numpy array
        The confidence scores of the out-of-distribution data.

    Returns
    -------
    auroc : float
        The AUROC score.
    aucpr : float
        The AUCPR score.
    fpr : float
        The false positive rate.
    tpr : float
        The true positive rate.
    """
    id_l = np.ones(len(id))
    ood_l = np.zeros(len(ood))
    true_labels = np.concatenate((id_l, ood_l))
    pred_probs = np.concatenate((id, ood))
    assert(len(true_labels) == len(pred_probs))
        # use unity based normalization to also catch negative values
    pred_probs = (pred_probs - np.min(pred_probs))/(np.max(pred_probs) - np.min(pred_probs) + 1e-8)
    fpr, tpr, _ = skm.roc_curve(y_true = true_labels, y_score = pred_probs, pos_label = 1) #positive class is 1; negative class is 0
    auroc = skm.auc(fpr, tpr)
    precision, recall, _ = skm.precision_recall_curve(true_labels, pred_probs)
    aucpr = skm.auc(recall, precision)
    
    if return_fpr_tpr:
        return auroc, aucpr, fpr, tpr
    return auroc, aucpr


def set_style(fontsize=12):
    """
    Sets the style of the plots.

    Parameters
    ----------
    fontsize : int, optional
        The fontsize of the plots. The default is 20.
    """
    mpl.rcParams['font.family'] = 'sans-serif'
    #mpl.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams.update({'font.size': fontsize})


def normalise_image(img_tensor):
    # Normalize img tensor to [0, 1] range
    min_val = torch.min(img_tensor)
    max_val = torch.max(img_tensor)
    normalized_img = (img_tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_img


def loader_with_paths(idloader,oodloader):
    class ModifiedDataset(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            image, target = self.base_dataset.__getitem__(index)
            image_path = self.base_dataset.image_paths[index]
            return image, target, image_path
    
    idloader_2 = ModifiedDataset(idloader.dataset)
    oodloader_2 = ModifiedDataset(oodloader.dataset)

    #return DataLoader(dataset, args.batch_size, shuffle=args.shuffle,pin_memory=True, num_workers=args.device_count*4, prefetch_factor = args.device_count, drop_last=drop_last_batch, persistent_workers=True, worker_init_fn=worker_init_fn)
    idloader_f = DataLoader(idloader_2, batch_size=idloader.batch_size, shuffle=False,pin_memory=idloader.pin_memory,num_workers=idloader.num_workers,
                            prefetch_factor=idloader.prefetch_factor,drop_last=idloader.drop_last,persistent_workers=idloader.persistent_workers,
                            worker_init_fn=idloader.worker_init_fn)
    
    oodloader_f = DataLoader(oodloader_2, batch_size=oodloader.batch_size, shuffle=False,pin_memory=oodloader.pin_memory,num_workers=oodloader.num_workers,
                            prefetch_factor=oodloader.prefetch_factor,drop_last=oodloader.drop_last,persistent_workers=oodloader.persistent_workers,
                            worker_init_fn=oodloader.worker_init_fn)
    
    return idloader_f, oodloader_f


def get_image_micro_results(idloader,oodloader,net,verbose=False,use_cuda=True):
    ood_logits_list = []
    ood_labels_list = []
    ood_correct_list = []
    ood_predicted_list = []
    id_logits_list = []
    id_labels_list = []
    id_correct_list = []
    id_predicted_list = []
    correct, total = 0, 0
    ood_total =0
    id_total =0

    id_names = []
    ood_names = []


    l = len(idloader)
    for batch_idx, (inputs, targets,names) in enumerate(idloader):
        id_names.extend(names)
        id_labels_list.extend([int(tensor.item()) for tensor in targets])
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
        with torch.no_grad():
            _, _, _, id_total, _, id_logits_list, _,id_correct_list,id_predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,id_total,id_logits_list,id_logits_list,id_correct_list,id_predicted_list,required_grad=False,required_correct_list=True)
            
    l = len(oodloader)
    for batch_idx, (inputs, targets,names) in enumerate(oodloader):
        ood_names.extend(names)
        ood_labels_list.extend([int(tensor.item()) for tensor in targets])
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
        with torch.no_grad():
            _, _, _, ood_total, _, ood_logits_list, _,ood_correct_list,ood_predicted_list = get_softmax_score_report_accuracy(inputs,
                            targets,use_cuda,net,correct,ood_total,ood_logits_list,ood_logits_list,ood_correct_list,ood_predicted_list,required_grad=False,required_correct_list=True)

            
    return [id_names,id_logits_list,id_labels_list,id_correct_list,id_predicted_list],[ood_names,ood_logits_list,ood_labels_list,ood_correct_list,ood_predicted_list]



