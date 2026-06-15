import torch
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names
from source.util.general_utils import print_progress, variable_use_cuda
from sklearn.neighbors import KernelDensity



def evaluate(net, idloader, oodloader, module, OOD_dict={'name': ['kde']},**kwargs):
    """
    Evaluate KDE distance score on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader 
        The dataloader for the OOD dataset
    module: int, list, tuple, numpy array, or str
        The index of the module after which to extract embeddings and apply kde distance score. 
        If it is an array then kde distance score is applied on several modules.
        If 'all' is passed, kde distance score is applied to all modules of the network.
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the module(s) after which kde distance score is applied.

    Returns
    -------
    list
        A confidence list containing pairs of lists for each module. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    confidence = []

    #Get the modules to extract embeddings from
    if isinstance(module, int):
        if module == -1:
            modules = len(get_graph_node_names(net)[0])-1
        modules = [module]
        kwargs['feature_combination'] = False
    elif isinstance(module,str) and module=='all':
        model_modules = net.list_layers()
        modules = np.arange(0,len(model_modules))
    elif isinstance(module, (list,type(np.array([])),tuple)) == False:
        raise ValueError('module must be an integer, list, tuple, or numpy array, or "all" for all modules')
    else:
        modules = module

    
    confidence = evaluate_kde_distance(net, idloader, oodloader, modules=modules, OOD_dict=OOD_dict, **kwargs)

    return confidence



def evaluate_kde_distance(net,idloader,oodloader,modules,modules_to_skip=[],num_classes=2,KDE_kernel='gaussian',KDE_mode='kde_class',use_cuda=True, verbose=True, 
             alpha=None,feature_combination=True,trainloader=None,OOD_dict={'name': ['kde']},**kwargs):
    """
    Evaluate kde distance score on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    modules: list, tuple, numpy array
        A list of modules after which to extract embeddings and apply kde distance score.
    modules_to_skip: list, tuple, numpy array
        A list of modules to skip which are in the list modules. Default: []
    use_cuda: bool
        Whether to use cuda. Default: True
    verbose: bool
        Whether to print progress. Default: True
    alpha: list
        A list of alpha values to use for combining the kde distance scores from different modules. 
        If None, the kde distance scores are standardised using the training data. Default: None.
    feature_combination: bool
        Whether to combine the kde distance scores from the different modules into one score, or to
        output the score at each module seperately. Default: True
    trainloader: torch.utils.data.DataLoader
        The dataloader for the training dataset. Required if alpha is None and feature_combination is True. Default: None.
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the module(s) after which kde distance score is applied.

    Returns
    -------
    list
        A confidence list containing pairs of lists for each module. The first list contains the confidence scores for the ID dataset
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.eval()
    net.training = False
    if use_cuda:
        net.cuda()

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    node_names = net.list_layers()

    try:
        module_names = [node_names[k] for k in modules if k not in modules_to_skip]
    except:
        raise ValueError('modules {} is out of range. The model has {} modules.'.format(modules,len(node_names[0])))
    
    valid_kernels = ["gaussian","tophat","epanechnikov","exponential","linear","cosine"]
    if KDE_kernel not in valid_kernels:
        raise ValueError('KDE_kernel must be one of '+str(valid_kernels))

    # Accept CLI defaults KDE / KDE_class (and legacy lowercase kde / kde_class)
    kde_mode_key = str(KDE_mode).upper().replace(' ', '_')
    if kde_mode_key == 'KDE':
        KDE_mode = 'kde'
    elif kde_mode_key in ('KDE_CLASS', 'KDECLASS'):
        KDE_mode = 'kde_class'
    elif KDE_mode in ('kde', 'kde_class'):
        pass
    else:
        raise ValueError('KDE_mode must be one of "KDE" or "KDE_class"')
    
    num_sample_per_class = [np.zeros(num_classes) for _ in range(len(module_names))]
    latent_dim_data = [[0 for _ in range(num_classes)] for _ in range(len(module_names))] #Stores the embedding vector for each class

    #Extract the embedding vectors for each image from the required modules
    print("Computing the sample means and precisions for each class in the training data")
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (data, target) in enumerate(trainloader): #Iterate through each image in training data
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        data = variable_use_cuda(data,use_cuda)
        batch_size = data.size(0)
    
        with torch.no_grad():
            out_features = net.feature_extractor(data, module_names = module_names)
        for module_count, module in enumerate(out_features.keys()):

            out_features[module] = out_features[module].view(out_features[module].size(0), out_features[module].size(1), -1) 
            out_features[module] = torch.mean(out_features[module].data, 2)
            # construct the sample matrix of embedding vectors for each class
            for i in range(batch_size): 
                if num_sample_per_class[module_count][target[i]] == 0: #If there are no samples for the class in this module
                    latent_dim_data[module_count][target[i]] = out_features[module][i].view(1, -1) 
                else:
                    latent_dim_data[module_count][target[i]] \
                            = torch.cat((latent_dim_data[module_count][target[i]], out_features[module][i].view(1, -1)), 0)
                num_sample_per_class[module_count][target[i]] += 1

        del out_features

    kde_models = {}  # Dictionary to store kde models

    if KDE_mode == 'kde_class':
        for module_count, module in enumerate(module_names):
            kde_models[module] = {}  # Initialise a sub-dictionary for each module
            
            for class_idx in range(num_classes):
                all_features = latent_dim_data[module_count][class_idx].cpu().numpy()
                dim = all_features.shape[1]      
                kde = KernelDensity(kernel=KDE_kernel, bandwidth=0.5)
                kde.fit(all_features)  # Fit kde model for the class

                # Store the kde model in the dict under the module and class_idx
                kde_models[str(module)][class_idx] = kde


    elif KDE_mode == 'kde':
        for module_count, module in enumerate(module_names):
            all_features = np.concatenate([latent_dim_data[module_count][i].cpu().numpy() for i in range(num_classes)], axis=0)   
            kde_models[str(module)] = KernelDensity(kernel=KDE_kernel, bandwidth=0.5)
            kde_models[str(module)].fit(all_features)

    with torch.no_grad():
        #Calculate the kde metric for the ID and OOD data
        conf_list = [[[] for _ in range(len(module_names))] for _ in range(2)]
        for OOD, loader in enumerate([idloader, oodloader]):
            if verbose==True:
                print('Evaluating '+['ID','OOD'][OOD]+' dataset')

            
            l = len(loader)
            print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
            for batch_idx, (inputs, target) in enumerate(loader):

                features_total = net.feature_extractor(inputs.cuda(), module_names = module_names)
                print_progress(batch_idx + 1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)


                for module_count, module in enumerate(module_names):
                    features = features_total[str(module)]
                    features = features.view(features.size(0), features.size(1), -1)
                    feature_means = torch.mean(features, dim=2)
                    feature_means = feature_means.cpu().numpy()
                    dim = feature_means.shape[1]

                    if KDE_mode == 'kde_class':
                        kde_class_score = [[] for _ in range(num_classes)]
                        for class_idx in range(num_classes):
                                kde_class_score[class_idx] = kde_models[str(module)][class_idx].score_samples(feature_means)
                                kde_class_score[class_idx] = np.clip(kde_class_score[class_idx],-1e10,1e10)  # Avoid -inf by setting a floor and ceiling
                        kde_scores = -1*np.min(kde_class_score, axis=0)

                    elif KDE_mode == 'kde':
                        kde_scores = kde_models[str(module)].score_samples(feature_means)
                        kde_scores = np.clip(kde_scores,-1e10,1e10)  # Avoid -inf by setting a floor and ceiling

                    conf_list[OOD][module_count].extend(kde_scores.tolist())



    OOD_dict['name'] = []
    for module in module_names:
        OOD_dict['name'].append(str(KDE_mode)+' (Kernel '+str(KDE_kernel)+', module '+str(module)+')')




    return list(map(list, zip(*conf_list)))




def format_modules(modules):
    """
    Format the list of modules to apply kde distance to a consise format.

    Parameters
    ----------
    modules : list
        List of modules to apply kde distance to.

    Returns
    -------
    str
        Formatted string of modules.
    """
    ranges = []
    skipped_modules = []
    start = modules[0]
    end = modules[0]

    for module in modules[1:]:
        if module == end + 1:
            end = module
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            for skipped_module in range(end+1, module):
                skipped_modules.append(skipped_module)
            start = end = module

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    formatted_modules = ', '.join(ranges)
    modules_skipped_string = (f"{modules[0]}-{modules[-1]}, skipped {','.join(map(str,skipped_modules))}")
    if len(formatted_modules) < len(modules_skipped_string):
        return formatted_modules
    else:
        return modules_skipped_string

