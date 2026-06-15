import numpy as np
import torch
from source.util.general_utils import print_progress, variable_use_cuda


def estimate_eigenvalues(data,class_means,Dim=10,**kwargs):
    class_eigens = []
    for target_class in range(len(data)):
        delta = data[target_class] - class_means[target_class]
        cov_matrix = torch.matmul(delta.T, delta) / delta.size(0)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        principal_components = eigenvectors[:, -Dim:]  # Last D eigenvectors
        class_eigens.append(principal_components)
    return class_eigens


def get_class_means(data,RMD=False):
    '''
    Calculates the mean of each class in the data.

    Parameters
    ----------
    data : list
        List of tensors containing the data for each class.
    RMD : bool, optional
        If True, the mean of all classes is calculated (required for Relative Residual Distance). The default is False.

    Returns
    -------
    class_means : tensor
        Tensor containing the mean of each class.
    total_mean : tensor
        Tensor containing the mean of all classes (only returned if RMD is True).
    '''
    for target_class in range(len(data)):
        if target_class == 0:
            class_means = torch.mean(data[target_class], 0).view(1,-1)
        else:
            class_means = torch.cat((class_means,torch.mean(data[target_class],0).view(1,-1)),0)
    if RMD: #RMD requires calculating the mean for every latent dimension regardless of class
        total_mean = torch.mean(class_means,axis=0)
        return class_means, total_mean
    return class_means



def evaluate(net, idloader, oodloader, module, OOD_dict={'name': ['Residual']},**kwargs):
    """
    Evaluate Residual distance score on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader 
        The dataloader for the OOD dataset
    module: int, list, tuple, numpy array, or str
        The index of the module after which to extract embeddings and apply Residual distance score. 
        If it is an array then Residual distance score is applied on several modules.
        If 'all' is passed, Residual distance score is applied to all modules of the network.
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the module(s) after which Residual distance score is applied.

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
            modules = len(net.list_layers())-1
        modules = [module]
        kwargs['feature_combination'] = False
    elif isinstance(module,str) and module=='all':
        model_modules = net.list_layers()
        modules = np.arange(0,len(model_modules))
    elif isinstance(module, (list,type(np.array([])),tuple)) == False:
        raise ValueError('module must be an integer, list, tuple, or numpy array, or "all" for all modules')
    else:
        modules = module
    
    confidence = evaluate_Residual_distance(net, idloader, oodloader, modules=modules, OOD_dict=OOD_dict, **kwargs)

    return confidence



def evaluate_Residual_distance(net,idloader,oodloader,modules,modules_to_skip=[],num_classes=2,Feature_based_Dim=20,use_cuda=True, verbose=True, 
             trainloader=None,OOD_dict={'name': ['Residual']},**kwargs):
    """
    Evaluate Residual on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    use_cuda: bool
        Whether to use cuda. Default: True
    verbose: bool
        Whether to print progress. Default: True

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.eval()

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

    class_eigens_total = []


    for module_count, module in enumerate(module_names):
        class_means = get_class_means(latent_dim_data[module_count])
        class_eigens = estimate_eigenvalues(latent_dim_data[module_count], class_means, Dim=int(Feature_based_Dim), **kwargs)
        class_eigens_total.append(class_eigens)

    def compute_residual(features,principal_components):
        projection = features @ principal_components @ principal_components.T
        residual = features - projection
        return -1*torch.linalg.norm(residual, axis=1)
    

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

                    for class_idx in range(num_classes):
                        residual_score = compute_residual(feature_means, class_eigens_total[module_count][class_idx])
                        conf_list[OOD][module_count].extend(residual_score.cpu().numpy().tolist())


    OOD_dict['name'] = []
    for module in module_names:
        OOD_dict['name'].append('Residual (Dimensionality '+str(Feature_based_Dim)+', module '+str(module)+')')

    return list(map(list, zip(*conf_list)))




