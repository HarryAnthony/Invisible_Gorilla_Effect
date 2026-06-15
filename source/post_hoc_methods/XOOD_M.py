import torch
import numpy as np
from source.util.general_utils import print_progress
from sklearn.preprocessing import PowerTransformer


def extract_extreme_values(out_features):
    """
    Extracts Xmin and Xmax from the feature maps of a neural network, 
    handling both 2D (FC layers) and 4D (CNN layers) activations.
    
    Args:
        out_features (dict): Dictionary where keys are layer names and values are feature maps (tensors).
    
    Returns:
        dict: Dictionary containing Xmin and Xmax for each layer.
    """
    Xmin_dict = {}
    Xmax_dict = {}

    for layer_name, features in out_features.items():
        if features.dim() == 4:  # CNN layers: [batch_size, channels, height, width]
            Xmin = features.amin(dim=(1, 2, 3))  # Min across spatial dimensions
            Xmax = features.amax(dim=(1, 2, 3))  # Max across spatial dimensions
        elif features.dim() == 3:
            Xmin = features.amin(dim=(1, 2))  # Min across spatial dimensions
            Xmax = features.amax(dim=(1, 2))  # Max across spatial dimensions
        elif features.dim() == 2:  # Fully Connected layers: [batch_size, features]
            Xmin = features.amin(dim=1, keepdim=True)  # Min over feature dimension
            Xmax = features.amax(dim=1, keepdim=True)  # Max over feature dimension
        else:
            raise ValueError(f"Unexpected feature shape {features.shape} in layer {layer_name}")

        # Store numpy arrays for later accumulation
        Xmin_dict[layer_name] = Xmin.detach().cpu().numpy().flatten()
        Xmax_dict[layer_name] = Xmax.detach().cpu().numpy().flatten()

    return Xmin_dict, Xmax_dict


def fit_power_transformers(Xmin_dict_total, Xmax_dict_total):
    """
    Fit one PowerTransformer per layer by concatenating Xmin and Xmax.
    
    Args:
        Xmin_dict_total (dict): Dictionary of accumulated Xmin per layer.
        Xmax_dict_total (dict): Dictionary of accumulated Xmax per layer.
    
    Returns:
        dict: Dictionary of fitted PowerTransformers per layer.
        dict: Dictionary of transformed extreme values per layer.
    """
    transformers = {}
    transformed_features = {}

    for layer_name in Xmin_dict_total.keys():
        # Stack all collected extreme values for this layer and concatenate Xmin and Xmax
        Xmin_all = np.vstack(Xmin_dict_total[layer_name])
        Xmax_all = np.vstack(Xmax_dict_total[layer_name])
        feature_matrix = np.hstack([Xmin_all, Xmax_all])  # Concatenating min and max values

        # Fit a single PowerTransformer per layer on concatenated features
        transformer = PowerTransformer(method='yeo-johnson')
        transformed_features[layer_name] = transformer.fit_transform(feature_matrix)

        # Store the transformer for later use
        transformers[layer_name] = transformer

    return transformers, transformed_features


def calculate_mean_covariance(transformed_features, C=0.1):
    """
    Calculate the mean and covariance matrix for transformed extreme values per layer.
    
    Args:
        transformed_features (dict): Dictionary of transformed extreme values per layer.
    
    Returns:
        dict: Dictionary of mean vectors per layer.
        dict: Dictionary of covariance matrices per layer.
    """
    mean_dict = {}
    precision_dict = {}

    for module, feature_matrix in transformed_features.items():
        # Compute mean and covariance per layer
        mean_vector = np.mean(feature_matrix, axis=0)
        covariance_matrix = np.cov(feature_matrix, rowvar=False)

        # Store results
        mean_dict[module] = mean_vector
        identity_matrix = np.eye(covariance_matrix.shape[0])
        M_reg = covariance_matrix + C * identity_matrix
        precision_dict[module] = np.linalg.pinv(M_reg)


    return mean_dict, precision_dict


def evaluate(net, idloader, oodloader, use_cuda=True, XOOD_M_C=1, XOOD_M_ensemble=False, Feature_based_modules=None,trainloader=None, OOD_dict={}, verbose=True, **kwargs):
    """
    Evaluate XOOD-M distance score on the ID and OOD datasets.
    
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
    net.training = False
    if use_cuda:
        net.cuda()

    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    module_names = net.list_layers()
    if Feature_based_modules != None:
        module_names = [module_names[i] for i in Feature_based_modules]

    conf_list = [[[] for _ in range(len(module_names))] for _ in range(2)]

    # Store all Xmin and Xmax values across training data
    Xmin_dict_total = {layer: [] for layer in module_names}
    Xmax_dict_total = {layer: [] for layer in module_names}

    # Pass through the entire training dataset to collect extreme values
    for batch_idx, (inputs, target) in enumerate(trainloader):
        out_features = net.feature_extractor(inputs.cuda(), module_names=module_names)
        Xmin_dict, Xmax_dict = extract_extreme_values(out_features)

        # Accumulate across all batches
        for layer_name in module_names:
            Xmin_dict_total[layer_name].extend(Xmin_dict[layer_name])
            Xmax_dict_total[layer_name].extend(Xmax_dict[layer_name])

    transformers, transformed_features = fit_power_transformers(Xmin_dict_total,Xmax_dict_total)
    mean_dict, precision_dict = calculate_mean_covariance(transformed_features,C=XOOD_M_C)


    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)

            out_features = net.feature_extractor(inputs.cuda(), module_names=module_names)
            Xmin_test, Xmax_test = extract_extreme_values(out_features)

            XOOD_list = []

            for module_idx,module in enumerate(module_names):
                
                x_test = np.hstack([np.vstack(Xmin_test[module]), np.vstack(Xmax_test[module])])
                x_test_transformed = transformers[module].transform(x_test)
                delta = x_test_transformed - mean_dict[module]
                M_inv = precision_dict[module]  # Inverse of the regularized covariance matrix

                gauss_score = np.matmul(np.matmul(delta, M_inv),delta.T)
                mahalanobis_distance = -1*np.sqrt(np.diagonal(gauss_score))
                XOOD_list.append(mahalanobis_distance)


    OOD_dict['name'] = []
    for module in module_names:
        OOD_dict['name'].append('XOOD-M (C='+str(XOOD_M_C)+', module='+str(module)+')')
    return list(map(list, zip(*conf_list)))


