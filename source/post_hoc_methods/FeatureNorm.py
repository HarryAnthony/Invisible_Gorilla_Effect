import torch
from source.util.general_utils import print_progress
import torch
import torch.nn.functional as F


def compute_feature_norm(feature_map):
    """
    Compute the FeatureNorm score from a feature map.
    
    Args:
        feature_map (torch.Tensor): Extracted feature map (B, C, H, W).
    
    Returns:
        torch.Tensor: FeatureNorm scores for each sample in batch.
    """
    # Apply ReLU to retain only positive activations
    feature_map = F.relu(feature_map)

    if feature_map.dim() == 4:
    # Compute Frobenius norm for each channel
        channel_norms = torch.sqrt(torch.sum(feature_map ** 2, dim=[2, 3]))  # (B, C)
    elif feature_map.dim() == 3:
    # Compute Frobenius norm for each channel
        channel_norms = torch.sqrt(torch.sum(feature_map ** 2, dim=[2]))  # (B, C)
    else:
        channel_norms = feature_map**2
        #channel_norms = torch.sqrt(torch.sum(feature_map ** 2, dim=[-1]))

    # Compute the average across all channels
    feature_norm_scores = torch.mean(channel_norms, dim=1)  # (B,)
    return feature_norm_scores


def evaluate(net, idloader, oodloader, use_cuda=True, Feature_based_modules=None, OOD_dict={}, verbose=True, **kwargs):
    """
    Evaluate Feature Norm distance score on the ID and OOD datasets.

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

    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)


            out_features = net.feature_extractor(inputs.cuda(),module_names=module_names)
            for module_idx, module in enumerate(module_names):
                FeatureNorm_scores = compute_feature_norm(out_features[module])
                conf_list[OOD][module_idx].extend(FeatureNorm_scores.detach().cpu().tolist())


    OOD_dict['name'] = []
    for module in module_names:
        OOD_dict['name'].append('FeatureNorm (module='+str(module)+')')


    return list(map(list, zip(*conf_list)))
        

