import torch
import numpy as np
from source.util.general_utils import print_progress



def evaluate(net, idloader, oodloader, use_cuda=True, NMD_per_layer=False, trainloader=None, OOD_dict={}, verbose=True, **kwargs):
    """
    Evaluate NMD distance score on the ID and OOD datasets.
    """

    net.eval()
    net.training = False
    if use_cuda:
        net.cuda()

    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    module_names = net.list_layers()
    if NMD_per_layer == 'True':
        conf_list = [[[] for _ in range(len(module_names))] for _ in range(2)]
    else:
        confidence = [[],[]]
    spatial_sums = {module: [] for module in module_names}  # Store tensors as lists first

    for inputs, _ in trainloader:
        out_features = net.feature_extractor(inputs.cuda(), module_names=module_names)

        for module in module_names:
            if out_features[module].dim() == 4:
                spatial_sums[module].append(out_features[module].mean(dim=[2, 3]).detach())  # Spatial mean
            elif out_features[module].dim() == 3:
                spatial_sums[module].append(out_features[module].mean(dim=[2]).detach())  # Spatial mean
            else:
                spatial_sums[module].append(out_features[module].detach())  # Regular mean

    # Convert lists to tensors
    for module in module_names:
        spatial_sums[module] = torch.vstack(spatial_sums[module])  # (Total_samples, Channels)
    train_neural_means = {module: spatial_sums[module].mean(dim=0) for module in module_names}


    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)

    
            out_features = net.feature_extractor(inputs.cuda(), module_names=module_names)
            neural_mean_discrepancy = []

            for module_idx,module in enumerate(module_names):
                if out_features[module].dim() == 4:
                    test_means = out_features[module].mean(dim=[2, 3]).detach()
                elif out_features[module].dim() == 3:
                    test_means = out_features[module].mean(dim=[2]).detach()
                else:
                    test_means = out_features[module].detach()

                if NMD_per_layer == 'True':
                    conf_list[OOD][module_idx].extend(torch.sum(test_means - train_neural_means[module],dim=1).detach().tolist())
                else:
                    neural_mean_discrepancy.append(torch.sum(test_means - train_neural_means[module],dim=1).detach().tolist())

            if NMD_per_layer == False:
               confidence[OOD] = np.mean(neural_mean_discrepancy,axis=0)

            
    if NMD_per_layer == 'True':
        OOD_dict['name'] = []
        for module in module_names:
            OOD_dict['name'].append('NMD (module='+str(module)+')')
        return list(map(list, zip(*conf_list)))
    else:
        return confidence
    
    
        



                






    
