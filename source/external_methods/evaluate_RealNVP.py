import torch
import numpy as np 
from source.util.general_utils import print_progress
from source.util.mahal_utils import  estimate_mean_precision
from source.external_methods.external_models.iResNet import RealNVP


    # Ensure symmetric covariance
def compute_whitening_matrix(covariance, eps=1e-15):
    if covariance.dim() == 3 and covariance.shape[0] == 1:
        covariance = covariance.squeeze(0)  

    covariance = (covariance + covariance.T) / 2.0

    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(covariance)

    if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
        raise ValueError("NaN or Inf detected in eigenvalues.")

    eigvals_clamped = torch.clamp(eigvals, min=eps)
    # Inverse sqrt of eigenvalues
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals_clamped))
    log_abs_det_A_inv = -0.5 * torch.sum(torch.log(eigvals_clamped))


    # Whitening matrix: A_inv = D^{-1/2} @ Q.T
    A_inv = D_inv_sqrt @ eigvecs.T

    return A_inv, log_abs_det_A_inv


def create_alternating_masks(num_features, num_masks):
    """
    Create alternating binary masks for RealNVP coupling layers.

    Args:
        num_features (int): Dimensionality of the input feature vector.
        num_masks (int): Number of masks to generate (i.e., flow layers).

    Returns:
        masks (torch.Tensor): Tensor of shape (num_masks, num_features)
    """
    half = num_features // 2

    # Build base masks
    mask_right = torch.cat([torch.zeros(half), torch.ones(num_features - half)])
    mask_left = 1.0 - mask_right

    # Alternate masks: [right, left, right, left, ...]
    base_masks = [mask_right, mask_left]
    masks = [base_masks[i % 2].clone() for i in range(num_masks)]

    return torch.stack(masks).cuda()



def evaluate(net, idloader, oodloader, use_cuda=True, verbose=True, module=10, RealNVP_load_dirs=['class_1_RealNVP.pth', 'class_2_RealNVP.pth'], save_dir=None,filename=None, trainloader=None, num_classes=3, **kwargs):
    """
    Evaluate RealNVP on the ID and OOD datasets.

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
    verbose: bool*
        Whether to print progress. Default: True

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    net.eval()
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = net.cuda()

    l = len(trainloader)
    print_progress(0, l, prefix='Progress:', suffix='Complete', length=50)

    layer_idx_ind = module
    confidence= [[],[]]
    module_names = [net.list_layers()[layer_idx_ind]]
    training_data_statistics = estimate_mean_precision(module_names=module_names,net=net,trainloader=trainloader,num_classes=num_classes,RMD=True)

    #Calculate training statistics
    A_inv_dict = {}
    A_dict = {}
    log_abs_det_A_inv = {}

    for _, module in enumerate(module_names):
        A_inv_dict[module], log_abs_det_A_inv[module] = compute_whitening_matrix(torch.linalg.pinv(training_data_statistics['total_precision_list'][0]))
        A_dict[module] = torch.linalg.pinv(A_inv_dict[module])

    num_sample_per_class = [np.zeros(num_classes) for _ in range(len(module_names))]
    latent_dim_data = [[0 for _ in range(num_classes)] for _ in range(len(module_names))] #Stores the embedding vector for each class

    for inputs, target in trainloader:
        with torch.no_grad():
                out_features = net.feature_extractor(inputs.cuda(), module_names = module_names)
                for module_count, module in enumerate(out_features.keys()):

                    out_features[module] = out_features[module].view(out_features[module].size(0), out_features[module].size(1), -1) 
                    out_features[module] = torch.mean(out_features[module].data, 2)
                    # construct the sample matrix of embedding vectors for each class
                    for i in range(len(target)): 
                        if num_sample_per_class[module_count][target[i]] == 0: #If there are no samples for the class in this module
                            latent_dim_data[module_count][target[i]] = out_features[module][i].view(1, -1) 
                        else:
                            latent_dim_data[module_count][target[i]] \
                                    = torch.cat((latent_dim_data[module_count][target[i]], out_features[module][i].view(1, -1)), 0)
                        num_sample_per_class[module_count][target[i]] += 1

                del out_features

    datasets = {}
    for module_idx, module in enumerate(module_names):
        datasets[module] = {}
        for c in range(num_classes):
            centred_features = torch.tensor(latent_dim_data[module_idx][c] - training_data_statistics['mean_list'][module_idx][c], dtype=torch.float32)
            datasets[module][c] = torch.utils.data.DataLoader(
                centred_features, batch_size=2, shuffle=True
            )

        mask_list = create_alternating_masks(centred_features.shape[1],num_masks=10)
        flows = []

        #Load class conditional Flow-based models
        for c in range(num_classes):
            flow = RealNVP(mask=mask_list, 
                        num_features=centred_features.shape[1], 
                        length_hidden=1,#length_hidden, 
                        A=A_dict[module], 
                        A_inv=A_inv_dict[module], 
                        log_abs_det_A_inv=log_abs_det_A_inv[module])
            
            checkpoint = torch.load(RealNVP_load_dirs[c],map_location='cpu')
            #Apply parameters and activation function to the network
            params = {}
            for k_old in checkpoint.keys():
                k_new = k_old.replace('module.', '')
                params[k_new] = checkpoint[k_old]
            flow.load_state_dict(params)
            flows.append(flow.cuda())


    module_names = [net.list_layers()[layer_idx_ind]]
    module_name = module_names[0]

    #OOD scoring function = max likelihood
    for OOD,(loader) in enumerate([idloader,oodloader]):

        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets, path) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            with torch.no_grad():
                validation_outputs = net.feature_extractor(inputs.cuda(), module_names = module_names)[module_name]
                validation_outputs = validation_outputs.view(validation_outputs.size(0), validation_outputs.size(1), -1) 
                validation_outputs = torch.mean(validation_outputs.data, 2)
                confidence[OOD].extend(np.max([flows[c].log_prob(torch.tensor(validation_outputs - training_data_statistics['mean_list'][0][c], dtype=torch.float32),training=False).detach().cpu().numpy() for c in range(num_classes)],axis=0))


    return confidence
