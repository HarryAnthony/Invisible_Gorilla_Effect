import torch
import torch.nn.functional as F
from source.util.general_utils import print_progress
import torch


def compute_inner_abnormalities(net, inputs, last_2d_layer,  use_cuda=True):
    """
    Computes the gradient of the final feature map A_last w.r.t. intermediate feature maps A_k.
    
    Args:
        net (torch.nn.Module): The neural network model.
        inputs (torch.Tensor): The input batch (e.g., images).
        module_names (list): List of layer names to track feature maps.
        use_cuda (bool): Whether to use GPU.
    
    Returns:
        dict: A dictionary containing the gradients for each module.
    """
    net.eval()
    if use_cuda:
        inputs = inputs.cuda()

    # Forward pass to extract features
    out_features = net.feature_extractor(inputs, module_names=last_2d_layer)
    
    # Get the final feature map (A_last)
    A_last = out_features[last_2d_layer[-1]]  # Last feature map before classification

    # Compute gradients w.r.t. intermediate feature maps
    gradients = []
    for module_idx, module in enumerate(last_2d_layer[:-1]):  # Exclude final layer
        A_k = out_features[module]
        grad = torch.autograd.grad(A_last.sum(), A_k, retain_graph=True)[0]  # Compute ∇Ak
        inner_grad = grad.detach()
        if inner_grad.dim() == 4:
            gradients.append(inner_grad.mean(dim=[2, 3]).abs())
        else:
            gradients.append(inner_grad.mean(dim=[2]).abs())

    return gradients


def compute_output_abnormalities(net, out_features, inputs,layers_2d, module_names):

    important_modules = [layers_2d[-1] , module_names[-1]]
    out_features = net.feature_extractor(inputs.cuda(), module_names=important_modules)
    out_features[layers_2d[-1]].retain_grad()
    out_features[module_names[-1]].retain_grad()


    output = out_features[module_names[-1]]
    softmax_output = torch.sum(torch.log_softmax(output, dim=1))  
    # Specify grad_outputs to match the shape of the output tensor
    grad_outputs = torch.ones_like(softmax_output)  # Gradient multiplier for each sample

    # Compute gradients for each sample
    grad = torch.autograd.grad(
        outputs=softmax_output, inputs=out_features[layers_2d[-1]], grad_outputs=grad_outputs,retain_graph=True)[0]
    
    output_grad = grad.detach()

    if output_grad.dim() == 4:
        output_grad = torch.mean(output_grad, dim=[1,2,3]).abs()
    else:
        output_grad = torch.mean(output_grad, dim=[1,2]).abs()

    return torch.sqrt(output_grad).unsqueeze(1)


def compute_zero_deflation_abnormality(net, inputs, layers_2d, module_names, use_cuda=True):

    layers_2d.append(module_names[-1])
    out_features = net.feature_extractor(inputs.cuda(),module_names = layers_2d)
    for module in layers_2d:
        out_features[module].retain_grad()
    output = out_features[module_names[-1]]
    softmax_output = torch.sum(torch.log_softmax(output, dim=1))  # Apply log softmax

    # Backprop with correct shape
    net.zero_grad()
    softmax_output.backward()#gradient=grad_out)

    zero_deflation_abnormality = []
    for module in layers_2d[:-1]:
        non_zero_grads = (out_features[module].grad.abs() > 0).float()
        zero_deflation_abnormality.append(torch.mean(non_zero_grads, dim=[2,3]))
        
    return zero_deflation_abnormality


def evaluate(net, idloader, oodloader, use_cuda=True, verbose=True, GAIA_mode='GAIA-Z', OOD_dict={}, **kwargs):
    """
    Evaluate GAIA score on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate.
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset.
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset.
    use_cuda: bool
        Whether to use CUDA. Default: True.
    verbose: bool
        Whether to print progress. Default: True.
    GAIA_mode: str
        The GAIA mode to use. Must be either 'GAIA-A' or 'GAIA-Z'. Default: 'GAIA-A'.

    Returns
    -------
    list
        A confidence list containing two lists: GAIA scores for the ID and OOD datasets.
    """
    net.eval()
    confidence = [[], []]  # Scores for ID and OOD datasets

    module_names = net.list_layers()  # List of layers in the network


    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose:
            print(f"Evaluating {'ID' if OOD == 0 else 'OOD'} dataset")

        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50, verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix='Progress:', suffix='Complete', length=50, verbose=verbose)


            out_features = net.feature_extractor(inputs.cuda(), module_names=module_names)
            layers_2d = []
            for module in module_names:
                if out_features[module].dim() == 4 and out_features[module].shape[-1] != 1:
                    layers_2d.append(module)

            if len(layers_2d) < 2:
                layers_2d = []
                for module in module_names:
                    if out_features[module].dim() == 3 and out_features[module].shape[-1] != 1:
                        layers_2d.append(module)


            if use_cuda:
                inputs = inputs.cuda()

            if GAIA_mode == 'GAIA-A':
                inner_abnormalities = compute_inner_abnormalities(net, inputs, layers_2d, use_cuda=True)
                output_abnormalities = compute_output_abnormalities(net, out_features, inputs, layers_2d, module_names)
                abnormalities = []
                for i in range(0,len(layers_2d)-1):
                    abnormalities.append(inner_abnormalities[i] / (output_abnormalities+1e-8))
            elif GAIA_mode == 'GAIA-Z':
                abnormalities = compute_zero_deflation_abnormality(net,inputs,layers_2d,module_names, use_cuda=True)

            else:
                raise Exception("GAIA_mode must be either GAIA-A or GAIA-Z")
            
            # Find the max number of channels across layers
            max_channels = max(abnormality.shape[1] for abnormality in abnormalities)  

            # Pad each layer's abnormality tensor and store them
            padded_abnormalities = []
            for abnormality in abnormalities:
                num_channels = abnormality.shape[1]
                pad_size = max_channels - num_channels
                padded_abnormality = torch.nn.functional.pad(abnormality, (0, pad_size))  # Pad missing channels with 0
                padded_abnormalities.append(padded_abnormality)

            # Stack into a final abnormality matrix Λ (Shape: [batch_size, num_layers, max_channels])
            Λ = torch.stack(padded_abnormalities, dim=1)  # Shape: [batch_size, num_layers, max_channels]

            frobenius_norm = -1* torch.norm(Λ, p='fro', dim=[1, 2])  # Frobenius norm along (layers, channels)
            confidence[OOD].extend(frobenius_norm.detach().tolist())


    if GAIA_mode == 'GAIA-A':
        OOD_dict['name'] = ['GAIA-A']
    else:
        OOD_dict['name'] = ['GAIA-Z']

    return confidence
