import torch
import numpy as np 
from source.util.general_utils import print_progress
import math


def get_ID_activations(model, dataloader):
    # Pass data through the model
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs,_) in enumerate(dataloader):

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            batch_activations = model.head(inputs).view(inputs.size(0), -1).cpu()
            if batch_idx == 0:
                activations = batch_activations
            else:
                activations = torch.cat((activations, batch_activations), dim=0)

    return activations



def torch_quantile(tensor, q, dim=None):
    """
    Calculate the quantile for a PyTorch tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.
    q : float
        The quantile to compute, between 0 and 1 (e.g., 0.6 for 60th percentile).
    dim : int, optional
        The dimension to compute the quantile along. If None, calculates across the entire tensor.

    Returns
    -------
    torch.Tensor
        The quantile value(s).
    """
    # Check that q is a valid quantile
    if not (0 <= q <= 1):
        raise ValueError("Quantile q should be in the range [0, 1]")

    if dim is None:
        # Flatten the tensor to calculate the quantile across all values
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)

    # Calculate the index of the quantile as a tensor
    rank = q * (sorted_tensor.size(dim) - 1)
    lower_idx = math.floor(rank)  # Lower index for interpolation
    upper_idx = math.ceil(rank)   # Upper index for interpolation

    # If rank is an integer, we can directly use the indexed value
    if lower_idx == upper_idx:
        quantile_value = sorted_tensor.index_select(dim, torch.tensor([lower_idx]))
    else:
        # Gather values at the lower and upper indices for interpolation
        lower_value = sorted_tensor.index_select(dim, torch.tensor([lower_idx]))
        upper_value = sorted_tensor.index_select(dim, torch.tensor([upper_idx]))

        # Interpolate between the two values
        weight = rank - lower_idx
        quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.squeeze()  # Remove any unnecessary dimensions


def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,trainloader=None,temper=1,ReAct_percentile=0.6,OOD_dict={},**kwargs):
    """
    Evaluate ReAct score on the ID and OOD datasets.

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
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Find the last layer of the specified types
    # Calculate the 90th percentile for the activations
    activations = get_ID_activations(net, trainloader)
    percentile = torch_quantile(activations.flatten(), ReAct_percentile).item()


    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)


            with torch.no_grad():
            #Classify the inputs
                if use_cuda:
                    inputs= inputs.cuda()
                features = net.head(inputs)
                clamped_features = torch.clamp(features, max=percentile)
                out = net.apply_head(clamped_features)

            energy = temper*torch.logsumexp(out / temper, dim=1)
            energy = [t.item() for t in energy]
            confidence[OOD].extend(energy)


        OOD_dict['name'] = ['ReAct (percentile: '+str(ReAct_percentile) +')']

    return confidence


def train():
    pass



