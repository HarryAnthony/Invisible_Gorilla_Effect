import torch
from source.util.general_utils import print_progress
import torch.nn as nn


def get_ID_activations(model, dataloader):
    # Pass data through the model
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs,_) in enumerate(dataloader):

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            batch_activations = model.head(inputs).cpu()
            if batch_idx == 0:
                activations = batch_activations
            else:
                activations = torch.cat((activations, batch_activations), dim=0)

    if activations.dim() == 4:
                activations = activations.squeeze(1)  # Remove the extra dimension added by the head method
                activations = activations[:, 0]

    return activations


def calculate_contribution_matrix(weights, activations):
    """
    Calculate the contribution matrix V based on weights and activations.

    Parameters:
    weights (torch.Tensor): The weight matrix of the final layer. Shape: [C, m]
    activations (torch.Tensor): The activations from the preceding layer. Shape: [N, m], 
                                where N is the number of samples in the dataset.

    Returns:
    torch.Tensor: The contribution matrix V. Shape: [m, C]
    """
    # Ensure weights and activations are on the same device (CPU or GPU)
    weights = weights.cuda()
    activations = activations.cuda()
    
    # Correct approach for element-wise multiplication and averaging
    # Broadcasting weights to [N, C, m] and activations to [N, m, 1] for element-wise multiplication
    V = torch.einsum('cm,nm->nmc', weights, activations)

    # Averaging across the samples (N), resulting in a shape of [m, C]
    V_mean = V.mean(dim=0).transpose(0, 1)

    return V_mean


def create_masking_matrix(V, p):
    """
    Create a masking matrix M based on the top-k largest elements in V.

    Parameters:
    V (torch.Tensor): The contribution matrix of shape [m, C].
    p (float): The sparsity parameter indicating the fraction of weights to be dropped.

    Returns:
    torch.Tensor: The masking matrix M of the same shape as V.
    """
    m, C = V.shape
    k = int((1 - p) * m * C)  # Calculate the number of elements to keep based on sparsity parameter p

    # Flatten V and find the k-th largest value
    V_flattened = V.flatten()
    top_k_values, _ = torch.topk(V_flattened, k)
    kth_largest_value = top_k_values[-1]  # The k-th largest value serves as a threshold

    # Create the mask: 1 for top-k elements, 0 otherwise
    M = (V >= kth_largest_value).float()

    return M



def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,trainloader=None,temper=1,DICE_sparsity_parameter=0.75,OOD_dict={},**kwargs):
    """
    Evaluate DICE score on the ID and OOD datasets.

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

    
    original_weights, _ = net.get_head_weights()
    activations = get_ID_activations(net, trainloader)

    V = calculate_contribution_matrix(original_weights, activations)
    M = create_masking_matrix(V, DICE_sparsity_parameter)
    net.set_head_weights(original_weights * M)


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
                out = net(inputs)

            energy = temper*torch.logsumexp(out / temper, dim=1)
            energy = [t.item() for t in energy]
            confidence[OOD].extend(energy)


    net.set_head_weights(original_weights)

    OOD_dict['name'] = ['DICE (sparsity parameter: '+str(DICE_sparsity_parameter) +')']


    return confidence


def train():
    pass



