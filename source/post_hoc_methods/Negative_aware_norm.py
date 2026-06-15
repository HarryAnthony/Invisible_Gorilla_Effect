import torch
from source.util.general_utils import print_progress


# Define a helper function to compute NAN score
def compute_nan_score(activations):
    # Compute L1 norm
    if activations.dim() == 4:
        activations = activations.view(activations.size(0), -1)
    l1_norm = activations.abs().sum(dim=1)

    # Compute sparsity term (inverse of count of non-zero elements)
    non_zero_count = (activations > 0).sum(dim=1).float()
    sparsity_term = 1 / (non_zero_count + 1e-6)  # Add small value to avoid division by zero

    # NAN score is the product of L1 norm and inverse sparsity
    nan_score = l1_norm * sparsity_term
    #nan_score
    return nan_score.flatten()

def evaluate(net, idloader, oodloader, NaN_module='last_hidden_layer', OOD_dict={}, use_cuda=True, verbose=True, **kwargs):
    """
    Evaluate Negative-Aware Norm (NAN) on ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    use_cuda: bool
        Whether to use CUDA. Default: True
    verbose: bool
        Whether to print progress. Default: True

    Returns
    -------
    list
        A list containing two lists: NAN scores for the ID dataset and for the OOD dataset.
    """
    net.eval()
    confidence = [[],[]]


    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if NaN_module == 'all':
        #input('here')
        layer_list = net.list_layers()
        confidence = [[[] for _ in range(2)] for _ in range(len(layer_list))]
        OOD_dict['name'] = []
        for layer_idx, layer in enumerate(layer_list):
            OOD_dict['name'].append('Negative aware norm (module '+str(layer) +')')

    
    # Evaluate on both ID and OOD loaders
    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)


            if use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                if NaN_module == 'last_hidden_layer':
                    # Forward pass through the network to get to the last hidden layer
                    activations = net.head(inputs)
                    # Compute NAN scores for the batch
                    nan_scores = compute_nan_score(activations)
                    # Append results to the correct list (ID or OOD)
                    confidence[OOD].extend(nan_scores.cpu().numpy())
                else:
                    activation_dict = net.get_all_activations(inputs)

                    if NaN_module in activation_dict.keys():
                        nan_scores = compute_nan_score(activation_dict[NaN_module])
                        confidence[OOD].extend(nan_scores.cpu().numpy())
                    elif NaN_module == 'all':
                        for layer_idx, key in enumerate(activation_dict.keys()):
                            nan_scores = compute_nan_score(activation_dict[key])
                            confidence[layer_idx][OOD].extend(nan_scores.cpu().numpy())
                            if batch_idx == 0 and OOD == 0:
                                OOD_dict['name'].append('Negative aware norm (layer: '+str(key) +')')
                    else:
                        raise ValueError(f'Layer {NaN_module} not found in the model.')
                    

    return confidence
