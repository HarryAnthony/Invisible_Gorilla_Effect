import torch
import torch.nn.functional as F
import torch
from source.util.general_utils import print_progress
import torch.nn.functional as F


def calculate_kl_divergence(logits):
    softmax_probs = F.softmax(logits, dim=1)
    uniform_probs = torch.ones_like(softmax_probs) / (softmax_probs.size(1))
    softmax_probs = softmax_probs.clamp_min(1e-12)
    kl_div = F.kl_div(softmax_probs.log(), uniform_probs, reduction='sum')
    softmax_probs.retain_grad()
    return kl_div


def gradnorm(model,data,grad_layers,gradnorm_summation_method='l2'):
    """
    Implement GradNorm for OOD detection.
    Args:
    model (torch.nn.Module): The neural network model.
    data (torch.Tensor): The input data.
    Returns:
    float: The L1-norm of the gradients from the model's output layer.
    """
    grad_norms = [[] for _ in grad_layers]
    
    for i in range(data.size(0)):  # Iterate over each image in the batch
        # Select the ith image
        image = data[i].unsqueeze(0)  # Add batch dimension
        image.requires_grad = True

        # Forward pass
        logits = model(image)

        # Calculate KL divergence loss
        loss = calculate_kl_divergence(logits)

        # Backward pass
        model.zero_grad()
        loss.backward()


        gradients = [param.grad.view(-1) for param in model.parameters() if param.grad is not None]

        for layer_idx, grad_layer in enumerate(grad_layers):
            output_layer_gradients = gradients[grad_layer]
            if gradnorm_summation_method == 'l2':
                grad_norm = torch.norm(output_layer_gradients, p=2).item()  # Compute L2-norm
            elif gradnorm_summation_method == 'l1':
                grad_norm = torch.sum(torch.abs(output_layer_gradients)).item()
            else:
                raise Exception('Gradnorm summation method should be either l1 or l2')
            grad_norms[layer_idx].append(grad_norm)

    return grad_norms



def evaluate(net, idloader, oodloader, gradnorm_summation_method='l2', OOD_dict={}, use_cuda=True,verbose=True,**kwargs):
    """
    Evaluate GradNorm on the ID and OOD datasets.

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

    # Initialise arrays to store names and indices
    weight_names = []
    weight_indices = []

    # Iterate through network parameters with index
    for idx, (name, param) in enumerate(net.named_parameters()):
        weight_names.append(name)
        weight_indices.append(idx)

    conf_list = [[[] for _ in range(len(weight_names))] for _ in range(2)]


    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            grad_norms = gradnorm(net, inputs.cuda(), weight_indices, gradnorm_summation_method=gradnorm_summation_method)

            for i, norm_list in enumerate(grad_norms):
                conf_list[OOD][i].extend(norm_list)

    
    OOD_dict['name'] = []
    for module in weight_names:
        OOD_dict['name'].append('GradNorm (parameter='+str(module)+', summation_method='+str(gradnorm_summation_method)+')')

    return list(map(list, zip(*conf_list)))








