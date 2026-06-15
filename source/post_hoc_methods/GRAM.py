import torch
import torch.nn as nn
from source.util.general_utils import print_progress
import numpy as np


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.hooks = []

        # Register hooks only on layers that output feature maps
        # Example: Convolutional layers (nn.Conv2d), but can be extended to other layers as needed
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
                hook = layer.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)

    def hook_fn(self, module, input, output):
        # Store the feature map for the layer
        self.feature_maps.append(output)

    def clear_feature_maps(self):
        # Clear the feature map list
        self.feature_maps = []

    def get_feature_maps(self, x):
        # Forward pass to get feature maps
        self.clear_feature_maps()
        with torch.no_grad():
            self.model(x)
        return self.feature_maps
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def calculate_gram_matrices(feature_maps, p):
    """
    Calculate the Gram matrix for each channel of the feature maps in a batch, 
    after raising feature map values by p and the resulting Gram matrix by 1/p.

    Args:
    feature_maps (Tensor): A tensor of feature maps from a batch in a layer of the model.
    p (int): The power to raise the values of the feature map.

    Returns:
    Tensor: A tensor of Gram matrices corresponding to each feature map in the batch.
    """
    # B = Batch size, C = Number of channels, H = Height, W = Width
    B, C, H, W = feature_maps.size()
    
    # Reshape the feature map to [B, C, H*W]
    feature_map = feature_maps.view(B, C, H * W)

    # Raise feature map values by p
    feature_map_p = feature_map.pow(p)

    # Calculate the Gram matrix for each sample in the batch
    gram_matrices = torch.matmul(feature_map_p, feature_map_p.transpose(1, 2))

    # Normalize the Gram matrix by dividing by the number of elements in each feature map
    # and raise it by 1/p
    if p % 2 != 0:
        gram_matrices = (abs(gram_matrices / (C * H * W))).pow(1/p)
    else:
        gram_matrices = (gram_matrices / (C * H * W)).pow(1/p)

    return gram_matrices


def extract_upper_triangular(gram_matrices):
    """
    Extract the upper triangular matrix including the diagonal entries from each Gram matrix in a batch.

    Args:
    gram_matrices (Tensor): A tensor of Gram matrices from a batch.

    Returns:
    Tensor: A tensor containing the upper triangular values from each Gram matrix in the batch.
    """
    B, C, _ = gram_matrices.size()  # B = Batch size, C = Number of channels

    # Extract the upper triangular part including the diagonal for each sample in the batch
    upper_triangular_values = gram_matrices[:, torch.triu_indices(C, C)[0], torch.triu_indices(C, C)[1]]

    return upper_triangular_values


def update_min_max_values(feature_maps_batch, min_values, max_values, powers):
    """
    Update the minimum and maximum values for each layer and power p for the upper triangular values of the Gram matrices for a batch of data.

    Args:
    feature_maps_batch (list of Tensors): A list of tensors of feature maps from each layer of the model for the batch.
    min_values (dict): A dictionary to store the minimum values, indexed by layer and power.
    max_values (dict): A dictionary to store the maximum values, indexed by layer and power.
    powers (list): A list of integer powers to apply to the Gram matrices.

    Returns:
    None: Updates min_values and max_values in-place.
    """
    for L, feature_maps in enumerate(feature_maps_batch):
        for p in powers:
            # Calculate the Gram matrix for each sample in the batch
            gram_matrices = calculate_gram_matrices(feature_maps, p)

            # Extract the upper triangular values for each sample in the batch
            upper_triangular_values_batch = extract_upper_triangular(gram_matrices)

            # Compute the min and max across the batch for the current layer and power
            current_min, current_max = torch.min(upper_triangular_values_batch, dim=0)[0], torch.max(upper_triangular_values_batch, dim=0)[0]

            if L not in min_values:
                min_values[L] = {}
                max_values[L] = {}
            if p not in min_values[L]:
                min_values[L][p] = current_min
                max_values[L][p] = current_max
            else:
                # Update the min and max values considering all samples in the batch
                min_values[L][p] = torch.min(min_values[L][p], current_min)
                max_values[L][p] = torch.max(max_values[L][p], current_max)

            del current_min, current_max, upper_triangular_values_batch, gram_matrices



def calculate_min_max_per_class(trainloader, extractor, powers,verbose=True):
    """
    Calculate and store the minimum and maximum upper triangular values of the Gram matrices for each class in the training data.

    Args:
    trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
    model (torch.nn.Module): The neural network model.
    powers (list): A list of integer powers to apply to the Gram matrices.

    Returns:
    dict, dict: Two dictionaries containing the min and max values for each class.
    """
    mins_per_class = {}
    maxs_per_class = {}
    
    if verbose:
        print('Calculating max and min of training data Gram matrices')
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
        # Extract feature maps for the current batch
        feature_maps = extractor.get_feature_maps(inputs.cuda())

        del inputs

        for label in labels.unique():
            label = label.item()

            # Filter feature maps for the current class
            feature_maps_class = [fm[labels == label] for fm in feature_maps]

            if label not in mins_per_class:
                mins_per_class[label] = {}
                maxs_per_class[label] = {}

            # Update min and max values for the current class
            update_min_max_values(feature_maps_class, mins_per_class[label], maxs_per_class[label], powers)

        del feature_maps_class, feature_maps


    return mins_per_class, maxs_per_class


def calculate_total_deviations(testloader, extractor, powers, mins_per_class, maxs_per_class, num_classes=2, layer_weights=None, verbose=True):
    """
    Calculate the total deviation as an OOD scoring function for test images,
    summing the deviations across all layers for each test sample. Optionally,
    deviations can be weighted by layer.

    Args:
    testloader (torch.utils.data.DataLoader): DataLoader for the test data.
    model (torch.nn.Module): The neural network model.
    powers (list): A list of integer powers to apply to the Gram matrices.
    mins_per_class (dict): A dictionary containing the min values for each class.
    maxs_per_class (dict): A dictionary containing the max values for each class.
    layer_weights (numpy.ndarray or None): Optional array of weights for each layer.

    Returns:
    numpy.ndarray: An array of total deviations for each test sample.
    """
    total_deviations = []

    l = len(testloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

    for batch_idx, (inputs, labels) in enumerate(testloader):
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        # Extract feature maps for the current batch
        feature_maps = extractor.get_feature_maps(inputs.cuda())
        batch_deviations = torch.zeros(inputs.size(0))
        
        for L, feature_map in enumerate(feature_maps):
            dev = 0
            for p in powers:
                # Calculate the Gram matrix
                gram_matrix = calculate_gram_matrices(feature_map, p)
                upper_triangular_values_batch = extract_upper_triangular(gram_matrix)
                
                for c in range(0,num_classes):
                    min_val = mins_per_class[c][L][p]
                    max_val = maxs_per_class[c][L][p]

                    dev_min = (torch.relu(min_val - upper_triangular_values_batch) / torch.abs(min_val + 1e-8)).sum(dim=1, keepdim=True)
                    dev_max = (torch.relu(upper_triangular_values_batch - max_val) / torch.abs(max_val + 1e-8)).sum(dim=1, keepdim=True)

                    dev_inter = dev_min + dev_max
                    batch_deviations = batch_deviations + dev_inter.squeeze(1).cpu().detach()


        del feature_maps, dev_min, dev_max, dev_inter, dev, upper_triangular_values_batch, gram_matrix
                    
        if len(total_deviations) == 0:
            total_deviations = -1*batch_deviations
        else:
            total_deviations = np.concatenate((total_deviations, -1*batch_deviations), axis=0)

        del batch_deviations

    return total_deviations



def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,trainloader=None,power=[2],layer_weights=None,OOD_dict={},**kwargs):
    """
    Evaluate GRAM matrices on the ID and OOD datasets.

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

    with torch.no_grad():

        extractor = FeatureExtractor(net)
        mins_per_class, maxs_per_class = calculate_min_max_per_class(trainloader, extractor, power)

        for OOD,(loader) in enumerate([idloader,oodloader]):
            if verbose==True:
                print('Evaluating '+['ID','OOD'][OOD]+' dataset')

            total_deviations = calculate_total_deviations(loader, extractor, power, mins_per_class, maxs_per_class, layer_weights=layer_weights, verbose=verbose)
            confidence[OOD] = total_deviations

            del total_deviations

        extractor.remove_hooks()

        del extractor, mins_per_class, maxs_per_class, trainloader, net, idloader, oodloader, layer_weights, loader

        OOD_dict['name'] = ['GRAM (powers='+str(power)+')']

        del power
        
    return confidence


def train():
    pass







