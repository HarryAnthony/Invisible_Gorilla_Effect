import torch
import torch.nn.functional as F
from source.util.general_utils import print_progress
import time


# Function to add perturbations to weights
def create_perturbed_weights(weights, delta, r):
    perturbed_weights = []
    for _ in range(r):
        # Generate a perturbation from standard normal distribution and normalize it
        eta = torch.randn_like(weights, device=weights.device)
        eta = delta * (eta / eta.norm(dim=1, keepdim=True)) * weights.norm(dim=1, keepdim=True)
        perturbed_weights.append(weights + eta)
    return perturbed_weights


# Define a helper function to compute softmax confidence score
def compute_mcp(logits):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    softmax_probs = torch.nn.functional.softmax(logits, dim=1)
    return torch.max(softmax_probs, dim=1).values


def get_training_logits(net, trainloader,verbose=True):
    training_logits = torch.Tensor().cuda()
    if verbose:
        print('Getting training logits')
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for idx, (inputs, _) in enumerate(trainloader):
        print_progress(idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
        inputs = inputs.cuda()
        with torch.no_grad():
            logits = net(inputs)
            training_logits = torch.cat((training_logits, logits), 0)
    return training_logits


def evaluate(net, idloader, oodloader, trainloader=None, WeiPer_delta=1, WeiPer_r=30, WeiPer_epsilon=0.01, WeiPer_s1=4, WeiPer_s2=40, WeiPer_lambda1=2.5, WeiPer_lambda2=0.1, 
             WeiPer_scoring_method='KL_div', WeiPer_nbins=50,OOD_dict={'name': ['Mahalanobis']}, use_cuda=True,verbose=True,**kwargs):
    """
    Evaluate WeiPer on the ID and OOD datasets.

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

    # Save original model state
    original_state = net.state_dict()
    training_logits = get_training_logits(net, trainloader, verbose=verbose)

    original_weights,original_bias = net.get_head_weights()

    confidence = [[],[]]

    # Generate perturbed weights for each run
    perturbed_weights_set = create_perturbed_weights(original_weights, WeiPer_delta, WeiPer_r)


    def calculate_kl_divergence(logits, id_activations, WeiPer_nbins=50, **kwargs):
        """
        Calculate the KL-divergence-based score (WeiPer+KLD) for each input image.

        Parameters
        ----------
        logits: torch.Tensor
            The logits for the input batch after applying WeiPer perturbations.
        id_activations: torch.Tensor
            The activations from ID samples to use as a reference for the density.
        WeiPer_nbins: int
            Number of bins for histogram-based density estimation.
        epsilon: float
            A small constant added to densities to prevent zero values.
        s1, s2: int
            Sizes of the kernel for smoothing.
        lambda1, lambda2: float
            Scaling factors for combining KL divergence scores.

        Returns
        -------
        torch.Tensor
            A tensor of WeiPer+KLD scores, one per input image.
        """
        # Nested function to estimate density for individual tensors
        def estimate_density(tensor, WeiPer_nbins, epsilon, smooth_kernel_size):
            # Calculate histogram and normalize
            hist_counts = torch.histc(tensor, bins=WeiPer_nbins, min=tensor.min().item(), max=tensor.max().item())
            density = hist_counts / hist_counts.sum()  # Normalize to get density
            
            # Smooth density with convolution using a uniform kernel
            kernel = torch.ones(smooth_kernel_size).cuda() / smooth_kernel_size
            density = F.conv1d(density.view(1, 1, -1), kernel.view(1, 1, -1), padding=smooth_kernel_size//2)
            
            # Add epsilon and normalize
            density += epsilon
            density = density / density.sum()
            
            return density.squeeze()

        # Calculate reference density for ID activations
        pz_density = estimate_density(id_activations, WeiPer_nbins, WeiPer_epsilon, WeiPer_s1)

        # Prepare a list to store the per-image WeiPer+KLD scores
        scores = []

        # Iterate through each input in the batch
        for i in range(logits.size(0)):
            # Extract the logits for the current image
            image_logits = logits[i]

            # Calculate the density for this image's logits
            p_Weiper_z_density = estimate_density(image_logits, WeiPer_nbins, WeiPer_epsilon, WeiPer_s2)

            # Calculate KL-divergence for this image
            kl_pz = F.kl_div(pz_density.log(), pz_density, reduction='batchmean')
            kl_p_Weiper_z = F.kl_div(p_Weiper_z_density.log(), pz_density, reduction='batchmean')

            # Calculate the CP score for this image
            msp_score = compute_mcp(image_logits)

            # Calculate WeiPer+KLD score for this image
            weiper_kld_score = kl_pz + WeiPer_lambda1 * kl_p_Weiper_z - WeiPer_lambda2 * msp_score
            weiper_kld_score = -1 * weiper_kld_score  # Invert the score to make it a confidence score
            scores.append(weiper_kld_score)

        # Convert scores list to tensor
        return torch.tensor(scores, device=logits.device)

    # Evaluate on both ID and OOD loaders
    def get_confidences(loader,WeiPer_scoring_method='KL_div',**kwargs):
        confidences = []

        time_elapsed = []

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            start_time = time.time()

            if use_cuda:
                inputs = inputs.cuda()
            
            # Collect MSP scores for all perturbations
            batch_confidences = []
            for perturbed_weights in perturbed_weights_set:

                net.set_head_weights(perturbed_weights)
                
                # Forward pass with perturbed weights
                logits = net(inputs)

                # Choose scoring method
                if WeiPer_scoring_method == "MCP":
                    score = compute_mcp(logits)
                elif WeiPer_scoring_method == "KL_div":
                    # Here we assume ID activations are available (loaded separately)
                    score = calculate_kl_divergence(logits, id_activations=training_logits,**kwargs)  # Can also store ID activations for KLDiv calculation
                    #input(score)
                else:
                    raise ValueError("Invalid scoring method. Choose 'MSP' or 'KLDiv'.")
                
                batch_confidences.append(score)

            # Use the mean confidence score across perturbations
            batch_confidences = torch.stack(batch_confidences).mean(dim=0)
            confidences.extend(batch_confidences.cpu().numpy())

            end_time = time.time()
            time_elapsed.append(end_time - start_time)
            

        return confidences

    with torch.no_grad():
        if verbose:
            print('Evaluating ID dataset')
        confidence[0] = get_confidences(idloader,WeiPer_scoring_method=WeiPer_scoring_method,**kwargs)
        if verbose:
            print('Evaluating OOD dataset')
        confidence[1] = get_confidences(oodloader,WeiPer_scoring_method=WeiPer_scoring_method,**kwargs)

    # Restore original model state
    net.load_state_dict(original_state)

    OOD_dict['name'] = ['WeiPer (Scoring method='+str(WeiPer_scoring_method)+', r='+str(WeiPer_r)+', delta='+str(WeiPer_delta)+', epsilon='+str(WeiPer_epsilon)+', lambda_1='+str(WeiPer_lambda1)+
                        ', lambda_2='+str(WeiPer_lambda2)+', nbins='+str(WeiPer_nbins)+', s1='+str(WeiPer_s1)+', s2='+str(WeiPer_s2)+')']

        
    return confidence