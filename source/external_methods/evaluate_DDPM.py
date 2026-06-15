import torch
import numpy as np 
from source.util.general_utils import print_progress
import torch.nn.functional as F
from torchvision.models import alexnet
from torchvision.models import AlexNet_Weights


def compute_lpips(x1, x2, model):
    """
    Compute LPIPS distance between two images using AlexNet features
    """
    with torch.no_grad():
        # Get features from AlexNet
        feat1 = model(x1)
        feat2 = model(x2)
        # Compute L2 distance between features
        return F.mse_loss(feat1, feat2)


def evaluate(net, idloader, oodloader, use_cuda=True, verbose=True, DDPM_metric='mse', trainloader=None, backbone_seed=None,save_dir=None,filename=None, **kwargs):
    """
    Evaluate DDPM-based OOD detection on the ID and OOD datasets.

    Parameters
    ----------
    net: torch.nn.Module
        The trained DDPM model
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader
        The dataloader for the OOD dataset
    use_cuda: bool
        Whether to use cuda. Default: True
    verbose: bool
        Whether to print progress. Default: True
    DDPM_metric: str
        The metric to use for OOD detection. Options are:
        - 'mse': Use only Mean Squared Error
        - 'lpips': Use only LPIPS (Learned Perceptual Image Patch Similarity)
        - 'both': Use both metrics and average their Z-scores (default)

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """
    
    confidence = [[], []]

    # Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net.eval()
    net.cuda()

    # Load AlexNet for LPIPS computation if needed
    if DDPM_metric in ['lpips', 'both']:
        alexnet_model = alexnet(weights=AlexNet_Weights.DEFAULT)
        alexnet_model = alexnet_model.features
        if use_cuda:
            alexnet_model = alexnet_model.cuda()
        alexnet_model.eval()

    # Define noise levels for reconstruction
    n_steps = 999  # Total number of diffusion steps
    n_reconstructions = 100  # Number of reconstructions to perform
    t_start = np.linspace(0, n_steps, n_reconstructions, dtype=int)  # Starting points for reconstruction

    # Collect validation set statistics for Z-score computation
    val_scores_mse = []
    val_scores_lpips = []
    
    if verbose:
        print("Computing validation set statistics...")
    
    # First pass: compute validation statistics
    for batch_idx, (inputs, _) in enumerate(trainloader):
        if verbose:
            print_progress(batch_idx + 1, len(trainloader), prefix='Validation:', suffix='Complete', length=50)
        
        if use_cuda:
            inputs = inputs.cuda()

        with torch.no_grad():
            
            # For each image in the batch
            for img_idx in range(inputs.shape[0]):
                x0 = inputs[img_idx:img_idx+1]
                
                # For each noise level
                for t in t_start:
                    # Add noise to the image
                    noise = torch.randn_like(x0)
                    alpha_t = net.get_alpha(t)
                    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
                    
                    # Reconstruct the image
                    x0_recon = net.sample(xt.cuda(), t)
                    
                    # Compute metrics based on selection
                    if DDPM_metric in ['mse', 'both']:
                        mse = F.mse_loss(x0, x0_recon)
                        val_scores_mse.append(mse.item())
                    
                    if DDPM_metric in ['lpips', 'both']:
                        lpips = compute_lpips(x0, x0_recon, alexnet_model)
                        val_scores_lpips.append(lpips.item())


    # Compute validation statistics
    if DDPM_metric in ['mse', 'both']:
        val_mse_mean = np.mean(val_scores_mse)
        val_mse_std = np.std(val_scores_mse)
    
    if DDPM_metric in ['lpips', 'both']:
        val_lpips_mean = np.mean(val_scores_lpips)
        val_lpips_std = np.std(val_scores_lpips)


    # Second pass: evaluate ID and OOD datasets
    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose:
            print('Evaluating ' + ['ID', 'OOD'][OOD] + ' dataset')


        for batch_idx, (inputs, targets) in enumerate(loader):
            if verbose:
                print_progress(batch_idx + 1, len(loader), prefix='Progress:', suffix='Complete', length=50)

            if use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():


            # For each image in the batch
                for img_idx in range(inputs.shape[0]):

                    x0 = inputs[img_idx:img_idx+1]
                    scores = []

                    # For each noise level
                    for t in t_start:
                        #input('yo')
                        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise     
                        # Reconstruct the image
                        x0_recon = net.sample(xt, t)

                        # Compute metrics based on selection
                        if DDPM_metric == 'mse':
                            mse = F.mse_loss(x0, x0_recon)
                            mse_zscore = (mse.item() - val_mse_mean) / val_mse_std
                            scores.append(mse_zscore)
                        
                        elif DDPM_metric == 'lpips':
                            lpips = compute_lpips(x0, x0_recon, alexnet_model)
                            lpips_zscore = (lpips.item() - val_lpips_mean) / val_lpips_std
                            scores.append(lpips_zscore)
                        
                        else:  # 'both'
                            mse = F.mse_loss(x0, x0_recon)
                            lpips = compute_lpips(x0, x0_recon, alexnet_model)
                            mse_zscore = (mse.item() - val_mse_mean) / val_mse_std
                            lpips_zscore = (lpips.item() - val_lpips_mean) / val_lpips_std
                            scores.append((mse_zscore + lpips_zscore) / 2)

                    confidence[OOD].append(-np.mean(scores))

    return confidence 