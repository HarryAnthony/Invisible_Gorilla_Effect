import torch
import numpy as np 
from source.util.general_utils import print_progress


def ash_b(x, percentile=0.65):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    assert x.dim() == 4
    b, c, h, w = x.shape

    # Calculate the total sum per sample before pruning
    s1 = x.sum(dim=[1, 2, 3])

    # Determine the number of activations to keep
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)

    # Set kept values to the mean of original activations (binarization)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile=0.65):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    assert x.dim() == 4

    b, c, h, w = x.shape
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x


def ash_s(x, percentile =0.65):
    #input(x.dim())
    if x.dim() == 3:
        x = x.unsqueeze(1)
    assert x.dim() == 4
    b, c, h, w = x.shape

    # Calculate initial activation sum
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # Calculate new sum and apply a scaling factor
    s2 = x.sum(dim=[1, 2, 3])
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])
    return x


def evaluate(net, idloader, oodloader, use_cuda=True, ASH_variant='ASH-s', ASH_percentile=0.65, verbose=True, OOD_dict={}, **kwargs):
    """
    Evaluate the ASH method on ID and OOD datasets.
    
    Parameters
    ----------
    net : torch.nn.Module
        The model to evaluate, assumed to output logits.
    idloader : torch.utils.data.DataLoader
        DataLoader for the in-distribution (ID) dataset.
    oodloader : torch.utils.data.DataLoader
        DataLoader for the out-of-distribution (OOD) dataset.
    use_cuda : bool
        Whether to use GPU acceleration. Default is True.
    verbose : bool
        Whether to print progress. Default is True.
    
    Returns
    -------
    dict
        Contains ASH scores for ID and OOD datasets.
    """
    net.eval()
    net.training = False
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

        # Helper function to get ASH-based OOD scores for a batch
    def get_ash_scores(inputs, ASH_variant='ASH-s', percentile=0.65, **kwargs):

        # Pass inputs through the backbone to get features
        features = net.head(inputs)
        if features.dim() == 2:
            features = features.unsqueeze(1).unsqueeze(1)

        # Apply ASH activation shaping
        variants = {
        "ASH-s": ash_s,
        "ASH-p": ash_p,
        "ASH-b": ash_b,
        }
        #OOD_dict['name'] = [ASH_variant]

        ash_function = variants[ASH_variant]
        features = ash_function(features, percentile)

        # Pass modified features through the head to get logits
        logits = net.apply_head(features)
        energy = torch.logsumexp(logits, dim=-1)
        scores = [t.item() for t in energy]

        return scores


    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose:
            print('Evaluating ' + ['ID', 'OOD'][OOD] + ' dataset')

        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50, verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix='Progress:', suffix='Complete', length=50, verbose=verbose)

            if use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                batch_scores = get_ash_scores(inputs,ASH_variant=ASH_variant,percentile=ASH_percentile)
                confidence[OOD].extend(batch_scores)  # Store scores for ID or OOD


    OOD_dict['name'] = ['ASH (' + ASH_variant + ', percentile=' + str(ASH_percentile) + ')']
    return confidence