import torch
import numpy as np 
from source.util.general_utils import print_progress


def evaluate(net, idloader, oodloader, use_cuda=True, trainloader=None, verbose=True, save_dir=None,filename=None, backbone_seed=None,**kwargs):
    """
    Evaluate DeepSVDD on the ID and OOD datasets.

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
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    net = net.cuda()
    n_samples = 0
    c = torch.zeros(32).cuda()

    #Define the centre
    with torch.no_grad():
        for data in trainloader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.cuda()
            outputs = net(inputs,mode='embedding')
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c = c / n_samples

    for OOD,(loader) in enumerate([idloader,oodloader]):

        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            outputs = net(inputs.cuda(),mode='embedding')
            distances = -1*torch.norm(outputs - c, dim=1)  # ||φ(x) - c||
            confidence[OOD].extend(distances.detach().cpu().tolist())


    return confidence
