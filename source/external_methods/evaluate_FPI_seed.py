import torch
from source.util.general_utils import print_progress
from source.util.evaluate_network_utils import loader_with_paths


def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,save_dir=None,filename=None,backbone_seed=None,FPI_threshold=0.5,**kwargs):
    """
    Evaluate FPI on the ID and OOD datasets.

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

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)


        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            outputs = net(inputs.cuda())
            outputs_above_threshold = outputs  > FPI_threshold
            pixels_above_threshold = torch.sum(outputs_above_threshold,dim=[1,2,3])
            pixels_above_threshold = [-1*x.item() for x in pixels_above_threshold]
            confidence[OOD].extend(pixels_above_threshold)

    return confidence
