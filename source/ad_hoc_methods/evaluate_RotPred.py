import torch
from source.util.general_utils import print_progress
import torch.nn.functional as F
from source.ad_hoc_methods.ad_hoc_models.RotPred_models import batch_random_rotation



def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,save_dir=None,filename=None,**kwargs):
    """
    Evaluate Rotation Prediction on the ID and OOD datasets.

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

    
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            with torch.no_grad():
                rotated_imgs, rotation_targets = batch_random_rotation(inputs)
                class_logits, rotation_logits = net(rotated_imgs.cuda())
                rotation_loss = F.cross_entropy(rotation_logits, rotation_targets.cuda(), reduction='none').cpu()
                rotation_loss /= 4
                total_loss = -1*rotation_loss

                confidence[OOD].extend(total_loss.tolist())


    return confidence
