import torch
from source.util.general_utils import print_progress
from source.ad_hoc_methods.ad_hoc_utils.bayesian_torch.utils.util import predictive_entropy, mutual_information


def evaluate(net, idloader, oodloader, use_cuda=True, num_monte_carlo=100, BNN_uncertainty='predictive_entropy', OOD_dict={},verbose=True,**kwargs):
    """
    Evaluate BNN on the ID and OOD datasets.

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

            with torch.no_grad():
                output_mc = []
                for mc_run in range(num_monte_carlo):
                    logits = net(inputs.cuda())
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    output_mc.append(probs)
                output = torch.stack(output_mc)

                if BNN_uncertainty == 'predictive_entropy':
                    confidence[OOD].extend(predictive_entropy(output.data.cpu().numpy()))
                elif BNN_uncertainty == 'mutual_information':
                    confidence[OOD].extend(mutual_information(output.data.cpu().numpy()))
 
    if BNN_uncertainty == 'predictive_entropy':
        OOD_dict['name'] = ['BNN (predictive_entropy)']
    elif BNN_uncertainty == 'mutual_information':
        OOD_dict['name'] = ['BNN (mutual_information)']

    return confidence

