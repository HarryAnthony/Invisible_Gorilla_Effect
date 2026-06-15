import torch
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
from source.util.general_utils import print_progress
from source.util.training_utils import append_dropout
from source.util.evaluate_network_utils import softmax



def enable_dropout(net, dropout_rate=0.3, two_dim_dropout_rate=0.0, **kwargs):
    """ 
    Function to enable the dropout layers during test-time
     
    Parameters
    ----------
    net: torch.nn.Module
        The model to enable dropout layers for.
    dropout_rate: float
        The dropout rate to set for the dropout layers.
    two_dim_dropout_rate: float or str
        The dropout rate to set for the 2D dropout layers. If 'same' is passed, `two_dim_dropout_rate` is set to `dropout_rate`.

    Returns
    -------
    torch.nn.Module
        The model with the dropout layers enabled.
    """
    if not isinstance(dropout_rate, float):
        raise ValueError('dropout_rate must be a float')
    
    if two_dim_dropout_rate == 'same':
        two_dim_dropout_rate = dropout_rate
    elif not isinstance(two_dim_dropout_rate, float):
        raise ValueError('two_dim_dropout_rate must be a float or the string "same"')

    for module in net.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
            module.train()  # Set to train mode to enable dropout at test-time
        elif isinstance(module, nn.Dropout2d):
            module.p = two_dim_dropout_rate
            module.train()

    return net


def evaluate(net, idloader, oodloader, use_cuda=True, samples=30, MCDP_dropout_rate=0.3,verbose=True,OOD_dict={},**kwargs):
    """
    Evaluate Monte Carlo dropout on the ID and OOD datasets.

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
    samples: int
        The number of samples to use for Monte Carlo dropout. Default: 100
    dropout_rate: float
        The dropout rate to set the dropout layers to. Default: 0.3
    verbose: bool
        Whether to print progress. Default: True

    Returns
    -------
    list
        A confidence list containing two lists. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset
    """
    
    net.training = False
    confidence = [[],[]]

    net.eval()

    net = append_dropout(net, rate=MCDP_dropout_rate)

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    net_dropout = enable_dropout(net, dropout_rate=MCDP_dropout_rate, two_dim_dropout_rate=0.0) #Enable dropout layers during test-time


    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)


            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            with torch.no_grad():
                out = [softmax(net_dropout(inputs)) for _ in range(samples)]
            out_stack = np.stack(out, axis=2) 
            softmax_score = np.mean(out_stack, axis=2)
            confidence[OOD].extend(np.max(softmax_score,axis=1).tolist())

    OOD_dict['name'] = ['Monte Carlo dropout (p = '+str(MCDP_dropout_rate)+', samples = '+str(samples)+')']

    return confidence 


def train():
    pass

