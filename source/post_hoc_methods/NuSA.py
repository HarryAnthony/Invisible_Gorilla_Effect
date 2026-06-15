import torch


def get_ID_activations(model, dataloader):
    # Pass data through the model
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs,_) in enumerate(dataloader):

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            batch_activations = model.head(inputs).cpu()
            if batch_idx == 0:
                activations = batch_activations
            else:
                activations = torch.cat((activations, batch_activations), dim=0)

    if activations.dim() == 4:
                activations = activations.squeeze(1)  # Remove the extra dimension added by the head method
                activations = activations[:, 0]

    return activations

def eval_NuSA(loader, net):

    W_torch, _ = net.get_head_weights()
    WW_T_torch = torch.mm(W_torch, W_torch.t()).cuda()
    inv_WW_T_torch = torch.pinverse(WW_T_torch).cuda()
    P_W_torch = torch.mm(W_torch.t(), torch.mm(inv_WW_T_torch, W_torch))

    activations = get_ID_activations(net, loader)
    activations = activations.view(activations.shape[0], -1)
    perc_values = []

    for j in range(0, len(activations)):
        X = activations[j].cuda()
        X = X.unsqueeze(-1)

        P_W_X = torch.mm(P_W_torch.cuda(),X.cuda())
        perc_in_W = torch.mm(P_W_X.t(), P_W_X)/torch.mm(X.t(),X)
        perc_values.append(perc_in_W.item())

    return perc_values


def evaluate(net, idloader, oodloader, use_cuda=True,verbose=True,**kwargs):
    """
    Evaluate NuSA on the ID and OOD datasets.

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
    #net.training = False
    #net.train()
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    confidence[0] = eval_NuSA(idloader, net)
    confidence[1] = eval_NuSA(oodloader, net)

    return confidence


def train():
    pass
