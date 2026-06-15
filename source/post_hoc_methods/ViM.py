import torch
from source.util.general_utils import print_progress, variable_use_cuda


def evaluate(net,idloader,oodloader,trainloader=None,use_cuda=True, verbose=True, ViM_alpha=0.5, Feature_based_Dim=20, OOD_dict={}, **kwargs):
    """
    Calculate the ViM score for test inputs using the principal space.
    

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
    if use_cuda:
        net.cuda()

    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print("Computing the sample means and precisions for each class in the training data")
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (data, target) in enumerate(trainloader): #Iterate through each image in training data
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        del target

        data = variable_use_cuda(data,use_cuda)

        with torch.no_grad():
            features = net.head(data).cpu() #Get the pre-logit features

            if features.dim() > 2:
                features = torch.mean(features.view(features.size(0), features.size(1), -1), dim=2) #Flatten the features

        del data
        if batch_idx == 0:
            features_all = features.cpu()
        else:
            features_all = torch.cat((features_all.cpu(),features.cpu()),0)
        del features


    mean = torch.mean(features_all, axis=0)
    delta = features_all - mean
    cov_matrix = torch.matmul(delta.T, delta) / delta.size(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    principal_components = eigenvectors[:, -Feature_based_Dim:].cuda()  # Last D eigenvectors
    del features_all, delta, cov_matrix, eigenvalues, eigenvectors

    def compute_residual(features,principal_components):
        projection = features @ principal_components @ principal_components.T
        residual = features - projection
        return torch.linalg.norm(residual, axis=1)
    
    
    with torch.no_grad():
        #Calculate the kde metric for the ID and OOD data
       
        for OOD, loader in enumerate([idloader, oodloader]):
            if verbose==True:
                print('Evaluating '+['ID','OOD'][OOD]+' dataset')

            
            l = len(loader)
            print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
            for batch_idx, (inputs, target) in enumerate(loader):

                features = net.head(inputs.cuda())
                if features.dim() > 2:
                    features = torch.mean(features.view(features.size(0), features.size(1), -1), dim=2) #Flatten the features

                print_progress(batch_idx + 1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
                residuals = compute_residual(features,principal_components)
                logits = net(inputs.cuda())
                virtual_logits = ViM_alpha * torch.sqrt(residuals)
                exp_logits = torch.exp(logits)
                exp_virtual_logits = torch.exp(virtual_logits)
                vim_score = exp_virtual_logits / (torch.sum(exp_logits, dim=1) + exp_virtual_logits + 1e-10)
                confidence[OOD].extend(vim_score.cpu().numpy())


    OOD_dict['name'] = ['ViM (alpha='+str(ViM_alpha)+', D='+ str(Feature_based_Dim)+')']

    return confidence
        
