import torch
from source.util.general_utils import print_progress
from torch.nn.functional import normalize


def evaluate(net, idloader, oodloader, module, OOD_dict={'name': ['CIDER']}, num_classes=2, trainloader=None, use_cuda=True, save_dir=None, filename=None, verbose=True,**kwargs):
    """
    Evaluate CIDER distance score on the ID and OOD datasets.
    ----------
    net: torch.nn.Module
        The model to evaluate
    idloader: torch.utils.data.DataLoader
        The dataloader for the ID dataset
    oodloader: torch.utils.data.DataLoader 
        The dataloader for the OOD dataset
    module: int, list, tuple, numpy array, or str
        The index of the module after which to extract embeddings and apply CIDER distance score. 
        If it is an array then CIDER distance score is applied on several modules.
        If 'all' is passed, CIDER distance score is applied to all modules of the network.
    OOD_dict: dict
        The dictionary containing details of OOD detection method. Is used to store tha name of the module(s) after which CIDER distance score is applied.

    Returns
    -------
    list
        A confidence list containing pairs of lists for each module. The first list contains the confidence scores for the ID dataset 
        and the second list contains the confidence scores for the OOD dataset.
    """

    net.eval()
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    latent_data = []
    #Extract the embedding vectors for each image from the required modules
    print("Computing the sample means and precisions for each class in the training data")
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for batch_idx, (inputs, target) in enumerate(trainloader): #Iterate through each image in training data
        print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        with torch.no_grad():
            penultimate = net.encoder(inputs.cuda()).squeeze()
            train_features= net.head(penultimate)
            train_features= normalize(train_features, dim=1)

        if batch_idx == 0:
            latent_data = train_features
        else:
            latent_data = torch.cat((latent_data,train_features))

        del train_features


    net = net.cuda()
    for OOD,(loader) in enumerate([idloader,oodloader]):


        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

                
            with torch.no_grad():
                penultimate = net.encoder(inputs.cuda()).squeeze()
                test_features = net.head(penultimate)
                test_features = normalize(test_features, dim=1)
                cosine_sim = torch.matmul(test_features, latent_data.T)
                cosine_dist = 1 - cosine_sim
                # Get distance to k-th nearest neighbor
                knn_dists, _ = torch.topk(cosine_dist, k=1, dim=1, largest=False)
                scoring_function = -1*knn_dists[:, -1]  # Distance to the k-th nearest neighbor
                confidence[OOD].extend(scoring_function.cpu().tolist())


    return confidence





   