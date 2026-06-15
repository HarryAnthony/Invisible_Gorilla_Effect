import torch
from source.util.general_utils import print_progress


def evaluate(net, idloader, oodloader, trainloader=None, use_cuda=True, verbose=True, seed=None, **kwargs):
    """
    Evaluate the Simplified Hopfield Energy (SHE) method on ID and OOD datasets.
    
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
        Contains SHE scores for ID and OOD datasets.
    """
    net.eval()
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Move model to the appropriate device
    device = torch.device("cuda" if use_cuda else "cpu")
    net = net.to(device)

    # Fit class patterns on the ID dataset
    patterns = fit_class_patterns(net, trainloader, device, verbose=verbose)


    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)


                inputs = inputs.to(device)
                features = net(inputs)  # Extract features
                
                # Get the predicted class for each input
                logits = torch.matmul(features, patterns.T)
                predicted_classes = logits.argmax(dim=1)
                
                # Compute SHE scores (inner product with predicted class pattern)
                batch_scores = torch.sum(features * patterns[predicted_classes], dim=1)
                confidence[OOD].extend(batch_scores.cpu().numpy())


    return confidence


def fit_class_patterns(model, loader, device, verbose=False):
    """
    Fit the class patterns by calculating the mean feature vector for each class.
    
    Parameters
    ----------
    model : torch.nn.Module
        The feature extractor model.
    loader : DataLoader
        DataLoader for the ID dataset.
    device : torch.device
        The device to use for computations.
    verbose : bool
        Whether to print progress. Default is False.
    
    Returns
    -------
    torch.Tensor
        A tensor containing the mean feature vector for each class.
    """
    # Dictionary to store features by class
    class_features = {}
    
    # Extract features for each sample and group them by class
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model(inputs)
            
            for feature, label in zip(features, labels):
                label = label.item()
                if label not in class_features:
                    class_features[label] = []
                class_features[label].append(feature.cpu())
    
    # Compute mean feature (pattern) for each class
    patterns = []
    for label in sorted(class_features.keys()):
        features = torch.stack(class_features[label])
        class_mean = features.mean(dim=0)
        patterns.append(class_mean)
    
    # Stack patterns into a tensor for efficient computation
    patterns = torch.stack(patterns).to(device)

    return patterns


def compute_she_scores(model, loader, patterns, device, verbose=False):
    """
    Compute SHE scores for each input in the loader based on patterns.
    
    Parameters
    ----------
    model : torch.nn.Module
        The feature extractor model.
    loader : DataLoader
        DataLoader for the dataset (ID or OOD).
    patterns : torch.Tensor
        The mean feature vector for each class.
    device : torch.device
        The device to use for computations.
    verbose : bool
        Whether to print progress. Default is False.
    
    Returns
    -------
    list
        A list of SHE scores for the dataset.
    """
    she_scores = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            features = model(inputs)  # Extract features
            
            # Get the predicted class for each input
            logits = torch.matmul(features, patterns.T)
            predicted_classes = logits.argmax(dim=1)
            
            # Compute SHE scores (inner product with predicted class pattern)
            batch_scores = torch.sum(features * patterns[predicted_classes], dim=1)
            
            # Append negative scores (higher score implies more likely ID)
            she_scores.extend(-batch_scores.cpu().numpy())
    
    if verbose:
        print(f"Computed SHE scores for {len(she_scores)} samples.")
    
    return she_scores