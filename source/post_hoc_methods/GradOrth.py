import torch
from source.util.general_utils import print_progress
import torch.nn as nn


def evaluate(net, idloader, oodloader, use_cuda=True, trainloader=None, GradOrth_eps_threshold=0.9,OOD_dict={}, num_classes=3, verbose=True, **kwargs):
    """
    Evaluate GradOrth distance score on the ID and OOD datasets.

    """
    net.eval()
    net.training = False
    if use_cuda:
        net.cuda()

    confidence = [[],[]]

    module_names = net.list_layers()
    module_names = module_names[-3:]
    
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_features_list = []
    with torch.no_grad():  # No gradient computation for efficiency
        for inputs, _ in trainloader:  # Ignore labels (target) if not needed
            inputs = inputs.cuda()
            features = net.head(inputs).cpu()  # Extract features using the model's head
            training_features_list.append(features)  # Store on CPU to save GPU memory

    # Concatenate all collected features along batch dimension
    training_features_tensor = torch.cat(training_features_list, dim=0)
    
    if training_features_tensor.dim() == 4:
        training_features_tensor = training_features_tensor.squeeze(1)  # Remove the extra dimension added by the head method
        training_features_tensor = training_features_tensor[:, 0]

    U, S, Vt = torch.svd(training_features_tensor.T)  # SVD decomposition

    # Compute total squared Frobenius norm
    total_norm = torch.sum(S**2).item()
    # Compute cumulative energy & select top-k singular vectors
    cumulative_norm = torch.cumsum(S**2, dim=0) / total_norm

    k = torch.sum(cumulative_norm < GradOrth_eps_threshold).item() + 1  # Select k where norm exceeds threshold
    subspace = U[:, :k].cuda() 


    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)

            inputs = inputs.cuda()
            inputs.requires_grad_()

            # Forward pass
            out_features = net.feature_extractor(inputs,module_names=module_names)
            if out_features[module_names[-2]].shape == out_features[module_names[-1]].shape:
                module_names.pop(-2)
                
            out_features[module_names[-2]].retain_grad()

            #Define a pseudo-label (same shape as features)
            pseudo_label = torch.zeros_like(out_features[module_names[-1]])  # Zero tensor as dummy label

            # Compute MSE Loss
            criterion = nn.MSELoss()
            loss = criterion(out_features[module_names[-1]], pseudo_label)
            #loss = torch.max(out_features[module_names[-1]])  # Compute a dummy loss for gradient calculation
            loss.backward()
            gradients = out_features[module_names[-2]].grad

            if out_features[module_names[-2]].dim() == 3:
                #training_features_tensor = training_features_tensor.squeeze(1)  # Remove the extra dimension added by the head method
               out_features[module_names[-2]] = out_features[module_names[-2]][:, 0]
               gradients = gradients[:, 0]

            # Flatten gradients into 2D (batch_size, feature_dim)
            gradients_flat = gradients.view(gradients.shape[0], -1)

            # Project gradients onto the ID subspace
            projection = gradients_flat @ subspace @ subspace.T  # Project onto the subspace

            # Compute projection norm (GradOrth OOD Score)
            projection_norm = torch.norm(projection, dim=1)  # Norm across feature dimension

            confidence[OOD].extend(projection_norm.cpu().numpy())

    OOD_dict['name'] = ['GradOrth (eps_threshold='+str(GradOrth_eps_threshold)+')']


    return confidence


                



        

        
        