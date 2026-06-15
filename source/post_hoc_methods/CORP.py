import torch
from source.util.general_utils import print_progress
import torch
import torch.nn.functional as F



def random_fourier_features(features, num_rff=500, gamma=0.1):
    """
    Compute Random Fourier Features (RFFs) to approximate the Gaussian kernel.

    Args:
        features (torch.Tensor): Input feature tensor of shape (batch_size, feature_dim)
        num_rff (int): Number of random Fourier features
        gamma (float): Gaussian kernel hyperparameter

    Returns:
        torch.Tensor: RFF-transformed features of shape (batch_size, num_rff)
    """
    feature_dim = features.shape[1]

    # Sample random weights for RFF
    omega = torch.randn(feature_dim, num_rff) * (2 * gamma) ** 0.5  # ~ N(0, sqrt(2*gamma))
    omega = omega.to(features.device)

    # Sample random phase shifts
    bias = torch.rand(num_rff) * 2 * torch.pi
    bias = bias.to(features.device)

    return omega, bias


def gaussian_mapping(features, num_rff, omega, bias):
    """
    Compute Random Fourier Features (RFFs) to approximate the Gaussian kernel.

    Args:
        features (torch.Tensor): Input feature tensor of shape (batch_size, feature_dim)
        num_rff (int): Number of random Fourier features
        gamma (float): Gaussian kernel hyperparameter

    Returns:
        torch.Tensor: RFF-transformed features of shape (batch_size, num_rff)
    """
    # Compute RFF transformation
    transformed_features = torch.cos(F.linear(features, omega.T) + bias)


    # Scale to ensure unit variance
    transformed_features *= (2.0 / num_rff) ** 0.5

    return transformed_features


def cosine_mapping(features):
    """
    Apply cosine normalization to feature vectors.
    Args:
        features (torch.Tensor): Feature tensor of shape [batch, feature_dim]
    Returns:
        torch.Tensor: Cosine-mapped feature tensor.
    """
    return features / torch.norm(features, p=2, dim=1, keepdim=True)


def compute_covariance(features):
    """
    Compute the covariance matrix in the mapped Φ(z)-space.
    Args:
        features (np.array): Feature matrix.
    Returns:
        tuple: Covariance matrix and mean vector.
    """
    mu_phi = torch.mean(features, axis=0, keepdims=True)  # Compute mean
    centered_features = features - mu_phi
    covariance_matrix = centered_features.T @ centered_features / (features.shape[0] - 1)
    return covariance_matrix, mu_phi

def compute_reconstruction_error(U_q, mu_phi, features):
    """
    Compute the reconstruction error for a given sample.
    Args:
        U_q (np.array): Projection matrix (top-q eigenvectors).
        mu_phi (np.array): Mean feature vector.
        features (np.array): Feature matrix.
    Returns:
        np.array: Reconstruction error for each sample.
    """
    centered_features = features - mu_phi
    projection = U_q @ U_q.T @ centered_features.T  # Projected version
    error = torch.linalg.norm(projection.T - centered_features, axis=1)  # L2 norm of difference
    return error


def eigen_decomposition(cov_matrix, num_components=50):
    """
    Perform eigendecomposition on the covariance matrix.
    Args:
        cov_matrix (np.array): Covariance matrix.
        num_components (int): Number of principal components to retain.
    Returns:
        tuple: Eigenvectors (U_q) and mean vector (μΦ).
    """
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)  # Eigen decomposition
    sorted_indices = torch.argsort(eigvals, descending=True)  # Sort in descending order
    top_q_vectors = eigvecs[:, sorted_indices[:num_components]]  # Select top q components
    return top_q_vectors



def evaluate(net, idloader, oodloader, use_cuda=True, CORP_num_rff=500, CORP_gamma=0.1, Feature_based_Dim=20,trainloader=None,verbose=True,OOD_dict={},**kwargs):
    """
    Evaluate CoRP score on the ID and OOD datasets.
    
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
    confidence = [[],[]]

    #Required to ensure that the results are reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    training_features_list = []
    with torch.no_grad():  # No gradient computation for efficiency
        for batch_idx, (inputs, _) in enumerate(trainloader):  # Ignore labels (target) if not needed
            inputs = inputs.cuda()
            features = net.head(inputs).cpu()  # Extract features using the model's head

            if features.dim() == 4:
                features = features.squeeze(1)  # Remove the extra dimension added by the head method
                features = features[:, 0]

            if batch_idx == 0:
                omega, bias = random_fourier_features(features, num_rff=CORP_num_rff, gamma=CORP_gamma)
            cosine_mapped_features = cosine_mapping(features)
            gaussian_mapped_features = gaussian_mapping(cosine_mapped_features,CORP_num_rff,omega,bias)
            training_features_list.append(gaussian_mapped_features)  # Store on CPU to save GPU memory

    # Concatenate all collected features along batch dimension
    training_features_tensor = torch.cat(training_features_list, dim=0)

    cov_matrix, mu_phi = compute_covariance(training_features_tensor)

    #Perform Eigen Decomposition
    U_q = eigen_decomposition(cov_matrix, num_components=Feature_based_Dim)

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')

        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (inputs, _) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            with torch.no_grad():
            #Classify the inputs

                inputs = inputs.cuda()
                features = net.head(inputs).cpu()  # Extract features using the model's head

                if features.dim() == 4:
                    features = features.squeeze(1)  # Remove the extra dimension added by the head method
                    features = features[:, 0]

                cosine_mapped_features = cosine_mapping(features)
                gaussian_mapped_features = gaussian_mapping(cosine_mapped_features,CORP_num_rff,omega,bias)
                CORP_score = compute_reconstruction_error(U_q,mu_phi,gaussian_mapped_features)
                CORP_score = [-1*score.item() for score in CORP_score]
                confidence[OOD].extend(CORP_score)


    OOD_dict['name'] = ['CoRP (D='+str(Feature_based_Dim)+', num_rff='+str(CORP_num_rff)+', gamma='+str(CORP_gamma)+')']

    return confidence