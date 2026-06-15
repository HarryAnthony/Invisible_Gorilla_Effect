import torch
from source.util.general_utils import print_progress
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer

    def forward(self, x):
        features = self.feature_extractor(x)
        return features.view(features.shape[0], -1)  # Flatten features


def train_gmm(features, num_clusters=5):
    """
    Train a Gaussian Mixture Model (GMM) on in-distribution features.

    Args:
        features (np.array): Extracted feature vectors (N, D)
        num_clusters (int): Number of GMM clusters (K)

    Returns:
        gmm (GaussianMixture): Trained GMM model
    """
    gmm = GaussianMixture(n_components=num_clusters, covariance_type="full")
    gmm.fit(features)
    return gmm


def mahalanobis_distance(x, mean, cov):
    """
    Compute the Mahalanobis distance of x from a given Gaussian cluster.

    Args:
        x (np.array): Feature vector (1, D)
        mean (np.array): Cluster mean (D,)
        cov (np.array): Cluster covariance matrix (D, D)

    Returns:
        float: Mahalanobis distance
    """
    inv_cov = torch.linalg.pinv(cov)
    delta = x-mean
    term_gau = torch.matmul(torch.matmul(delta,inv_cov),delta.T).diag()
    return term_gau


def compute_tap_mahalanobis(feature, gmm):
    """
    Compute TAP-Mahalanobis score for a given test feature.

    Args:
        feature (np.array): Feature vector (1, D)
        gmm (GaussianMixture): Trained GMM model

    Returns:
        float: TAP-Mahalanobis OOD score
    """
    means = gmm.means_  # (K, D)
    covariances = gmm.covariances_  # (K, D, D)
    
    # Compute Mahalanobis distance for all clusters and take the minimum
    distances = torch.vstack([mahalanobis_distance(feature, torch.Tensor(means[c]), torch.Tensor(covariances[c])) for c in range(gmm.n_components)])

    return -torch.min(distances,dim=0).values  # Negate to follow the convention (higher score = more ID-like)


def compute_tap_ensemble(feature, gmm_models):
    """
    Compute TAP-Ensemble score by averaging multiple TAP-Mahalanobis scores.

    Args:
        feature (np.array): Feature vector (1, D)
        gmm_models (list): List of GMM models trained with different cluster numbers

    Returns:
        float: TAP-Ensemble OOD score
    """
    
    scores = torch.vstack([compute_tap_mahalanobis(feature, gmm) for gmm in gmm_models])
    return torch.mean(scores,dim=0)


def evaluate(net, idloader, oodloader, use_cuda=True, trainloader=None, TAPUUD_num_clusters_list=[3],OOD_dict={}, num_classes=3, verbose=True, **kwargs):
    """
    Evaluate Mahalanobis distance score on the ID and OOD datasets.

    """

    net.eval()
    net.training = False
    if use_cuda:
        net.cuda()

    confidence = [[],[]]

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

    gmm_models = [train_gmm(training_features_tensor, num_clusters=k) for k in TAPUUD_num_clusters_list]


    with torch.no_grad():

        for OOD, loader in enumerate([idloader, oodloader]):
            if verbose==True:
                print('Evaluating '+['ID','OOD'][OOD]+' dataset')
            l = len(loader)
            print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
            for batch_idx, (inputs, target) in enumerate(loader):
                print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)

                test_features = net.head(inputs.cuda()).cpu()

                if test_features.dim() == 4:
                    test_features = test_features.squeeze(1)  # Remove the extra dimension added by the head method
                    test_features = test_features[:, 0]

                TAPUUD = compute_tap_ensemble(test_features,gmm_models)
                confidence[OOD].extend(TAPUUD.tolist())


    OOD_dict['name'] = ['TAP-UUD (Number of clusters='+str(TAPUUD_num_clusters_list)+')']

    return confidence


                



        

        
        