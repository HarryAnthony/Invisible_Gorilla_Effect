import torch
import matplotlib.pyplot as plt
from source.util.general_utils import print_progress
import numpy as np
from source.post_hoc_methods.post_hoc_utils.zennit.src.zennit.composites import EpsilonPlusFlat
from source.post_hoc_methods.post_hoc_utils.zennit.src.zennit.canonizers import SequentialMergeBatchNorm
from source.post_hoc_methods.post_hoc_utils.crp.crp.attribution import CondAttribution
from source.post_hoc_methods.post_hoc_utils.crp.crp.helper import get_layer_names
from source.post_hoc_methods.post_hoc_utils.crp.crp.concepts import ChannelConcept
from torch.autograd import Variable


def mahalanobis_distance(x, mean, inv_covariance):
    """
    Calculate the Mahalanobis distance.

    Parameters:
    - x: numpy array, the sample vector.
    - mean: numpy array, the mean vector of the class.
    - covariance: numpy array, the covariance matrix of the class.

    Returns:
    - The Mahalanobis distance of the sample from the class.
    """
    x_minus_mu = x.detach().cpu() - mean
    distance = np.sqrt(np.dot(np.dot(x_minus_mu.T, inv_covariance), x_minus_mu))
    return distance


def evaluate(net, idloader, oodloader, module, OOD_dict={'name': ['CRP_dist']},trainloader=None,num_classes=None,**kwargs):
    """
    Evaluate PCX (CRP features + Mahalanobis distance) on the ID and OOD datasets.

    Uses conditional Concept Relevance Propagation (CRP) via zennit-crp and zennit to
    extract class-conditional channel relevances, then scores samples with Mahalanobis
    distance in that feature space.

    References
    ----------
    Achtibat et al., Nature Machine Intelligence, 2023:
    https://doi.org/10.1038/s42256-023-00711-8

    Anders et al., Zennit software paper, 2021:
    https://arxiv.org/abs/2106.13200

    See also `source/post_hoc_methods/post_hoc_utils/crp/ReadMe`.
    """
    confidence = crp_mahal(trainloader,net,num_classes,idloader,oodloader,OOD_dict=OOD_dict,verbose=True)
    return confidence


def crp_mahal(trainloader,model,classes,idloader,oodloader,OOD_dict={},verbose=True):
    model.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_classes = classes
    cc = ChannelConcept()
    composite = EpsilonPlusFlat(SequentialMergeBatchNorm)
    attribution = CondAttribution(model.cuda(), no_param_grad=True)
    layer_names = model.list_layers()
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

    #remove layers with out_proj in the name
    layer_names = [layer for layer in layer_names if 'out_proj' not in layer]

    val_per_class = [[[] for i in range(num_classes)] for _ in range(len(layer_names))]
    layer_means = [[[] for i in range(num_classes)] for _ in range(len(layer_names))]
    layer_convs = [[np.empty((0,0)) for i in range(num_classes)] for _ in range(len(layer_names))]


    if verbose==True:
        print('Calculating class means and covariances')
    l = len(trainloader)
    print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)
    for idx, (sample, target) in enumerate(trainloader):
        print_progress(idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        prediction = model(sample.cuda()).argmax(dim=1)
        
        for class_val in range(num_classes):
            conditions = [{"y": class_val}]
            sample.requires_grad = True
            sample.retain_grad()
            sample = Variable(sample,requires_grad=True)
            attr = attribution(sample.cuda(), conditions, composite, record_layer=layer_names)
            for layer_idx, layer in enumerate(layer_names):
                rel_c = cc.attribute(attr.relevances[layer], abs_norm=False)
                for im_idx in range(rel_c.shape[0]):
                    if prediction[im_idx] == class_val:
                        val_per_class[layer_idx][prediction[im_idx]].append(rel_c[im_idx].detach().cpu().numpy())

    # Calculate class means
    for layer_idx, layer in enumerate(layer_names):
        for class_idx in range(num_classes):
            layer_means[layer_idx][class_idx] = np.mean(val_per_class[layer_idx][class_idx], axis=0)
            
    # Calculate class covariance matrices
    for layer_idx, layer in enumerate(layer_names):
        for class_idx in range(num_classes):
            layer_convs[layer_idx][class_idx] = np.linalg.pinv(np.cov(np.array(val_per_class[layer_idx][class_idx]).T))


    OOD_dict['name'] = []
    for layer_idx, layer in enumerate(layer_names):
        OOD_dict['name'].append('PCX (module '+str(layer)+')')
    
    confidence = [[[] for i in range(2)] for _ in range(len(layer_names))]

    for OOD,(loader) in enumerate([idloader,oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

        for batch_idx, (sample, targets) in enumerate(loader):
            print_progress(batch_idx + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50,verbose=verbose)

            sample = sample.cuda()
            prediction = model(sample.cuda()).argmax(dim=1)

            for class_val in range(num_classes):
                conditions = [{"y": class_val}]
                sample.requires_grad = True
                sample.retain_grad()
                sample = Variable(sample,requires_grad=True)
                attr = attribution(sample.cuda(), conditions, composite, record_layer=layer_names)

                for layer_idx, layer in enumerate(layer_names):
                    rel_c = cc.attribute(attr.relevances[layer], abs_norm=False)
                    for im_idx in range(rel_c.shape[0]):
                        if prediction[im_idx] == class_val:
                            z = rel_c[im_idx]
                            class_distances = []
                            for class_idx in range(num_classes):
                                class_distances.append(mahalanobis_distance(z, layer_means[layer_idx][class_idx], layer_convs[layer_idx][class_idx]))
                            confidence[layer_idx][OOD].append(-1*np.min(class_distances))

    return confidence

