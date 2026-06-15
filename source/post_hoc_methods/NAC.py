import torch
from source.util.general_utils import print_progress
import torch.nn.functional as F


def calculate_kl_divergence(logits):
    softmax_probs = F.softmax(logits, dim=1)
    uniform_probs = torch.ones_like(softmax_probs) / softmax_probs.size(1)
    kl_div = F.kl_div(softmax_probs.log(), uniform_probs, reduction='batchmean')
 
    softmax_probs.retain_grad()
    return kl_div


def compute_pdf(activation_states, NAC_num_bins=100):
    """
    Estimate probability density function (PDF) using histogram, optimized for speed.

    Args:
        activation_states (torch.Tensor): Activation states of shape (N, neurons).
        NAC_num_bins (int): Number of bins to estimate density.

    Returns:
        torch.Tensor: Estimated probability density for each neuron.
        list[torch.Tensor]: Bin edges for each neuron.
    """
    # Flatten activations across batch dimension
    activation_states = activation_states.view(activation_states.shape[0], -1)

    # Compute min & max for all neurons at once (vectorized)
    neuron_mins = activation_states.min(dim=0).values
    neuron_maxs = activation_states.max(dim=0).values

    # Create bins for all neurons
    neuron_bins = [torch.linspace(neuron_mins[i].item(), neuron_maxs[i].item(), steps=NAC_num_bins, device=activation_states.device) 
                   for i in range(activation_states.shape[1])]

    # Compute histogram for all neurons simultaneously (vectorized)
    histograms = torch.stack([
        torch.histc(activation_states[:, i], bins=NAC_num_bins, min=neuron_mins[i].item(), max=neuron_maxs[i].item())
        for i in range(activation_states.shape[1])
    ], dim=1)

    # Normalize to get probability distribution
    pdfs = histograms / histograms.sum(dim=0, keepdim=True)
    

    return pdfs.T, neuron_bins



def extract_neuron_activation_states(net,trainloader,module_names,NAC_sigmoid_alpha=3,NAC_num_bins = 100,use_cuda=True,verbose=True):

    activation_statistics = {}
    neuron_pdfs_dict = {}
    neuron_bins_dict = {}

    for module in module_names:
        activation_statistics[module] = torch.Tensor().cuda()

    l = len(trainloader)
    if verbose==True:
        print('Calculating NAC for training data')
    print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
    for batch_idx, (inputs, _) in enumerate(trainloader):
        print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        out_features = net.feature_extractor(inputs.cuda(),module_names=module_names)

        for module in module_names:
            out_features[module].retain_grad()

        loss = calculate_kl_divergence(out_features[module_names[-1]])
        net.zero_grad()
        loss.backward()

        for module in module_names:
            activation_state = out_features[module] * out_features[module].grad
            activation_state = torch.sigmoid(NAC_sigmoid_alpha * activation_state)
            activation_statistics[module] = torch.cat([activation_state.detach(),activation_statistics[module]],dim=0)

    for module in module_names:
        neuron_pdfs_dict[module], neuron_bins_dict[module] = compute_pdf(activation_statistics[module], NAC_num_bins=NAC_num_bins)

    return neuron_pdfs_dict, neuron_bins_dict



def calculate_neuron_activation_coverage(net,inputs,neuron_pdfs_dict, neuron_bins_dict,NAC_sigmoid_alpha,module_names,NAC_r=0.01):

    out_features = net.feature_extractor(inputs.cuda(),module_names=module_names)

    for module in module_names:
        out_features[module].retain_grad()

    loss = calculate_kl_divergence(out_features[module_names[-1]])

    net.zero_grad()
    loss.backward()

    coverage_score = [torch.Tensor().cuda() for module in module_names]
    NAC_total = [[] for module in module_names]

    for module_idx, module in enumerate(module_names):
        activation_state = out_features[module] * out_features[module].grad
        activation_state = torch.sigmoid(NAC_sigmoid_alpha * activation_state)
        activation_state = activation_state.view(activation_state.shape[0],-1)
        bins_module = neuron_bins_dict[module]
        pdf_module = neuron_pdfs_dict[module]

        for neuron_idx in range(0,activation_state.shape[1]):

            bins_neuron = bins_module[neuron_idx]
            activation_state_neuron = activation_state[:,neuron_idx]
            pdf_neuron = pdf_module[neuron_idx]
            # Expand dimensions to compute absolute difference with bins
            differences = torch.abs(activation_state_neuron[:, None] - bins_neuron[None, :])  # Shape: (batch_size, NAC_num_bins)

            # Find the index of the closest bin
            closest_bin_indices = torch.argmin(differences, dim=1)
            pdf_vals = pdf_neuron[closest_bin_indices]
            min_values = torch.min(pdf_vals, torch.tensor([NAC_r]).cuda())  # min(κi_X, r)
            neuron_activation_coverage = min_values* (1/NAC_r)
            if neuron_idx == 0:
                coverage_score[module_idx] = neuron_activation_coverage
            else:
                coverage_score[module_idx] = torch.vstack([neuron_activation_coverage,coverage_score[module_idx]])


        NAC_total[module_idx] = torch.mean(coverage_score[module_idx],dim=0).detach().tolist()

    return NAC_total 



def evaluate(net, idloader, oodloader, use_cuda=True, trainloader=None, OOD_dict={}, NAC_sigmoid_alpha=1.0, NAC_num_bins=100, NAC_r=0.5, Feature_based_modules=[-2], verbose=True, **kwargs):
    """
    Evaluate NAC distance score on the ID and OOD datasets.
    https://arxiv.org/pdf/2306.02879v3


    """
    net.eval()
    net.training = False
    if use_cuda:
        net.cuda()

    module_names = net.list_layers()
    if Feature_based_modules != None:
        module_names = [module_names[i] for i in Feature_based_modules]
    module_names = [module_names[-2]]


    conf_list = [[[] for _ in range(len(module_names))] for _ in range(2)]
    
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    neuron_pdfs_dict, neuron_bins_dict = extract_neuron_activation_states(net, trainloader, module_names, NAC_sigmoid_alpha=NAC_sigmoid_alpha, NAC_num_bins=NAC_num_bins)

    for OOD, loader in enumerate([idloader, oodloader]):
        if verbose==True:
            print('Evaluating '+['ID','OOD'][OOD]+' dataset')
        l = len(loader)
        print_progress(0, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)
        for batch_idx, (inputs, target) in enumerate(loader):
            print_progress(batch_idx+1, l, prefix='Progress:', suffix='Complete', length=50,verbose=verbose)

            inputs = inputs.cuda()

            NAC_total = calculate_neuron_activation_coverage(net,inputs,neuron_pdfs_dict,neuron_bins_dict,NAC_sigmoid_alpha,module_names,NAC_r=NAC_r)

            for module_idx,module in enumerate(module_names):
                conf_list[OOD][module_idx].extend(NAC_total[module_idx])

    
    OOD_dict['name'] = []
    for module in module_names:
        OOD_dict['name'].append('NAC (sigmoid_alpha='+str(NAC_sigmoid_alpha)+', r='+str(NAC_r)+', bins='+str(NAC_num_bins)+', module='+str(module)+')')

    return list(map(list, zip(*conf_list)))









