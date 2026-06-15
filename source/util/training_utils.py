"""
Helper functions for training a neural network.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from source.models.wide_resnet import Wide_ResNet
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet18, efficientnet_v2_s, vgg11, vgg16_bn, resnet34, resnet50, vgg16, vit_b_16, vit_b_32, vit_l_16, swin_t
from csv import writer

def get_network_architecture(args,num_classes,suffix):
    """
    Returns the network architecture and the file name for the network.

    Parameters
    ----------
    args : argparse
        The arguments that are passed to the program.
    num_classes : int
        The number of classes in the dataset.
    suffix : str
        The suffix for the file name.

    Returns
    -------
    net : torch.nn.Module
        The network architecture.
    file_name : str
        The file name for the network.

    """
    #Load the required neural network architecture
    if args.net_type == 'wide-resnet':
        net = Wide_ResNet(args.depth, args.widen_factor,args.dropout, num_classes)
        file_name = str(args.net_type)+'-'+str(args.depth)+'x'+str(args.widen_factor)+'_'+str(suffix)
    elif args.net_type == 'ResNet18':
        net = resnet18(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'ResNet34':
        net = resnet34(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'ResNet50':
        net = resnet50(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'efficientnet':
        net = efficientnet_v2_s(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vgg11':
        net = vgg11(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vgg16_bn':
        net = vgg16_bn(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vgg16':
        net = vgg16(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vit_b_16':
        net = vit_b_16(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vit_b_32':
        net = vit_b_32(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'vit_l_16':
        net = vit_l_16(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    elif args.net_type == 'swin_t':
        net = swin_t(num_classes=num_classes)
        file_name = str(args.net_type)+'_'+str(suffix)
    else:
        raise Exception("Network architecture  not avaliable, pick from list [wide-resnet,ResNet18,efficientnet,vgg11,vgg16]")
    
    net = add_dropout_network_architechture(net,net_info = {'Model architecture': args.net_type,
                                                            'Dropout': args.dropout,
                                                            'Act_func_Dropout': args.act_func_dropout,
                                                            'num_classes': num_classes})
 
    return net, file_name


def add_dropout_network_architechture(net,net_info):
    """
    Add dropout to the network architecture.

    Parameters
    ----------
    net : torch.nn.Module
        The net to add dropout to.
    net_info : dict
        A dictionary containing the net information.

    Returns
    -------
    net : torch.nn.Module
        The net with dropout added.
    """
    if net_info['Act_func_Dropout'] > 0:
        net = append_dropout(net,rate=net_info['Act_func_Dropout'])
    if net_info['Dropout'] > 0:    
        if net_info['Model architecture'] == 'ResNet18':
            num_ftrs = net.fc.in_features  # Get the number of input features
            net.fc = nn.Sequential(
                nn.Dropout(p=net_info['Dropout']),
                nn.Linear(num_ftrs, net_info['num_classes'])  # Modify the output size as needed
                )
    return net


def append_dropout(model, rate=0.2):
    """
    Append dropout to the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to append dropout to.
    rate : float, optional
        The dropout rate. The default is 0.2.

    Returns
    -------
    model : torch.nn.Module
        The model with dropout appended.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, rate)  # Recursively apply to child modules
        if isinstance(module, nn.ReLU) or isinstance(module, nn.SiLU) or isinstance(module, nn.ELU) or isinstance(module, nn.GELU) or isinstance(module, nn.CELU) or isinstance(module, nn.LeakyReLU):
            # Insert a Dropout2d layer after the ReLU
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)
    return model


def seed_used_before(seed_num,save_dir):
    """
    Checks to see if the seed has been used before. If it has been used before
    then the function returns True. If it has not been used before then the
    function returns False.

    Parameters
    ----------
    seed_num : int
        The seed number to check.
    save_dir : str
        The directory to check for previous experiments.

    Returns
    -------
    bool
        True if the seed has been used before, False if it has not been used before.
    """

    save_dir =os.listdir(save_dir) #Get list of previous experiments
    #Check the last few characters match the seed_num
    used_before = np.array([x[-4-len(str(seed_num)):-4] == str(seed_num) and
     x[-5-len(str(seed_num))].isdigit() is False for x in save_dir]).sum()
    return used_before > 0


def select_experiment_seed(seed_num,save_dir,allow_repeats=False):
    """
    Selects a seed for the experiment. If seed_num is not 'random' then the seed
    is checked to see if it has been used before. If it has been used before and 
    allow_repeats is False, then an exception is raised. If seed_num is 'random' 
    then a random seed is chosen and checked to see if it has been used before. 
    If it has been used before then a new random seed is chosen until a seed that 
    has not been used before is found.

    Parameters
    ----------
    seed_num : str or int
        Seed number to use for the experiment. If 'random' then a random seed is
        chosen.
    save_dir : str
        Directory to save the experiment results.
    allow_repeats : bool, optional
        If True then the same seed can be used multiple times. The default is False.

    Returns
    -------
    int
        Seed number to use for the experiment.

    """
    if seed_num != 'random':
        if seed_num.isdigit() == False:
            raise Exception('Seed must be an integer')
        if seed_used_before(seed_num,save_dir) and allow_repeats==False:
            raise Exception('Seed '+str(seed_num)+' has already been used. Chose another seed or allow repeats.')
        return int(seed_num)
    elif allow_repeats==False:
        seed_num = np.random.randint(0,100000)
        while seed_used_before(seed_num,save_dir):
                seed_num = np.random.randint(0,100000)
    return int(seed_num)


def get_class_weights(df):
    """
    get class weights for imbalanced dataset

    Parameters
    ----------
    df : pandas dataframe
        dataframe containing the class labels.

    Returns
    -------
    np.array
        array of class weights.

    """
    class_sample_count = np.array([len(np.where(df['class'] == t)[0]) for t in np.unique(df['class'])])
    weight = 1. - (class_sample_count / len(df))
    return np.array([weight[t] for t in df['class']])


def get_next_model_idx(model_record_filename: str) -> int:
    """
    Return the next Model_idx for appending a row to the model registry CSV.

    Uses 0-based indexing: returns the number of existing model rows (excluding
    the header line).

    Parameters
    ----------
    model_record_filename : str
        Path to model_list.csv.

    Returns
    -------
    int
        Next Model_idx value to use when appending a row.
    """
    if not os.path.isfile(model_record_filename):
        return 0
    with open(model_record_filename, 'r') as f:
        line_count = sum(1 for _ in f)
    return max(line_count - 1, 0)


def record_model(model_record_filename:str,list:list):
    """
    record model details in a csv file

    Parameters
    ----------
    model_record_filename : str
        filename of the csv file to record the model details.
    list : list
        list of model details to be recorded.
    """
    with open(model_record_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list)
        f_object.close()


def initialise_network(net,initialisation_method='he'):
    """
    Initialise the weights of the network.

    Parameters
    ----------
    net : torch.nn.Module
        The network to initialise.
    initialisation_method : str, optional
        Name of the initialisation method to use. The default is 'he'.

    Returns
    -------
    net : torch.nn.Module
        The initialised network.
    """
    if initialisation_method in ['glorot', 'he', 'lecun']:
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                if initialisation_method == 'glorot':
                    torch.nn.init.xavier_uniform_(module.weight)
                elif initialisation_method == 'he':
                    torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif initialisation_method == 'lecun':
                    torch.nn.init.normal_(module.weight, mean=0, std=1 / (module.weight.shape[1] ** 0.5))
    return net


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=2, gamma_neg=4):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss_pos = targets * (1 - probs) ** self.gamma_pos * torch.log(probs + 1e-8)
        loss_neg = (1 - targets) * probs ** self.gamma_neg * torch.log(1 - probs + 1e-8)
        return -torch.mean(loss_pos + loss_neg)
    


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=1, class_weight=None):
        """
        Asymmetric Loss with class-specific weighting for multi-label classification.
        
        Parameters
        ----------
        gamma_pos : float
            Focusing parameter for positive examples. Higher values focus more on hard positive examples.
        gamma_neg : float
            Focusing parameter for negative examples. Higher values down-weight easy negative examples.
        class_weight : torch.Tensor or None
            A tensor of shape (num_classes,) with a weight for each class, or None for no class weighting.
        """
        super(WeightedAsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.class_weight = class_weight

    def forward(self, logits, targets):
        """
        Forward pass for the weighted asymmetric loss.
        
        Parameters
        ----------
        logits : torch.Tensor
            The raw model outputs (logits), of shape (batch_size, num_classes).
        targets : torch.Tensor
            The binary ground truth labels, of shape (batch_size, num_classes).
            
        Returns
        -------
        torch.Tensor
            The calculated weighted asymmetric loss.
        """
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Calculate positive and negative losses
        loss_pos = targets * (1 - probs) ** self.gamma_pos * torch.log(probs + 1e-8)
        loss_neg = (1 - targets) * probs ** self.gamma_neg * torch.log(1 - probs + 1e-8)

        # Combine losses and apply class weights if provided
        loss = loss_pos + loss_neg
        if self.class_weight is not None:
            # Reshape class weights to match the shape of the loss
            class_weight = self.class_weight.view(1, -1)  # Shape (1, num_classes)
            loss *= class_weight*100  # Element-wise multiplication with class weights

        # Return the mean loss
        return -torch.mean(loss)



def get_criterion(criterion_name='CrossEntropyLoss', label_smoothing=0.0,pos_weight=None):
    """
    Get the criterion (loss function) for the model.

    Parameters
    ----------
    criterion_name : str, optional
        The name of the criterion. The default is 'CrossEntropyLoss'.
    label_smoothing : float, optional
        The label smoothing value. The default is 0.

    Returns
    -------
    criterion : torch.nn.modules.loss
        The criterion (loss function) for the model.
    """
    from libauc.losses import AUCMLoss
    criterion_dict = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        'BCELoss': nn.BCELoss(),
        'MSELoss': nn.MSELoss(),
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        'NLLLoss': nn.NLLLoss(),
        'smoothL1Loss': nn.SmoothL1Loss(),
        'WeightedAsymmetricLoss': WeightedAsymmetricLoss(class_weight=pos_weight),
        'AUCMLoss': AUCMLoss,
    }

    #input(pos_weight)


    if criterion_name in criterion_dict:
        criterion = criterion_dict[criterion_name]
    else:
        raise Exception('Criterion %s unknown.' % criterion_name)

    return criterion


def get_optimiser_scheduler(net, args, cf, trainloader, num_epochs):

    from libauc.optimizers import PESG
    
    momentum = cf.momentum
    weight_decay = args.weight_decay
    from libauc.losses import AUCMLoss

    # Set the optimiser
    if args.optimiser == 'SGD':
        optimiser = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    elif args.optimiser == 'Adam':
        optimiser = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    elif args.optimiser == 'AdamW':
        optimiser = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    elif args.optimiser == 'PESG':
        optimiser = PESG(net.parameters(),
               loss_fn=AUCMLoss(),
               lr=args.lr,
               momentum=0.9,
               #margin=margin,
               #epoch_decay=epoch_decay,
               weight_decay=weight_decay)
    else:
        raise Exception('Optimiser %s unknown. Choose from SGD, AdamW, and Adam.' % args.optimiser)

    # Set the scheduler
    if args.scheduler == 'MultiStepLR':
        lr_milestones = cf.lr_milestones
        lr_gamma = cf.lr_gamma
        scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=lr_milestones, gamma=lr_gamma)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.2, patience=30)
    elif args.scheduler == 'OneCycleLR':
        max_lr = float(np.max([float(args.max_lr), args.lr * 10]))
        steps_per_epoch = len(trainloader)
        div_factor = 10
        scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, total_steps=steps_per_epoch*num_epochs)
        #scheduler = lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs, div_factor=div_factor)
    elif args.scheduler == 'ConstantLR':
        scheduler = lr_scheduler.ConstantLR(optimiser, total_iters=num_epochs)
    else:
        raise Exception('Scheduler %s unknown. Choose from MultiStepLR, OneCycleLR, and ReduceLROnPlateau' % args.scheduler)

    return optimiser, scheduler


def set_activation_function(net, activation_function='ReLU'):
    """
    Set the activation function of the net.

    Parameters
    ----------
    net : torch.nn.Module
        The net to set the activation function of.
    activation_function : str
        The name of the activation function.

    Returns
    -------
    net : torch.nn.Module
        The net with the activation function set.
    """
    #Dictionary of activation functions
    activation_mapping = {
        'LeakyReLU': nn.LeakyReLU,
        'SiLU': nn.SiLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'CELU': nn.CELU
    }

    #Raise error if activation function is invalid
    if activation_function != 'ReLU' and activation_function not in activation_mapping.keys():
        raise ValueError(f'Invalid activation function: {activation_function}')

    #If activation function is ReLU, do nothing
    if activation_function == 'ReLU':
        return net
    
    activation_func = activation_mapping.get(activation_function)
    #Otherwise, convert the activation function
    net = convert_activation(net,activation_func)
    return net


def convert_activation(net,activation_func):
    """
    Convert the activation function of the net.

    Parameters
    ----------
    model : torch.nn.Module
        The net to convert the activation function of.
    activation_mapping : dict
        The dictionary of activation functions.
    activation_function : str
        The name of the activation function.

    """
    for child_name, child in net.named_children():
        if isinstance(child, nn.ReLU):
            setattr(net, child_name, activation_func())
        else:
            convert_activation(child,activation_func)
    return net

