from source.util.general_utils import try_literal_eval


METHODS_NEED_TRAINING_DATA = [
    'Norm_flow', 'CIDER', 'g_odin', 'ddpm', 'NAC', 'CORP', 'COP', 'GradOrth', 'TAPUUD',
    'XOOD-M', 'mahalanobis', 'MBM', 'ReAct', 'GRAM', 'KNN', 'DICE', 'NMD',
    'PCX', 'MCP', 'WeiPer', 'SHE', 'negative_aware_norm', 'LOF', 'KDE', 'Residual', 'ViM',
    'GAIA', 'DeepSVDD',
]


def build_ood_evaluation_kwargs(args, use_cuda, net_dict, num_classes, trainloader=None):
    """
    Build keyword arguments passed to evaluate_ood_detection_method() / ood_evaluation().

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from evaluate_OOD_detection_method.py.
    use_cuda : bool
        Whether to run on GPU.
    net_dict : dict
        Model metadata returned by load_net().
    num_classes : int
        Number of ID classes for the loaded model.
    trainloader : torch.utils.data.DataLoader, optional
        Training dataloader for methods that require ID training data.

    Returns
    -------
    dict
        Keyword arguments for OOD evaluation.
    """
    kwargs_test = {'use_cuda': use_cuda, 'verbose': args.verbose}

    if args.save_results == True:
        kwargs_test['save_results'] = True
        kwargs_test['save_dir'] = args.save_results_path
        kwargs_test['combine_metrics'] = bool(args.combine_metrics)
        kwargs_test['save_results_micro'] = bool(args.save_results_micro)
        kwargs_test['first_edition'] = bool(args.first_edition)
        kwargs_test['backbone_seed'] = int(args.backbone_seed)
        kwargs_test['seed'] = int(args.seed)
        if args.plot_metric == True:
            kwargs_test['plot_metric'] = True

    if args.filename != 'practise':
        kwargs_test['filename'] = '_' + str(args.filename)

    if args.method == 'ODIN':
        kwargs_test['temper'] = args.temperature
        kwargs_test['noiseMagnitude'] = args.noiseMagnitude

    if args.method == 'MCDP':
        kwargs_test['samples'] = args.MCDP_samples
        kwargs_test['two_dim_dropout_rate'] = net_dict['act_func_dropout_rate']

    if args.method == 'deepensemble':
        if args.deep_ensemble_seed_list == '[]':
            raise Exception('Please specify a list of seeds to be used for deep ensemble')
        kwargs_test['net_dict'] = net_dict
        kwargs_test['seed_list'] = args.deep_ensemble_seed_list

    if args.method in METHODS_NEED_TRAINING_DATA:
        kwargs_test['trainloader'] = trainloader
        kwargs_test['module'] = try_literal_eval(args.mahalanobis_module)
        kwargs_test['num_classes'] = num_classes
        kwargs_test['feature_combination'] = (
            True if (args.method == 'MBM') or args.mahalanobis_feature_combination == True else False
        )
        kwargs_test['alpha'] = args.mahalanobis_alpha
        kwargs_test['preprocess'] = args.mahalanobis_preprocess
        kwargs_test['RMD'] = args.mahalanobis_RMD

    if args.method == 'MBM':
        kwargs_test['net_type'] = (
            net_dict['net_type'] + '_with_dropout'
            if net_dict['act_func_dropout_rate'] > 0
            else net_dict['net_type']
        )
        kwargs_test['MBM_type'] = args.MBM_type

    if args.method == 'ASH':
        kwargs_test['ASH_variant'] = args.ASH_variant
        kwargs_test['ASH_percentile'] = args.ASH_percentile

    if args.method in ['XOOD-M', 'FeatureNorm', 'NAC']:
        kwargs_test['Feature_based_modules'] = (
            try_literal_eval(args.Feature_based_modules) if args.Feature_based_modules != None else None
        )
        kwargs_test['Feature_based_alpha'] = (
            try_literal_eval(args.Feature_based_alpha) if args.Feature_based_alpha != None else None
        )

    if args.method == 'XOOD-M':
        kwargs_test['XOOD_M_C'] = args.XOOD_M_C

    if args.method == 'TAPUUD':
        kwargs_test['TAPUUD_num_clusters_list'] = try_literal_eval(args.TAPUUD_num_clusters_list)

    if args.method == 'GradOrth':
        kwargs_test['GradOrth_eps_threshold'] = args.GradOrth_eps_threshold

    if args.method in ['KNN', 'LOF']:
        kwargs_test['n_neighbours'] = args.n_neighbours
        kwargs_test['KNN_mode'] = args.KNN_mode

    if args.method == 'KDE':
        kwargs_test['KDE_mode'] = args.KDE_mode
        kwargs_test['KDE_kernel'] = args.KDE_kernel

    if args.method == 'Negative_aware_norm':
        kwargs_test['NaN_module'] = args.NaN_module

    if args.method == 'LOF':
        kwargs_test['LOF_mode'] = args.LOF_mode

    if args.method == 'Residual':
        kwargs_test['Feature_based_Dim'] = args.Feature_based_Dim

    if args.method == 'ReAct':
        kwargs_test['ReAct_percentile'] = args.ReAct_percentile

    if args.method == 'DICE':
        kwargs_test['DICE_sparsity_parameter'] = args.DICE_sparsification_param

    if args.method == 'GRAM':
        kwargs_test['power'] = args.GRAM_power

    if args.method == 'gradnorm':
        kwargs_test['gradnorm_summation_method'] = args.gradnorm_summation_method

    if args.method in ['Residual', 'ViM', 'COP', 'CORP']:
        kwargs_test['Feature_based_Dim'] = args.Feature_based_Dim

    if args.method == 'ViM':
        kwargs_test['ViM_alpha'] = args.ViM_alpha

    if args.method == 'WeiPer':
        kwargs_test['WeiPer_delta'] = args.WeiPer_delta
        kwargs_test['WeiPer_epsilon'] = args.WeiPer_epsilon
        kwargs_test['WeiPer_lambda1'] = args.WeiPer_lambda1
        kwargs_test['WeiPer_lambda2'] = args.WeiPer_lambda2
        kwargs_test['WeiPer_nbins'] = args.WeiPer_nbins
        kwargs_test['WeiPer_scoring_method'] = args.WeiPer_scoring_method

    if args.method == 'CORP':
        kwargs_test['CORP_num_rff'] = args.CORP_num_rff
        kwargs_test['CORP_gamma'] = args.CORP_gamma

    if args.method == 'NAC':
        kwargs_test['NAC_sigmoid_alpha'] = args.NAC_sigmoid_alpha
        kwargs_test['NAC_num_bins'] = args.NAC_num_bins
        kwargs_test['NAC_r'] = args.NAC_r

    if args.method == 'FPI_seed':
        kwargs_test['FPI_threshold'] = args.FPI_threshold

    if args.method == 'ddpm_seed':
        kwargs_test['DDPM_metric'] = args.DDPM_metric

    return kwargs_test
