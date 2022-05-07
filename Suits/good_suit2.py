def info_creator():
    info = {}

    # all time most important configuration
    info['skipping_errors'] = True

    # where to find data
    info['tickers_names_file_path'] = 'Data2/Bloomberg/tickers_All_but_FX'
    info['tickers_base_path'] = 'Data2/Bloomberg/All Futures Momentum Features Bloomberg14/'

    # features
    info['features_names_file_paths'] = ['Data2/Bloomberg/features_davi', 'Data2/Bloomberg/features_macd']

    # optuna configuration
    info['num_shuffling'] = 100
    info['optuna_sampler'] = 'TPE'
    info['optuna_metric_name'] = 'sharpe_of_average_transaction_cost'
    info['optuna_n_jobs'] = 1

    # time stamps
    info['start_date'] = '1995-01-11'
    info['test_begin'] = '2012-01-01'
    info['validation_begin'] = '2016-01-01'
    info['end_date'] = '2021-01-01'

    # leverage
    info['leverage_type'] = 'volatility_scaling'
    info['vol_tgt'] = 0.15

    # transaction cost
    #info['test_transaction_cost'] =  np.full(shape=17, fill_value=1)
    info['test_transaction_cost'] =  np.concatenate((np.full(shape=17, fill_value=1), np.full(shape=29, fill_value=3), np.full(shape=14, fill_value=0.4)))

    # feature importance
    info['feature_importance_method'] = 'IntegratedGradients'
    info['feature_importance_minutes'] = 0.

    # device for the computation
    info['device'] = torch.device('cuda')

    # other stuff
    info['max_allocation'] = 'None'
    info['no_last_day'] = True
    info['long_only_benchmark'] = False

    # not important
    info['num_threads'] = 10
    info['num_trials'] = 1
    assert(info['num_trials'] == 1), "this functionality is not provided yet"
    return info

def search_space_creator(x):
    m = {
    # main hyphotesis
    "last_layer_type" : x.suggest_categorical("last_layer_type", ['Softmax']),
    "cross_sectional": x.suggest_categorical("cross_sectional", [False]),
    "bias" : x.suggest_categorical("bias", [True]),

    # regularization
    "train_transaction_cost_factor": x.suggest_uniform("train_transaction_cost_factor", 1, 3),
    "shrinkage" : x.suggest_categorical("shrinkage", [0.]),
    "dropout" : x.suggest_uniform("dropout", 0.2, 0.8),
    "type_dropout" : x.suggest_categorical("type_dropout", ["Davi"]),
    "label_type": x.suggest_categorical("label_type", [1, 5, 21]),

    # standard hyper parameters
    "batch_size" : x.suggest_int("batch_size", 10, 50),
    "seq_length" : x.suggest_categorical("seq_length", [64, 128, 256]),

    # model configuration
    "hidden_size" : x.suggest_int("hidden_size", 20, 160),
    "num_layers" : x.suggest_categorical("num_layers", [1]),
    "type_net" : x.suggest_categorical("type_net", ["LSTM"]),

    # attention hyper parameters
    "n_head" : x.suggest_int("n_head", 1, 5),
    "d_k" : x.suggest_int("d_k", 10, 40),
    "d_v" : x.suggest_int("d_v", 10, 40),

    # optimizer stuff
    "lr" : x.suggest_categorical("lr", [0.001, 0.005, 0.01, 0.05]),
    "grad_clip" : x.suggest_categorical("grad_clip", [0., 1]),
    "momentum" : x.suggest_categorical("momentum", [0.9]),
    "adaptation" : x.suggest_categorical("adaptation", [0.99]),
    "optimizer_type" : x.suggest_categorical("optimizer_type", ["Adam"]),

    # swa
    "type_train" : x.suggest_categorical("type_train", [1]),
    "using_swa" : x.suggest_categorical("using_swa", [False]),
    "swa_decay" : x.suggest_categorical("swa_decay", [0.]),
    "swa_epochs_or_steps" : x.suggest_categorical("swa_epochs_or_steps", [True]),
    "swa_start" : x.suggest_categorical("swa_start", [10, 14]),
    "swa_freq" : x.suggest_int("swa_freq", 1, 1),
    "swa_lr" : x.suggest_categorical("swa_lr", [None]),

    # loss functions
    "loss_function_from_returns" : x.suggest_categorical("loss_function_from_returns", ["sharpe"]),

    # epochs
    "epochs" : x.suggest_int("epochs", 5, 30),
    "early_stopping_dist" : x.suggest_categorical("early_stopping_dist", [0.]),
    "number_of_days_per_epoch": x.suggest_categorical("number_of_days_per_epoch", [70000]),

    # bootstrapping
    "bootstrap_ratio" : x.suggest_categorical("bootstrap_ratio", [0.]),
    "bootstrap_seed" : x.suggest_categorical("bootstrap_seed", [None]),

    # normalization
    "input_normalization" : x.suggest_categorical("input_normalization", [False, True]),

    # useless
    "type_eval": x.suggest_categorical("type_eval", ['new']),
    "removing_artificial_data" : x.suggest_categorical("removing_artificial_data", [False]),
    "chosen_targets" : x.suggest_categorical("chosen_targets", [0.]),
    }
    return m

