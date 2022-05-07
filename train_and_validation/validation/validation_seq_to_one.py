from .helper_execution import sample_returns, find_alpha2, comparison, save_cumulative_data, trial_summary_extractor


def seq_to_one_execution_and_collector_of_basic_metrics(model, data, real_returns, model_hyper_parameters, info):
    # validates the model in both the validation_data_sets, collects of performance metrics
    # model_hyper_parameters was already explained in the docs folder.
    # data has dimensions: (num_securities, num_sequences, seq_length, num_features + 1)
    # real_returns has dimensions (num_of_securities, num_days)
    import time
    import torch
    import numpy as np
    time6 = time.process_time()
    """ extracting the data from data_preparation_hyper_parameters and model_hyper_parameters"""
    seq_length = model_hyper_parameters['seq_length']
    end_to_end = info['end_to_end']
    device = info['device']

    with torch.no_grad():
        num_of_securities = data.shape[0]
        num_of_features = data.shape[3]

        """  the bins variable is to help the computation of the artificial returns from the respective bins of each data. """
        bins = np.linspace(lower_bound, upper_bound, d - 1)
        # pnl_list, pos_list, and prob_list have dimensions [2 (to differentiate train and test set), num_of_securities, num_of_data points processes until the respective moment.]
        # actually prob_list is a litte bit different, becuase it has a last of dimension of: num_bins
        """ Initializing the aggregate_data, in order to save it and analyze it later. """
        pnl_list = np.empty([num_of_securities, 0])
        pos_list = np.empty([num_of_securities, 0])
        rets_list = np.empty([num_of_securities, 0])
        # prob_list = [np.empty([num_of_securities,0,d]), np.empty([num_of_securities,0,d])]

        target_index = seq_length - 1
        data = data.view(num_of_securities, -1, num_of_features)
        label = data[:, :, 0].int()
        # data has dimensions: (num_securities, num of all data points, the number of features + 1)
        for i in range(data.shape[1] - seq_length + 1):
            # x_inp has dimensions (num_of_securities, seq_length, num_of_features)
            x_inp = data[:, (i):(i + seq_length), :]
            # note that the method get_logits_from_neural_net_output may return predictions for the whole
            # seq_length, but we only take the last prediction.
            if not end_to_end:
                logits = model.get_logits_from_neural_net_output(model.get_neural_net_output_from_input(x_inp, 'validation'))[:, [-1], :]
                # logits should have dimensions (num_of_securities, 1, num_bins)
                target_label = label[:, [i + seq_length - 1]]
                # target_label has dimension (num_of_securities, 1)
                # important to note, than in this instance, loss is just a number, it is the average of the all losses over all predictions. Just like we do in the training mode.

                
                """ part 1 probabilities estimation"""
                prob = torch.softmax(logits, dim=2)
                prob = prob.cpu().detach().numpy()
                # prob has dimensions(num_of_securities, 1, num_bins)
                # part 2 returns per class estimation, needs to be checked , may have an issue with dims
                classes_rets = np.repeat(np.expand_dims(sample_returns(np.arange(d), bins, using_middle_returns), 0), repeats=validation_num_of_securities * 1, axis=0)
                # part 3 maximize p_i x ln (1+alpha+ret) to get the position
                prob_fit_for_find_alpha2 = prob.reshape(-1, d)
                pos1 = find_alpha2(prob_fit_for_find_alpha2, classes_rets, alphas)
                pos = pos1.reshape([validation_num_of_securities, 1])

                # pos has dimensions [num_of_securites, 1]

                # part 4 next day return

                if using_discretized_bins:
                    flattened_future_labels = target_label.clone().view(-1).long()
                    ret = sample_returns(np.array(flattened_future_labels.cpu()), bins, using_middle_returns).reshape([validation_num_of_securities, 1])
                else:
                    ret = real_returns[:, [target_index]]
            else:
                
            target_index = target_index + 1
            pnl = ret * pos

            """ In this part we update the lists."""
            pnl_list = np.concatenate((pnl_list[train_or_test], pnl), axis=1)
            pos_list = np.concatenate((pos_list[train_or_test], pos), axis=1)
            rets_list = np.concatenate((rets_list[train_or_test], ret), axis=1)
            # prob_list[train_or_test] = np.concatenate((prob_list[train_or_test], prob), axis=1)

        trial_summary = trial_summary_extractor(pos_list, pnl_list, [aggregate_losses, aggregate_MSE_divergences, aggregate_KL_divergences])

        save_cumulative_data(data_preparation_hyper_parameters, model_hyper_parameters, cums, trial_summary, pnl_list, pos_list)

        time7 = time.process_time()
        print("End of Validation process, total time: ", time7 - time6)
        return trial_summary
