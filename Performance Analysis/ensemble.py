from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def remove_trash(array):
    import pandas as pd
    df = array
    df = df.fillna(0)
    df = df.replace(np.inf, 0)
    df = df.replace(-np.inf, 0)
    return df

def get_pnl_transaction_cost(df_pos, df_pnl, transaction_costs):
    import numpy as np
    import pandas as pd
    num_days = df_pos.shape[0]
    arrays = [transaction_costs/10000 for _ in range(num_days)]
    repeated_transaction_cost = np.stack(arrays, axis=0)
    transaction_cost_pnl = (df_pnl - abs(repeated_transaction_cost*(df_pos - df_pos.shift(1))))
    transaction_cost_pnl = remove_trash(transaction_cost_pnl)
    return transaction_cost_pnl

def ensemble_position_getter(pos_list, x_final):
    ensemble_pos = 0
    for it in range(len(pos_list)):
        ensemble_pos = ensemble_pos + pos_list[it] * x_final[it]
    return ensemble_pos

def ensemble_weights_getter(test_pos_list, test_returns, method_type, info):
    if method_type == 'simple_mean':
        return [1/len(test_pos_list)] * len(test_pos_list)
    elif method_type == 'simple_super_learner':
        def minus_sharpe(t):
            ensemble_pos = 0
            for it in range(len(test_pos_list)):
                ensemble_pos = ensemble_pos + test_pos_list[it]*abs(t[it])
            ensemble_pnl = get_pnl_transaction_cost(ensemble_pos, ensemble_pos*test_returns, info['test_transaction_cost'])
            portifolio = ensemble_pnl.mean(axis=1)
            return - portifolio.mean()/portifolio.std()
        life = minimize(minus_sharpe, np.full(shape=len(test_pos_list), fill_value=1), options={'maxiter': 70, 'disp': True})
        x_final = abs(life.x)
        x_final = x_final/sum(x_final)
        plt.plot(x_final)
        plt.show()
        print('we are half awake in a fake empire')
        return x_final
    else:
        raise "you have selected an option that is not currently available"