import matplotlib.pyplot as plt
import numpy as np
import torch
from .validation.validation_rnn import rnn_execution
from .validation.helper_validation import get_pnl_transaction_cost


def classic_eval_loss(model, data_loader):
    # there is a big difference between the eval_loss
    # in the probability mode and in the end to end mode
    # in the probability mode it computes the average of the
    # losses of all inputs
    # whereas in the end to end, it computes the sharpe of all of 
    # the days of the portifolio (during the test set)
    model.eval()
    total_loss = torch.empty(0)
    end_to_end = model.end_to_end
    aggregate_portifolio_returns = torch.empty(0).to(model.device)
    with torch.no_grad():
        for x in data_loader:
            if model.device == torch.device('cuda'):
                x = x.cuda().contiguous()
            else:
                x = x.contiguous()
            if (end_to_end):
                aggregate_portifolio_returns = torch.cat((aggregate_portifolio_returns, model.get_returns_from_input_with_transaction_costs(x, 'test')), 0)
                continue
            loss = model.loss(x)
            total_loss = torch.cat((total_loss, loss), 0)
        if end_to_end:
            return - ((aggregate_portifolio_returns.mean())/(aggregate_portifolio_returns.std())).item()
        else:
            return total_loss.mean().item()


def new_eval_loss(model, datasets, model_hyper_parameters, info):
    _, new_pos, new_pnl, _ = rnn_execution(model, datasets[1][0], datasets[1][1])
    transaction_cost_pnl = get_pnl_transaction_cost(new_pos, new_pnl, info['test_transaction_cost']/10000)
    portifolio = transaction_cost_pnl.mean(axis=0)
    return - np.sqrt(252)*portifolio.mean()/portifolio.std()


def get_total_grad(model):
    total_norm = 0.0
    for parameter in model.parameters():
       if parameter.grad is not None:
           param_norm = parameter.grad.data.norm(2)
           total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def save_plot(losses, title, fname, yticks=  False):
    plt.figure()
    n_epochs = len(losses) - 1
    x = np.arange(n_epochs + 1)

    plt.plot(x, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    if (yticks):
        plt.yticks(np.arange(0, max(losses), 0.2))
    plt.savefig(fname)
    plt.clf()


def parameter_flattening(model):
    new_params = torch.zeros([0]).to(model.device)
    for _, param in model.named_parameters():
        if param.requires_grad:
            new_params = torch.cat((new_params, param.data.clone().reshape(-1)))
    return new_params