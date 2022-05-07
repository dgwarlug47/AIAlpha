from .helper_train import save_plot
from .models.Deep_Base import Deep_Model
from .helper_train import classic_eval_loss, new_eval_loss, get_total_grad, save_plot
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchcontrib
import copy
import time


def get_trained_model(data_sets, train_loader, model_hyper_parameters, info):
    # data_loaders is a list with 2 elements, both for the learning phase, the first is of the training, and second is of testing.
    # where both have load objects with the following dimensions: (batch_size, seq_len, num_features)
    # train distribution is self explanatory.
    # go to hyper_parameters.txt

    """ part 1 trains the model. """
    device = info['device']

    print("Start Learning")
    time4 = time.process_time()

    train_type = model_hyper_parameters['type_train']
    if train_type == 0:
        model = train_epochs(data_sets, train_loader, model_hyper_parameters, info)
    if train_type == 1:
        model = new_train_epochs(data_sets, train_loader, model_hyper_parameters, info)

    time6 = time.process_time()
    print("Total Training time, ", time6 - time4)
    print("Start Validation")
    return model


def train(model, train_loader, optimizer, epoch, model_hyper_parameters, info):
    model.train()
    batch_size = model_hyper_parameters['batch_size']
    seq_length = model_hyper_parameters['seq_length']
    grad_clip = model_hyper_parameters['grad_clip']
    number_of_days_per_epoch = model_hyper_parameters['number_of_days_per_epoch']
    number_of_batches_per_epoch = int(number_of_days_per_epoch / (batch_size * seq_length))
    if number_of_batches_per_epoch == 0:
        number_of_batches_per_epoch = 1

    grads_during_epoch = []
    train_losses = []
    
    # these information are useful when the model is end to end
    batch_size = model.batch_size
    aggregate_portifolio_returns = torch.empty(0).to(model.device)
    counter = 0

    for _ in range(batch_size * number_of_batches_per_epoch):
        x = train_loader.getitem()

        if model.device == torch.device('cuda'):
            x = x.cuda().contiguous()
        else:
            x = x.contiguous()
        
        counter = counter + 1
        model.mode_toggle('train')
        portifolio_returns = model.get_returns_from_input(x, epoch=epoch)
        aggregate_portifolio_returns = torch.cat((aggregate_portifolio_returns, portifolio_returns), 0)
        if counter == batch_size:
            loss = model.get_loss_from_returns(aggregate_portifolio_returns)
            counter = 0
            aggregate_portifolio_returns = torch.empty(0).to(model.device)
        else:
            continue
        optimizer.zero_grad()
        loss.backward()
        if grad_clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        new_total_grad = get_total_grad(model)
        grads_during_epoch.append(new_total_grad)
        train_losses.append(loss.item())

    return train_losses, grads_during_epoch


def train_epochs(data_sets, train_loader, model_hyper_parameters, info):
    epochs = model_hyper_parameters['epochs']
    lr = model_hyper_parameters['lr']
    using_swa = model_hyper_parameters['using_swa']
    swa_epochs_or_steps = model_hyper_parameters['swa_epochs_or_steps']
    swa_lr = model_hyper_parameters['swa_lr']
    swa_start = model_hyper_parameters['swa_start']
    swa_freq = model_hyper_parameters['swa_freq']
    momentum = model_hyper_parameters['momentum']
    adaptation = model_hyper_parameters['adaptation']
    early_stopping_dist = model_hyper_parameters['early_stopping_dist']
    type_eval = model_hyper_parameters['type_eval']
    device = info['device']
    if early_stopping_dist < 1.:
        early_stopping = False
    else:
        early_stopping = True
    if early_stopping_dist == -1:
        max_stopping = True
    else:
        max_stopping = False

    model = Deep_Model(model_hyper_parameters, info).to(device)
    if model_hyper_parameters['optimizer_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), betas=(momentum, adaptation), lr=lr)
    elif model_hyper_parameters['optimizer_type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise "this optimizer is not provided yet"

    if using_swa:
        if swa_epochs_or_steps:
            if swa_lr is None:
                opt = torchcontrib.optim.SWA(optimizer)
            else:
                opt = torchcontrib.optim.SWA(optimizer, swa_lr=swa_lr)
        else:
            if swa_lr is None:
                opt = torchcontrib.optim.SWA(optimizer, swa_start=swa_start, swa_freq=swa_freq)
            else:
                opt = torchcontrib.optim.SWA(optimizer, swa_start=swa_start, swa_freq=swa_freq, swa_lr=swa_lr)
    else:
        opt = optimizer

    train_losses = []
    giant_steps = []
    models = []

    if type_eval == 'classic':
        test_losses = [classic_eval_loss(model, test_loader)]
    elif type_eval == 'new':
        test_losses = [new_eval_loss(model, data_sets, model_hyper_parameters, info)]
    else:
        raise "this functionality is not provided yet"
    models.append(copy.deepcopy(model))

    for epoch in range(epochs):
        model.train()
        new_train_loss, new_step = train(model, train_loader, opt, epoch, model_hyper_parameters, info)
        train_losses.extend(new_train_loss)
        giant_steps.extend(new_step)
        if using_swa and swa_epochs_or_steps:
            if epoch >= swa_start:
                if epoch - swa_start % swa_freq == 0:
                    opt.update_swa()
        if type_eval == 'classic':
            test_loss = classic_eval_loss(model, test_loader)
        elif type_eval == 'new':
            test_loss = new_eval_loss(model, data_sets, model_hyper_parameters, info)
        else:
            raise "this functionality is not provided yet"
        test_losses.append(test_loss)
        print(f'Epoch {epoch}, Test Sharpe {test_loss:.4f}')
        if early_stopping or max_stopping:
            models.append(copy.deepcopy(model))
            if early_stopping and len(test_losses) > early_stopping_dist + 1:
                old_test_loss = test_losses[-early_stopping_dist]
                best_test_loss_after_old = np.min(test_losses[(-early_stopping_dist + 1):])
                if best_test_loss_after_old >= old_test_loss - np.abs(old_test_loss)*0.01:
                    model = models[-early_stopping_dist]
                    break
    if max_stopping:
        minimum_loss = np.inf
        good_index = -1
        for i in range(len(test_losses)):
            if minimum_loss > test_losses[i]:
                minimum_loss = test_losses[i]
                good_index = i
        if good_index == -1:
            raise "This should not happen, the loss should not be plus infinite"
        model = models[good_index]

    """ plotting train graphs, and saving the model """
    path = './Experiments/'+ info['prefix'] + '/' + str(info['current_experiment'])+ '/'
    torch.save(model.state_dict(), path + 'model.pt')
    save_plot(train_losses, 'Train Loss', path + 'Train')
    save_plot(test_losses, 'Test Loss', path + 'Test')
    save_plot(giant_steps, 'Grads norms', path + 'Grads')

    if using_swa:
        opt.swap_swa_sgd()
    return model


def new_train_epochs(data_sets, train_loader, model_hyper_parameters, info):
    epochs = model_hyper_parameters['epochs']
    lr = model_hyper_parameters['lr']
    using_swa = model_hyper_parameters['using_swa']
    swa_epochs_or_steps = model_hyper_parameters['swa_epochs_or_steps']
    swa_lr = model_hyper_parameters['swa_lr']
    swa_start = model_hyper_parameters['swa_start']
    swa_freq = model_hyper_parameters['swa_freq']
    momentum = model_hyper_parameters['momentum']
    adaptation = model_hyper_parameters['adaptation']
    early_stopping_dist = model_hyper_parameters['early_stopping_dist']
    type_eval = model_hyper_parameters['type_eval']
    device = info['device']
    if early_stopping_dist < 1.:
        early_stopping = False
    else:
        early_stopping = True
    if early_stopping_dist == -1:
        max_stopping = True
    else:
        max_stopping = False

    model = Deep_Model(model_hyper_parameters, info).to(device)
    if model_hyper_parameters['optimizer_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), betas=(momentum, adaptation), lr=lr)
    elif model_hyper_parameters['optimizer_type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise "this optimizer is not provided yet"

    opt = optimizer

    train_losses = []
    giant_steps = []
    models = []

    if type_eval == 'classic':
        test_losses = [classic_eval_loss(model, test_loader)]
    elif type_eval == 'new':
        test_losses = [new_eval_loss(model, data_sets, model_hyper_parameters, info)]
    else:
        raise "this functionality is not provided yet"
    models.append(copy.deepcopy(model))

    for epoch in range(epochs):
        model.train()
        new_train_loss, new_step = train(model, train_loader, opt, epoch, model_hyper_parameters, info)
        train_losses.extend(new_train_loss)
        giant_steps.extend(new_step)
        if using_swa and epoch >= swa_start:
            model.update_swa()
        if type_eval == 'classic':
            test_loss = classic_eval_loss(model, test_loader)
        elif type_eval == 'new':
            test_loss = new_eval_loss(model, data_sets, model_hyper_parameters, info)
        else:
            raise "this functionality is not provided yet"
        test_losses.append(test_loss)
        print(f'Epoch {epoch}, Test Sharpe {test_loss:.4f}')
        if early_stopping or max_stopping:
            models.append(copy.deepcopy(model))
            if early_stopping and len(test_losses) > early_stopping_dist + 1:
                old_test_loss = test_losses[-early_stopping_dist]
                best_test_loss_after_old = np.min(test_losses[(-early_stopping_dist + 1):])
                if best_test_loss_after_old >= old_test_loss - np.abs(old_test_loss)*0.01:
                    model = models[-early_stopping_dist]
                    break

    if max_stopping:
        minimum_loss = np.inf
        good_index = -1
        for i in range(len(test_losses)):
            if minimum_loss > test_losses[i]:
                minimum_loss = test_losses[i]
                good_index = i
        if good_index == -1:
            raise "This should not happen, the loss should not be plus infinite"
        model = models[good_index]

    """ plotting train graphs, and saving the model """
    path = './Experiments/'+ info['prefix'] + '/' + str(info['current_experiment'])+ '/'
    torch.save(model.state_dict(), path + 'model.pt')
    save_plot(train_losses, 'Train Loss', path + 'Train')
    save_plot(test_losses, 'Test Loss', path + 'Test')
    save_plot(giant_steps, 'Grads norms', path + 'Grads')

    return model