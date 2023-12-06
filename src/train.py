import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.validate import eval_model


def train_epoch(model, trainload, epoch, criterion, optimizer,
                train_stat, testload, writer, device, num_epochs_pretrain=0, targeted_ab=None):
    model.train()
    hist_loss = 0
    for _, data in enumerate(trainload, 0):  # get batch
        # parse batch
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        # sets the gradients of all optimized tensors to zero.
        optimizer.zero_grad()
        # get outputs
        if targeted_ab:
            outputs = model(features, targeted_ab)
        else:
            outputs = model(features)
        # calculate loss
        loss = criterion(outputs, labels.unsqueeze(1))
        # calculate gradients
        loss.backward()
        # performs a single optimization step (parameter update).
        optimizer.step()
        hist_loss += loss.item()

    if isinstance(testload, dict):
        testload = testload[targeted_ab]
    val_loss, *_, val_roc_auc, val_average_precision = eval_model(
        model, testload, criterion, targeted_ab, device)

    *_, train_roc_auc, train_average_precision = eval_model(
        model, trainload, criterion, targeted_ab, device)
    model.train()
    if train_stat:
        writer.add_scalars("Loss", {"Validation": val_loss,
                                    "Train": hist_loss / len(trainload)}, num_epochs_pretrain + epoch)

        writer.add_scalars("ROC AUC", {"Validation": val_roc_auc,
                                       "Train": train_roc_auc}, num_epochs_pretrain + epoch)

        writer.close()
    return val_loss


def train_model(model, trainload, num_epochs=20, learning_rate=0.001, patience=10, criterion=nn.BCEWithLogitsLoss,
                optim=torch.optim.Adam, train_stat=False, testload=None, tag=None, device='cpu'):
    criterion = criterion()
    optimizer = optim(model.parameters(), lr=learning_rate)
    val_losses = []

    if train_stat:
        writer = SummaryWriter(comment=tag)
    else:
        writer = None

    for ep in range(num_epochs):

        val_loss = train_epoch(model, trainload, ep, criterion, optimizer,
                               train_stat, testload, writer, device)

        if ep >= patience and max(val_losses[-(patience):]) <= val_loss:
            break

        val_losses.append(val_loss)


def train_epoch_multi(model, trainload, epoch, criterion, optimizer, train_stat, testload,
                      writer, device, antibodies, targeted_ab):
    hist_loss = 0
    targeted_ab_repeats = 1
    iterators = {antibody: iter(loader)
                 for antibody, loader in trainload.items()}

    is_over = {}
    for ab in antibodies:
        is_over[ab] = False
    if targeted_ab:
        is_over.pop(targeted_ab)
    while not all(is_over.values()):
        for antibody, loader_iter in iterators.items():
            try:
                features, labels = next(loader_iter)
            except StopIteration:
                if antibody == targeted_ab:
                    iterators[antibody] = iter(trainload[antibody])
                    features, labels = next(iterators[antibody])
                    targeted_ab_repeats += 1
                else:
                    is_over[antibody] = True
                continue
            features = features.to(device)
            labels = labels.to(device)
            # sets the gradients of all optimized tensors to zero.
            optimizer.zero_grad()
            # get outputs
            outputs = model(features, antibody)
            # calculate loss
            loss = criterion(outputs, labels.unsqueeze(1))
            # calculate gradients
            loss.backward()
            # performs a single optimization step (parameter update).
            optimizer.step()
            if antibody == targeted_ab or targeted_ab is None:
                hist_loss += loss.item()

    if isinstance(testload, DataLoader):
        val_loss, *_, val_roc_auc, val_average_precision = eval_model(
            model, testload, criterion, targeted_ab, device)
    else:
        val_loss_list, val_roc_auc_list = [], []
        for antibody, testload_ab in testload.items():
            val_loss, _ *, val_roc_auc, val_average_precision = eval_model(
                model, testload_ab, criterion, antibody, device)
            val_loss_list.append(val_loss)
            val_roc_auc_list.append(val_roc_auc)
        val_loss = sum(val_loss_list) / len(val_loss_list)
        val_roc_auc = sum(val_roc_auc_list) / len(val_roc_auc_list)


    model.train()
    if train_stat:
        if targeted_ab:
            hist_loss_mean = hist_loss / (len(trainload[targeted_ab] * targeted_ab_repeats))
        else:
            hist_loss_mean = hist_loss / sum((len(val) for val in trainload.values()))
        writer.add_scalars("Loss", {"Validation": val_loss,
                                    "Train": hist_loss_mean}, epoch)

        writer.add_scalars("ROC AUC", {"Validation": val_roc_auc}, epoch)

        writer.close()
    return val_loss


def train_multi_model(model, trainload, num_epochs_pretrain, learning_rate, patience, antibodies, targeted_ab,
                      criterion=nn.BCEWithLogitsLoss, optim=torch.optim.Adam, train_stat=False, testload=None,
                      tag=None, device='cpu', target_num_epochs=0):
    model.train()

    criterion = criterion()
    optimizer = optim(model.parameters(), lr=learning_rate)

    if train_stat:
        writer = SummaryWriter(comment=tag)
    else:
        writer = None

    val_losses = []
    for ep in range(num_epochs_pretrain):
        val_loss = train_epoch_multi(model, trainload, ep, criterion, optimizer, train_stat, testload,
                                     writer, device, antibodies, targeted_ab)

        if ep >= patience and max(val_losses[-(patience):]) <= val_loss:
            num_epochs_pretrain = ep
            break

        val_losses.append(val_loss)

    val_losses = []
    for ep in range(target_num_epochs):
        val_loss = train_epoch(model, trainload[targeted_ab], ep, criterion, optimizer,
                               train_stat, testload, writer, device, num_epochs_pretrain, targeted_ab)

        if ep >= patience and max(val_losses[-(patience):]) <= val_loss:
            break

        val_losses.append(val_loss)