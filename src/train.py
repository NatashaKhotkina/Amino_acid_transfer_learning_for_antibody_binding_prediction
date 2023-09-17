import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.validate import eval_model


def train_epoch(model, trainload, epoch, criterion, optimizer,
                train_stat, testload, writer, device, num_epochs_pretrain=0, targeted_ab=None):
    model.train()
    hist_loss = 0
    roc_auc = 0
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
        roc_auc += roc_auc_score(labels.unsqueeze(1), outputs)

    if train_stat:
        val_loss, accuracy, precision, recall, f1, val_roc_auc = eval_model(model, testload, criterion,
                                                                            targeted_ab, device)
        writer.add_scalars("Loss", {"Validation": val_loss,
                                    "Train": hist_loss / len(trainload)}, num_epochs_pretrain + epoch)

        writer.add_scalars("ROC AUC", {"Validation": val_roc_auc,
                                       "Train": roc_auc / len(trainload)}, num_epochs_pretrain + epoch)

        writer.close()


def train_model(model, trainload, num_epochs=20, learning_rate=0.001, criterion=nn.BCEWithLogitsLoss,
                optim=torch.optim.Adam, train_stat=False, testload=None, device='cpu'):
    criterion = criterion()
    optimizer = optim(model.parameters(), lr=learning_rate)

    if train_stat:
        writer = SummaryWriter(comment=tag)
    else:
        writer = None

    for ep in range(num_epochs):

        train_epoch(model, trainload, ep, criterion, optimizer,
                    train_stat, testload, writer, device)


def train_epoch_multi(model, trainload, epoch, criterion, optimizer, train_stat, testload,
                      writer, device, antibodies, targeted_ab):
    hist_loss = 0
    roc_auc = 0
    iterators = {antibody: iter(loader)
                 for antibody, loader in trainload.items()}

    is_over = {}
    for ab in antibodies:
        is_over[ab] = False
    is_over.pop(targeted_ab)
    while not all(is_over.values()):
        for antibody, loader_iter in iterators.items():
            try:
                features, labels = next(loader_iter)
            except StopIteration:
                if antibody == targeted_ab:
                    iterators[antibody] = iter(trainload[antibody])
                    features, labels = next(iterators[antibody])
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
            if antibody == targeted_ab:
                hist_loss += loss.item()
                roc_auc += roc_auc_score(labels.unsqueeze(1), outputs)

    if train_stat:
        val_loss, accuracy, precision, recall, f1, val_roc_auc = eval_model(model, testload, criterion,
                                                                            targeted_ab, device)
        writer.add_scalars("Loss", {"Validation": val_loss,
                                    "Train": hist_loss / len(trainload)}, epoch)

        writer.add_scalars("ROC AUC", {"Validation": val_roc_auc,
                                       "Train": roc_auc / len(trainload)}, epoch)

        writer.close()


def train_multi_model(model, trainload, num_epochs_pretrain, learning_rate, antibodies, targeted_ab,
                      criterion=nn.BCEWithLogitsLoss, optim=torch.optim.Adam, train_stat=False, testload=None,
                      writer=None, device='cpu', target_num_epochs=0):
    model.train()

    criterion = criterion()
    optimizer = optim(model.parameters(), lr=learning_rate)

    for ep in range(num_epochs_pretrain):
        train_epoch_multi(model, trainload, ep, criterion, optimizer, train_stat, testload,
                          writer, device, antibodies, targeted_ab)

    for ep in range(target_num_epochs):
        train_epoch(model, trainload[targeted_AB], ep, criterion, optimizer,
                    train_stat, testload, writer, device, num_epochs_pretrain, targeted_ab)
