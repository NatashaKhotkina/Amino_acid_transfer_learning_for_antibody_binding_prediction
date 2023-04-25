from collections import defaultdict

import torch
from torch import nn


def train_model(model, trainload, num_epochs=20, learning_rate=0.001, criterion=nn.BCEWithLogitsLoss,
                optim=torch.optim.Adam, print_epoch=True, device='cpu'):
    model.train()

    criterion = criterion()
    optimizer = optim(model.parameters(), lr=learning_rate)

    loss_hist = []
    for ep in range(num_epochs):
        hist_loss = 0
        for _, data in enumerate(trainload, 0):  # get batch
            # parse batch
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            # sets the gradients of all optimized tensors to zero.
            optimizer.zero_grad()
            # get outputs
            outputs = model(features)
            # calculate loss
            loss = criterion(outputs, labels.unsqueeze(1))
            # calculate gradients
            loss.backward()
            # performs a single optimization step (parameter update).
            optimizer.step()
            hist_loss += loss.item()

        if print_epoch:
            loss_hist.append(hist_loss / len(trainload))
            print(f"Epoch={ep} loss={loss_hist[ep]:.4f}")


def train_multi_model(model, trainload, num_epochs=20, learning_rate=0.001, criterion=nn.BCEWithLogitsLoss,
                      optim=torch.optim.Adam, print_epoch=False, device='cpu', targeted_AB='ly16'):
    model.train()

    criterion = criterion()
    optimizer = optim(model.parameters(), lr=learning_rate)

    loss_hist = defaultdict(list)
    for ep in range(num_epochs):
        hist_loss = defaultdict(float)
        iterators = {antibody: iter(loader)
                     for antibody, loader in trainload.items()}
        is_over = {'ly555': False,
                  'ly16': False,
                  'REGN33': False,
                  'REGN87': False}
        is_over.pop(targeted_AB)
        while not all(is_over.values()):
            for antibody, loader_iter in iterators.items():
                try:
                    features, labels = next(loader_iter)
                except StopIteration:
                    if antibody == targeted_AB:
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
                hist_loss[antibody] += loss.item()

        if print_epoch:
            for antibody in iterators:
                loss_hist[antibody].append(hist_loss[antibody] / len(trainload[antibody]))
                print(f"Epoch={ep} loss={loss_hist[antibody][ep]}, antibody={antibody}")
