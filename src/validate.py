from collections import defaultdict

import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)
from torch import nn
from torch.utils.data import DataLoader

from src.models import LSTMMultiModel


def eval_model(model, testload, criterion, targeted_ab=None, device='cpu', print_output=False):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    average_precision = []
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for data in testload:
            features, labels = data
            features = features.to(device)

            if isinstance(targeted_ab, str) and isinstance(model, LSTMMultiModel):
                logits = model(features, targeted_ab)

            elif isinstance(targeted_ab, list) and isinstance(model, LSTMMultiModel):
                logits_list = []
                for antibody in targeted_ab:
                    logits = model(features, antibody)
                    logits_list.append(logits)
                logits = torch.mean(torch.stack(logits_list), dim=0)

            else:
                logits = model(features)
            loss = criterion(logits.cpu(), labels.unsqueeze(1))
            val_loss += loss.item()
            outputs = nn.Sigmoid()(logits)
            outputs = outputs.squeeze().cpu()

            roc_auc.append(roc_auc_score(labels, outputs))
            average_precision.append(average_precision_score(labels, outputs))

            if print_output:
                print(labels)
                print(outputs)
            outputs = outputs.round()
            accuracy.append(accuracy_score(labels, outputs))
            precision.append(precision_score(labels, outputs, zero_division=0.0))
            recall.append(recall_score(labels, outputs, zero_division=0.0))
            f1.append(f1_score(labels, outputs))

    mean_accuracy = sum(accuracy) / len(accuracy)
    mean_precision = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)
    mean_f1 = sum(f1) / len(f1)
    mean_roc_auc = sum(roc_auc) / len(roc_auc)
    mean_loss = val_loss / (len(testload))
    mean_average_precision = sum(average_precision) / len(average_precision)

    return (mean_loss, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_roc_auc,
            mean_average_precision)