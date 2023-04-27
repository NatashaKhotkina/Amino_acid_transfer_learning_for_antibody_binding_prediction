from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch import nn

from src.models import LSTMMultiModel


def eval_model(model, testload, multi_task_targeted_AB=None, device='cpu'):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    model.eval()

    with torch.no_grad():
        for data in testload:
            features, labels = data
            features = features.to(device)
            if multi_task_targeted_AB and isinstance(model, LSTMMultiModel):
                logits = model(features, multi_task_targeted_AB)
            else:
                logits = model(features)
            outputs = nn.Sigmoid()(logits)
            outputs = outputs.squeeze().cpu()
            roc_auc.append(roc_auc_score(labels, outputs))
            outputs = outputs.round()
            accuracy.append(accuracy_score(labels, outputs))
            precision.append(precision_score(labels, outputs))
            recall.append(recall_score(labels, outputs))
            f1.append(f1_score(labels, outputs))

    mean_accuracy = sum(accuracy) / len(accuracy)
    mean_precision = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)
    mean_f1 = sum(f1) / len(f1)
    mean_roc_auc = sum(roc_auc) / len(roc_auc)

    return mean_accuracy, mean_precision, mean_recall, mean_f1, mean_roc_auc


# def eval_multi_model(model, testload, device='cpu', targeted_AB='ly16'):
#     accuracy = []
#     precision = []
#     recall = []
#     f1 = []
#     roc_auc = []
#     model.eval()
#
#     with torch.no_grad():
#         for data in testload[targeted_AB]:
#             features, labels = data
#             features = features.to(device)
#             labels = labels
#             logits = model(features, targeted_AB)
#             outputs = nn.Sigmoid()(logits)
#             outputs = outputs.squeeze().cpu()
#             roc_auc.append(roc_auc_score(labels, outputs))
#             outputs = outputs.round()
#             accuracy.append(accuracy_score(labels, outputs))
#             precision.append(precision_score(labels, outputs))
#             recall.append(recall_score(labels, outputs))
#             f1.append(f1_score(labels, outputs))
#
#     mean_accuracy = sum(accuracy) / len(accuracy)
#     mean_precision = sum(precision) / len(precision)
#     mean_recall = sum(recall) / len(recall)
#     mean_f1 = sum(f1) / len(f1)
#     mean_roc_auc = sum(roc_auc) / len(roc_auc)
#     return mean_accuracy, mean_precision, mean_recall, mean_f1, mean_roc_auc
