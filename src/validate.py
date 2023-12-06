from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch import nn

from src.models import LSTMMultiModel


def eval_model(model, testload, criterion, targeted_ab=None, device='cpu', output_filename=None):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    average_precision = []
    val_loss = 0
    output_tensor = torch.Tensor()
    label_tensor = torch.Tensor()
    model.eval()

    with torch.no_grad():
        for data in testload:
            features, labels = data
            features = features.to(device)
            if targeted_ab and isinstance(model, LSTMMultiModel):
                logits = model(features, targeted_ab)
            else:
                logits = model(features)
            loss = criterion(logits.cpu(), labels.unsqueeze(1))
            val_loss += loss.item()
            outputs = nn.Sigmoid()(logits)
            outputs = outputs.squeeze().cpu()

            roc_auc.append(roc_auc_score(labels, outputs))

            # if labels.sum() > 0.5 * len(labels):
            #     labels_ap = 1 - labels
            #     outputs_ap = 1 - outputs
            # else:
            #     labels_ap = labels
            #     outputs_ap = outputs
            # average_precision.append(average_precision_score(labels_ap, outputs_ap))
            average_precision.append(average_precision_score(labels, outputs))

            if output_filename is not None:
                output_tensor = torch.cat((output_tensor, outputs))
                label_tensor = torch.cat((label_tensor, labels))

            outputs = outputs.round()
            accuracy.append(accuracy_score(labels, outputs))
            precision.append(precision_score(labels, outputs, zero_division=0.0))
            recall.append(recall_score(labels, outputs, zero_division=0.0))
            f1.append(f1_score(labels, outputs))

    if output_filename is not None:
        torch.save(output_tensor, f'outputs_{output_filename}')
        torch.save(label_tensor, f'labels_{output_filename}')
    mean_accuracy = sum(accuracy) / len(accuracy)
    mean_precision = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)
    mean_f1 = sum(f1) / len(f1)
    mean_roc_auc = sum(roc_auc) / len(roc_auc)
    mean_loss = val_loss / (len(testload))
    mean_average_precision = sum(average_precision) / len(average_precision)

    return mean_loss, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_roc_auc, mean_average_precision