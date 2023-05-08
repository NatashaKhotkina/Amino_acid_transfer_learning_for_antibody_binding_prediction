# Amino_acid_transfer_learning_for_antibody_binding_prediction

## Background:
Taft and colleagues ([J. M. Taft, et al., 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9428596/)) created mutagenesis libraries of RBD domain of SARS-CoV-2 spike protein. RBD variants were expressed on the yeast surface as a C-terminal fusion to another protein. Binding to ACE2 and 4 therapeutic antibodies was measured.
Machine learning models predicted ACE2 binding and antibody escape. Models tested included KNN, Log Reg, naive Bayes, SVMs, RFs; RNNs. RF and RNN models showed the best metrics.

## Purpose:
Minimize the size of a train dataset with transfer learning or other methods.

## Methods:

### Multi-task
Multi-task is a machine learning approach where the same information is used for multiple tasks. In current work, predicting escape from different antibodies for the same RBD sequence is considered as "multiple task".
![img](images/multi_task.jpg)

## Results:
### Dependence of ROC AUC on train size
ROC AUC depends on size of train set. It grows with the increase of the train dataset, but then reaches a plateau.

![img](images/train_size.png)

### LSTM vs Multi-task LSTM
Pretraining on other antibodies with multi-task approach improves metrics, but the increase is small.

![img](images/roc_auc.png)

## Data and packages
### Training and test data

All trainand test data used in this project is stored in **data** directory.

### Functions and classes

All reusable and non-interactive code is stored in **src** package.

## Usage

### Launch notebook copy in Google Colab (web)

https://github.com/NatashaKhotkina/Amino_acid_transfer_learning_for_antibody_binding_prediction/blob/main/Antibody_binding.ipynb