# Amino_acid_transfer_learning_for_antibody_binding_prediction

## Background:
Taft and colleagues ([J. M. Taft, et al., 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9428596/)) created mutagenesis libraries of RBD domain of SARS-CoV-2 spike protein. RBD variants were expressed on the yeast surface as a C-terminal fusion to another protein. Binding to ACE2 and 4 therapeutic antibodies was measured.
Machine learning models predicted ACE2 binding and antibody escape. Models tested included KNN, Log Reg, naive Bayes, SVMs, RFs; RNNs. RF and RNN models showed the best metrics.

## Purpose:
Minimize the size of a train dataset with transfer learning or other methods.

## Methods:

### Multi-task:
Multi-task is a machine learning approach where the same information is used for multiple tasks. In current work, predicting escape from different antibodies for the same RBD sequence is considered as "multiple tasks".
![img](images/multi_task.jpg)

## Results:

