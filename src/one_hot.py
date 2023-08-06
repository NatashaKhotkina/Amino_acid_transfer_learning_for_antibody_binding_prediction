import numpy as np
from sklearn.preprocessing import OneHotEncoder


aa_alph = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
enc = OneHotEncoder()
encoding = enc.fit_transform(np.array(aa_alph).reshape(-1, 1)).toarray()
encoding_dict = dict(zip(aa_alph, encoding))


def encoding_func(dataset, sequence_column_name):
    encoded_list = []
    for i in range(len(dataset)):
        seq = dataset[sequence_column_name].iloc[i]
        seq_encoded = []
        for letter in seq:
            seq_encoded.append(encoding_dict[letter])
        seq_encoded = np.array(seq_encoded)
        encoded_list.append(seq_encoded)
    encoded_list = np.array(encoded_list)
    return encoded_list
