import torch
from torch.utils.data import Dataset

def make_mlp(units, activation = torch.nn.ReLU(), last_activation = torch.nn.Identity(), dropout=None):
    operation_list = []

    for i in range(len(units)-2):
        operation_list.append(torch.nn.Linear(units[i], units[i+1]))
        if dropout:
            operation_list.append(torch.nn.Dropout(p=dropout))
        operation_list.append(activation)

    operation_list.append(torch.nn.Linear(units[-2], units[-1]))
    operation_list.append(last_activation)

    return torch.nn.Sequential(*operation_list)

def preprocess_nyt_data(path, test_portion = 0.1):
    with open(path+'/nyt_data.txt') as f:
        raw_file = f.readlines()

    word_to_id = {}
    id_to_word = {}

    usa = open(path+'/nyt_vocab.dat')
    counter = 0
    for line in usa:
        word = line.strip()
        word_to_id[word] = counter
        id_to_word[counter] = word
        counter += 1

    word_list = []
    count_list = []

    for _, doc in enumerate(raw_file):
        words = doc.strip().split(',')
        w_l = []
        c_l = []

        for vocab in words:
            ind, num = vocab.split(':')
            w_l.append(int(ind) - 1)
            c_l.append(int(num))

        word_list.append(w_l)
        count_list.append(c_l)

    padded_words = torch.torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in word_list], batch_first = True)
    padded_counts = torch.torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in count_list], batch_first = True)

    statistics = {'D': len(word_list),
                  'V': len(word_to_id),
                  'word_to_id': word_to_id,
                  'id_to_word': id_to_word,}

    return padded_words, padded_counts, statistics

class BoWDataset(Dataset):

    def __init__(self, w_data, c_data):
        assert w_data.shape == c_data.shape, "word and count shape does not match"
        self.w = w_data
        self.c = c_data

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.w[idx, :], self.c[idx, :]), idx

def lbeta_v0(a):
    log_gamma_sum = torch.lgamma(torch.sum(a, dim = -1, keepdim=True))
    log_gamma_a = torch.lgamma(a)
    return torch.sum(log_gamma_a, dim=-1, keepdim=True) - log_gamma_sum

def kl_divergence_dirichlets(a, b):
    digamma_sum_d1 = torch.digamma(
        torch.sum(a, dim=-1, keepdims=True))
    digamma_diff = torch.digamma(a) - digamma_sum_d1
    concentration_diff = a - b

    return (
            torch.sum(concentration_diff * digamma_diff, dim=-1, keepdim=True) -
            lbeta_v0(a) + lbeta_v0(b))

class LogBeta(torch.autograd.Function):
    def forward(self, a, b):
        logbeta_ab = a.lgamma() + b.lgamma() - (a + b).lgamma()
        self.save_for_backward(a, b)
        return logbeta_ab

    def backward(self, grad_output):
        a, b = self.saved_tensors
        digamma_ab = torch.polygamma(0, a + b)
        return grad_output * (torch.polygamma(0, a) - digamma_ab), grad_output * (torch.polygamma(0, b) - digamma_ab)

if __name__ == '__main__':
    pass
#     def test_func(a=0, b=1):
#         return a+b
#     print(test_func(**{'b': 2}, a=1))
#
#     import numpy as np
#     units = [2, 3, 5, 7, 11]
#     test_x = torch.tensor(np.arange(20).reshape(10, 2).astype('float32'))
#     test_mlp = make_mlp(units)
#     print(test_mlp(test_x).shape)

