import matplotlib.pyplot as plt
from models.LDA import SmoothedLDA
from models.utils import preprocess_nyt_data, BoWDataset
from training.trainers import train_model_na
from torch.utils.data import random_split, DataLoader
from evaluation_metrics.lda_metrics import log_perplexity
import torch

def get_top_words_per_topic(topics, id_to_word = {}, top_k=20):
    # doesn't matter they are logits or not.. exp is a monotonic function
    top_words = torch.argsort(-topics, axis=-1)[:, :top_k]
    res = []
    for t in range(topics.shape[0]):
        res.append([id_to_word[i.item()] for i in top_words[t, :]])

    return res

if __name__ == '__main__':
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    W_data, C_data, data_statistics = preprocess_nyt_data('datasets/nytimes')
    full_dataset = BoWDataset(W_data, C_data)

    batch_size = 64
    epochs = 100
    num_topics = 30
    train_set, test_set = random_split(full_dataset,
                                       [int(W_data.shape[0]*0.9), W_data.shape[0]-int(W_data.shape[0]*0.9)])
    print(len(train_set))
    print(len(test_set))
    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,)
    testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,)

    model_args = {
        "num_topics": num_topics,
        "num_words": data_statistics['V'],
        "num_documents": len(train_set),
    }

    adam_results = train_model_na(SmoothedLDA,
                                   model_args,
                                   (len(train_set) + len(test_set), num_topics),
                                   trainloader,
                                   testloader,
                                   log_perplexity,
                                   DEVICE,
                                   optimizer=torch.optim.Adam,
                                   epochs=epochs,
                                   natgrad=False,
                                   lr=1e-3
                                   )

    ngd_results = train_model_na(SmoothedLDA,
                                  model_args,
                                  (len(train_set) + len(test_set), num_topics),
                                  trainloader,
                                  testloader,
                                  log_perplexity,
                                  DEVICE,
                                  optimizer=torch.optim.SGD,
                                  epochs=epochs,
                                  natgrad=True,
                                  lr=1e-3,
                                  )

    adam_loss_results = adam_results['loss_records']
    ngd_loss_results = ngd_results['loss_records']

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10,6))

    ax[0].plot(adam_loss_results['nll'], label='adam')
    ax[0].plot(ngd_loss_results['nll'], label='ngd')
    ax[0].set_title('neg log likelihood')
    ax[0].grid()

    ax[1].plot(adam_loss_results['kl'], label='adam')
    ax[1].plot(ngd_loss_results['kl'], label='ngd')
    ax[1].set_title('kl')
    ax[1].grid()

    fig.suptitle('-ELBo in parts')
    fig.legend(bbox_to_anchor=(1.05, 0))
    plt.show()

    plt.plot(adam_loss_results['-elbo'], label='adam')
    plt.plot(ngd_loss_results['-elbo'], label='ngd')
    plt.title('-ELBo')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(torch.linspace(0, epochs, 10), adam_results['metric_record'], label='adam')
    plt.plot(torch.linspace(0, epochs, 10), ngd_results['metric_record'], label='ngd')
    plt.legend()
    plt.title('log perplexity')
    plt.grid()
    plt.show()

    id2word = data_statistics['id_to_word']
    top_topic_adam = get_top_words_per_topic(adam_results['model'].t_logits, id_to_word=id2word)
    top_topic_ngd = get_top_words_per_topic(ngd_results['model'].t_logits, id_to_word=id2word)

    print('================ adam topics ================')
    for i,l in enumerate(top_topic_adam):
        print('** topic {} **'.format(i))
        print(l)

    print('================ ngd topics ================')
    for i, l in enumerate(top_topic_ngd):
        print('** topic {} **'.format(i))
        print(l)


