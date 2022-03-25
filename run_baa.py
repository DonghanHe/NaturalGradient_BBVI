import matplotlib.pyplot as plt
from models.BayesianArchetypeAnalysis import BayesianAA
from training.trainers import train_model_na
from torch.utils.data import DataLoader
import torch
from numpy import arange
from numpy.random import randint, choice
from sklearn.datasets import fetch_olivetti_faces

def show_random_reconstructed_faces_with_histogram0(inds, z_logits, model, data, shape=(64, 64), mean = False):
    x_hat = model.sample(z_logits[inds, :], mean=mean)


    for j, i in enumerate(inds):
        x_hat_i = x_hat[j, :].data
        x_i = data[i, :]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle('face {}'.format(i))
        ax1.imshow(-x_i.reshape(shape), cmap=plt.cm.binary)
        ax1.set_title('original')
        ax2.imshow(-x_hat_i.reshape(shape), cmap=plt.cm.binary)
        ax2.set_title('reconstructed')
        ax3.bar(arange(model.T), torch.nn.Softplus()(z_logits[inds[j], :]).data)
        ax3.set_title('un-normalized template')
        plt.show()


def show_random_reconstructed_faces_with_histogram(inds, z_logits, model, data, shape=(64, 64), mean=False):
    x_hat = model.sample(z_logits[inds, :], mean=mean)

    fig, ax = plt.subplots(ncols=len(inds), nrows=3, figsize=(24,16))

    for j, i in enumerate(inds):
        x_hat_i = x_hat[j, :].data
        x_i = data[i, :]

        ax[0, j].imshow(-x_i.reshape(shape), cmap=plt.cm.binary)
        ax[0, j].set_title('original {}'.format(i))
        ax[1, j].imshow(-x_hat_i.reshape(shape), cmap=plt.cm.binary)
        ax[1, j].set_title('reconstructed {}'.format(i))
        ax[2, j].bar(arange(model.T), torch.nn.Softplus()(z_logits[inds[j], :]).data)
        ax[2, j].set_title('template {}'.format(i))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    face_data = fetch_olivetti_faces(data_home='./datasets/', shuffle=True)

    raw_data = face_data['data']

    batch_size = 10
    epochs = 500
    num_archetypes = 40

    data_and_ind = torch.utils.data.TensorDataset(torch.tensor(raw_data), torch.tensor(arange(raw_data.shape[0])))
    trainloader = DataLoader(dataset=data_and_ind, batch_size=batch_size, shuffle=True, )

    model_args = {
        "num_archetypes": num_archetypes,
        "num_data": raw_data.shape[0],
        "dataset": torch.tensor(raw_data, requires_grad=False),
        "loss": 'bce',
        "data_archetype_prior": 0.2,
        "archetype_data_prior": 0.2,
    }

    adam_results = train_model_na(BayesianAA,
                                  model_args,
                                  (raw_data.shape[0], num_archetypes),
                                  trainloader,
                                  None,
                                  None,
                                  DEVICE,
                                  optimizer=torch.optim.Adam,
                                  epochs=epochs,
                                  natgrad=False,
                                  lr=1e-1,
                                  )

    ngd_results = train_model_na(BayesianAA,
                                  model_args,
                                  (raw_data.shape[0], num_archetypes),
                                  trainloader,
                                  None,
                                  None,
                                  DEVICE,
                                  optimizer=torch.optim.Adam,
                                  epochs=epochs,
                                  natgrad=True,
                                  lr=1e-1,
                                  )


    adam_loss_results = adam_results['loss_records']
    ngd_loss_results = ngd_results['loss_records']

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))

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

    # then we plot a couple samples from each training routine
    samples = 10
    inds = choice(arange(raw_data.shape[0]), size=samples, replace=False)
    show_random_reconstructed_faces_with_histogram(inds, adam_results['var_params'], adam_results['model'], raw_data)
    show_random_reconstructed_faces_with_histogram(inds, ngd_results['var_params'], ngd_results['model'], raw_data)