import matplotlib.pyplot as plt
from models.VAE import VariationalDecoder
from models.utils import preprocess_nyt_data, BoWDataset
from training.trainers import train_model_na
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

if __name__ == '__main__':
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    train_set = MNIST('./datasets/', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(lambda x: torch.flatten(x)),
                      ]), train=True)

    test_set = MNIST('./datasets/', download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Lambda(lambda x: torch.flatten(x)),
                     ]), train=False)

    batch_size = 1000
    epochs = 20
    latent_dim = 32

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, )
    eval_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, )

    model_args = {
        'latent_dim': latent_dim,
        'intermediate_units': [64, 256,],
        'output_dim': 784,
        'output_activation': torch.nn.Sigmoid(),
        'loss_function': 'bce',
        'num_data': len(train_set)
    }

    adam_results = train_model_na(
        VariationalDecoder,
        model_args,
        (len(train_set) + len(test_set), latent_dim, 2),
        train_loader,
        eval_loader,
        None,
        DEVICE,
        optimizer = torch.optim.Adam,
        epochs = epochs,
        natgrad = False,
        lr = 1e-3,
        eval_every = epochs+1,
    )

    ngd_results = train_model_na(
        VariationalDecoder,
        model_args,
        (len(train_set) + len(test_set), latent_dim, 2),
        train_loader,
        eval_loader,
        None,
        DEVICE,
        optimizer=torch.optim.Adam,
        epochs=epochs,
        natgrad=True,
        lr=1e-3,
        eval_every=epochs + 1,
    )

    adam_loss_results = adam_results['loss_records']
    ngd_loss_results = ngd_results['loss_records']

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))

    ax[0].plot(adam_loss_results['nll'], label='adam')
    ax[0].plot(ngd_loss_results['nll'], label='ngd')
    ax[0].set_title('neg log likelihood')
    ax[0].grid()

    ax[1].plot(adam_loss_results['kl_d'], label='adam')
    ax[1].plot(ngd_loss_results['kl_d'], label='ngd')
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




