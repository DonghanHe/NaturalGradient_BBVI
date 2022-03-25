import torch
from natural_grad.fisher_information_operators import gaussian_natgrad_backprop
from models.utils import make_mlp
import torch.distributions as td

class VariationalDecoder(torch.nn.Module):

    def __init__(self,
                 latent_dim,
                 intermediate_units,
                 output_dim,
                 num_data,
                 latent_prior=(0., 1.),
                 activation=torch.nn.ReLU(),
                 output_activation=torch.nn.Identity(),
                 loss_function = 'l2',
                 natgrad=False):
        super(VariationalDecoder, self).__init__()
        self.decoder = make_mlp([latent_dim]+intermediate_units+[output_dim,],
                                activation=activation,
                                last_activation=output_activation)
        self.N = num_data
        self.z_prior = td.Normal(loc=latent_prior[0],scale=latent_prior[1])
        self.natgrad=natgrad
        if loss_function == 'l2':
            self.loss = self.l2loss2
        elif loss_function == 'bce':
            self.loss = self.bceloss

    @staticmethod
    def l2loss(x_mu, x):
        return torch.sum(-0.5*(x-x_mu)**2 - 0.5 * torch.log(torch.tensor(3.1415926*2)), dim=-1)

    @staticmethod
    def l2loss2(x_mu, x):
        return -torch.nn.MSELoss(reduction='sum')(x_mu, x)/2

    @staticmethod
    def bceloss(x_mu, x):
        return -torch.nn.BCELoss(reduction='sum')(x_mu, x)

    def forward(self, x, z_mu_and_log_sigma, ):
        # z_mu and z_log_sigma of shape [D_batch, latent_dim, 1]
        if self.natgrad:
            z_mu_and_logsigma = gaussian_natgrad_backprop.apply(z_mu_and_log_sigma)
            z_mu = z_mu_and_logsigma[:, :, 0]
            z_sigma = torch.exp(z_mu_and_logsigma[:, :, 1])
            z_post = td.Normal(loc=z_mu, scale=z_sigma)
        else:
            z_sigma = torch.exp(z_mu_and_log_sigma[:, :, 1])
            z_post = td.Normal(loc=z_mu_and_log_sigma[:, :, 0], scale=z_sigma)
        z_sample = z_post.rsample()
        kl_z = torch.mean(td.kl_divergence(z_post, self.z_prior))

        x_mu = self.decoder(z_sample)
        logp = self.loss(x_mu, x)

        reconstruct_loss = logp/z_sigma.shape[0]

        return reconstruct_loss * self.N, kl_z * self.N

class VAE(torch.nn.Module):
    # TODO: write variational autoencoder
    def __init__(self, ):
        super(VAE, self).__init__()
        pass

# if __name__ == '__main__':
#     test_x = torch.zeros(31, 100)
#     test_params = torch.stack([torch.zeros(31, 10), torch.ones(31, 10)], dim=-1)
#     test_units = [15, 25, 50]
#     test_model = VariationalDecoder(10, test_units, 100, 31)
#     l, kl = test_model(test_x, test_params)
#     elbo = -l + kl
#     print(elbo)
#     print(elbo.backward())
