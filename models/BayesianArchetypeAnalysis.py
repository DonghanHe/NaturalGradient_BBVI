import torch
from natural_grad.fisher_information_operators import dirichlet_natgrad_backprop
import torch.distributions as td

class BayesianAA(torch.nn.Module):

    def __init__(self,
                 num_archetypes,
                 num_data,
                 dataset,
                 data_archetype_prior = 1.,
                 archetype_data_prior = 1.,
                 natgrad = False,
                 loss = 'l2'
                 ):
        super(BayesianAA, self).__init__()
        self.T = num_archetypes
        self.N = num_data
        self.alpha = data_archetype_prior
        self.eta = archetype_data_prior
        self.natgrad = natgrad
        self.data = dataset # should be of shape (D, F)

        self.z_prior = td.Dirichlet(torch.ones((self.T,)) * self.alpha)
        self.t_prior = td.Dirichlet(torch.ones((self.N,)) * self.eta)
        self.t_logits = torch.nn.Parameter(torch.FloatTensor(self.T, self.N, ))
        torch.nn.init.xavier_uniform_(self.t_logits, )

        if loss == 'l2':
            self.loss = self.l2loss2
        elif loss == 'bce':
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

    def forward(self, x, z_logits):
        z_params = torch.clamp(torch.nn.Softplus()(z_logits), min=1e-3, max=1e3)
        if self.natgrad:
            z_post = td.Dirichlet(dirichlet_natgrad_backprop.apply(z_params), validate_args=False)
        else:
            z_post = td.Dirichlet(z_params, validate_args=False)
        z_sample = z_post.rsample()
        kl_z = torch.mean(self.klqp(z_post, self.z_prior))  # of shape (D_batch, 1) before taking mean

        t_params = torch.clamp(torch.nn.Softplus()(self.t_logits), min = 1e-3, max=1e3)
        if self.natgrad:
            t_post = td.Dirichlet(dirichlet_natgrad_backprop.apply(t_params), validate_args=False)
        else:
            t_post = td.Dirichlet(t_params, validate_args=False)
        t_sample = t_post.rsample()
        kl_t = torch.sum(self.klqp(t_post, self.t_prior))  # of shape (T, 1) before taking sum

        x_hat = torch.matmul(z_sample, torch.matmul(t_sample, self.data))
        logp = self.loss(x_hat, x)

        reconstruct_loss = logp / z_logits.shape[0]

        return reconstruct_loss * self.N, kl_z * self.N + kl_t

    def sample(self, z_logits, mean = False):
        z_params = torch.clamp(torch.nn.Softplus()(z_logits), min=1e-3, max=1e3)
        z_post = td.Dirichlet(z_params, validate_args=False)
        if mean == False:
            z_sample = z_post.rsample()
        else:
            z_sample = z_post.mean

        t_params = torch.clamp(torch.nn.Softplus()(self.t_logits), min=.1, max=1e3)
        t_post = td.Dirichlet(t_params, validate_args=False)
        if mean == False:
            t_sample = t_post.rsample()
        else:
            t_sample = t_post.mean

        x_hat = torch.matmul(z_sample, torch.matmul(t_sample, self.data))

        return x_hat

    def klqp(self, approx_posterior, prior):
        return td.kl_divergence(approx_posterior, prior)
