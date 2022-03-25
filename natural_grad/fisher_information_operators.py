import torch

def dirichlet_multiply_by_fisher_inv(alphas, x):
    # alphas is of shape (N, D)
    # x of shape (N, D)
    alpha0 = torch.sum(alphas, dim=-1, keepdim=True)
    dg_a0 = torch.polygamma(1, alpha0)
    dg_a = torch.polygamma(1, alphas)
    inv_dg_a = 1 /dg_a

    # first get the multiply by inv diag matrix
    A_inv_mul_x = inv_dg_a * x
    Corr_top_mul_x = inv_dg_a * torch.sum(A_inv_mul_x, dim=-1, keepdim=True)  # (N, D) * (N, 1)
    Corr_bot = 1 / (-dg_a0) + torch.sum(inv_dg_a, dim=-1, keepdim=True)
    Corr_mul_x = Corr_top_mul_x / Corr_bot

    # if torch.any(torch.isnan(A_inv_mul_x)):
    #     print('A^-1 x has nan'.format(torch.any(torch.isnan(A_inv_mul_x))))
    #     print(dg_a.max(), dg_a.min())
    #     print(inv_dg_a)
    #     print(x.max(), x.min())
    #     print(A_inv_mul_x)
    #     print('alpha has nan: {}'.format(torch.any(torch.isnan(alphas))))
    #     print('gradient has nan: {}'.format(torch.any(torch.isnan(x))))
    #     print('dg_a0 has nan: {}'.format(torch.any(torch.isnan(dg_a0))))
    #     print('dg_a0 has nan: {}'.format(torch.any(torch.isnan(dg_a0))))
    #     print('inv_dg_a has nan: {}'.format(torch.any(torch.isnan(inv_dg_a))))
    #
    # if torch.any(torch.isnan(Corr_mul_x)):
    #     print('rank one term nan')
    #     print('alpha has nan: {}'.format(torch.any(torch.isnan(alphas))))
    #     print('gradient has nan: {}'.format(torch.any(torch.isnan(x))))
    #     print('dg_a0 has nan: {}'.format(torch.any(torch.isnan(dg_a0))))
    #     print('dg_a0 has nan: {}'.format(torch.any(torch.isnan(dg_a0))))
    #     print('inv_dg_a has nan: {}'.format(torch.any(torch.isnan(inv_dg_a))))
    return A_inv_mul_x - Corr_mul_x

def gaussian_multiply_by_fisher_inv(mu_and_logsigma, x):
    # mu_and_logsigma is of shape (N, 2), [\in R^N, R+^2]
    # x of shape (N, 2)
    # first [N, 1] will be multiply by \sigma^2
    sigma_sq = torch.exp(mu_and_logsigma[:, :, 1]*2)
    # second [N, 1] will be just 2
    twos = torch.ones_like(sigma_sq)/2.

    return x * torch.stack([sigma_sq, twos], dim=-1)

def gamma_multiply_by_fisher_inv(alpha_and_beta, x):
    # alpha_and_beta of shape (N, 2)
    # x of shape (N, 2)
    alphas, betas = torch.split(alpha_and_beta, split_size_or_sections=1, dim=-1)
    dg_alpha = torch.polygamma(1, alphas)
    inv_beta = 1./betas
    alpha_div_beta_sq = alphas/(betas**2)

    fisher_matrix = torch.reshape(torch.stack([torch.stack([dg_alpha, inv_beta], dim=-1),
                                 torch.stack([inv_beta, alpha_div_beta_sq], dim=-1)], dim=-1),
                                  (x.shape[0], x.shape[1], x.shape[1]))

    return torch.linalg.solve(fisher_matrix, x)

class dirichlet_natgrad_backprop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        # forward propagation is just the identity
        return inputs

    @staticmethod
    def backward(ctx, dinputs):
        inputs, = ctx.saved_tensors
        dinputs_tilde = dirichlet_multiply_by_fisher_inv(inputs, dinputs)
        return dinputs_tilde

class gaussian_natgrad_backprop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        # forward propagation is just the identity
        return inputs

    @staticmethod
    def backward(ctx, dinputs):
        inputs, = ctx.saved_tensors
        dinputs_tilde = gaussian_multiply_by_fisher_inv(inputs, dinputs)
        return dinputs_tilde

class gamma_natgrad_backprop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        # forward propagation is just the identity
        return inputs

    @staticmethod
    def backward(ctx, dinputs):
        inputs, = ctx.saved_tensors
        dinputs_tilde = gamma_multiply_by_fisher_inv(inputs, dinputs)
        return dinputs_tilde

# if __name__ == '__main__':
#     test_x = torch.ones((3, 5, 2))
#     test_ml = torch.ones((3, 5, 2))
#     print('== test gaussian ==')
#     print(gaussian_multiply_by_fisher_inv(test_ml, test_x))
#     print('above\'s rows should be [7.3891, 0.5]')
#
#     test_x = torch.ones((3, 3))
#     test_ml = torch.ones((3, 3))
#     print('== test dirichlet ==')
#     print(dirichlet_multiply_by_fisher_inv(test_x, test_ml))
#     print('above\'s rows should be [2.1732, 2.1732, 2.1732]')
#
#     test_x = torch.ones((3, 2))
#     test_ml = torch.ones((3, 2))
#     print('== test gamma ==')
#     gamma_multiply_by_fisher_inv(test_ml, test_x)