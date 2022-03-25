import torch
from natural_grad.fisher_information_operators import dirichlet_natgrad_backprop
import torch.distributions as td

class SmoothedLDA(torch.nn.Module):

    def __init__(self,
                 num_topics,
                 num_words,
                 num_documents,
                 document_topic_prior=1.,
                 topic_word_prior=1.,
                 natgrad=False,
                 ):
        super(SmoothedLDA, self).__init__()
        self.T = num_topics
        self.V = num_words
        self.D = num_documents
        self.alpha = document_topic_prior
        self.eta = topic_word_prior
        self.natgrad = natgrad
        self.d_prior = td.Dirichlet(torch.ones((self.T, ))*self.alpha)
        self.t_prior = td.Dirichlet(torch.ones((self.V, ))*self.eta)
        self.t_logits = torch.nn.Parameter(torch.FloatTensor(self.T, self.V, ))
        torch.nn.init.xavier_normal_(self.t_logits, 0.5)

    def forward(self, w, c, d_logits):
        d_params = torch.clamp(torch.nn.Softplus()(d_logits), min=.1, max=1e3)
        if self.natgrad:
            d_post = td.Dirichlet(dirichlet_natgrad_backprop.apply(d_params), validate_args=False)
        else:
            d_post = td.Dirichlet(d_params, validate_args=False)
        d_sample = d_post.rsample()
        kl_d = torch.mean(self.klqp(d_post, self.d_prior)) # of shape (D_batch, 1) before taking mean

        t_params = torch.clamp(torch.nn.Softplus()(self.t_logits), min=.1, max=1e3)
        if self.natgrad:
            t_post = td.Dirichlet(dirichlet_natgrad_backprop.apply(t_params), validate_args=False)
        else:
            t_post = td.Dirichlet(t_params, validate_args=False)
        t_sample = t_post.rsample()
        kl_t = torch.sum(self.klqp(t_post, self.t_prior)) # of shape (T, 1) before taking sum

        probs = torch.matmul(d_sample, t_sample)
        logp = torch.log(torch.gather(input = probs, index=w, dim=1))
        reconstruct_loss = torch.mean(torch.sum(c * logp, dim=1)) # of shape (D_batch, 1) before taking mean

        return reconstruct_loss * self.D, kl_d*self.D + kl_t

    def klqp(self, approx_posterior, prior):
        return td.kl_divergence(approx_posterior, prior)


# if __name__ == '__main__':
#     test_model = SmoothedLDA(30, 3012, 8447, 1., 1., natgrad=True)
#     test_c = torch.tensor([[1., 0., 0.],
#                            [1., 1., 0.]])
#     test_w = torch.tensor([[800, 397, 1024],
#                            [0, 112, 32]])
#     test_d_logits = torch.nn.Parameter(torch.FloatTensor(2, 30))
#     torch.nn.init.kaiming_uniform_(test_d_logits)
#     rl, kld, klt = test_model(test_w, test_c, test_d_logits)
#     print(rl, kld, klt)