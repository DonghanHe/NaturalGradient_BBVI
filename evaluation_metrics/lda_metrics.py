import torch
import torch.distributions as td
from numpy import take_along_axis, array

# metrics need to be the same for different batches

def log_perplexity_uniform(data, model):
    test_words, test_counts = data

    num_topics = model.T
    test_document_topics = torch.ones((test_words.shape[0], num_topics)) / num_topics  # uniform topic distribution

    topic_dist = td.Dirichlet(torch.clamp(torch.nn.Softplus()(model.t_logits), min=.1, max=1e3))
    topic_mean = topic_dist.mean

    test_probs = torch.matmul(test_document_topics, topic_mean)

    logp = torch.log(torch.gather(input=test_probs, index=test_words, dim=1))
    sum_logp = torch.sum(test_counts * logp, ) # the numerator
    sum_count = torch.sum(test_counts)

    return -sum_logp/sum_count

def log_perplexity(dataloader, model, device):
    topic_dist = td.Dirichlet(torch.clamp(torch.nn.Softplus()(model.t_logits), min=.1, max=1e3))
    topic_mean = topic_dist.mean

    metric_num = 0.
    metric_denom = 0.

    for data, _ in dataloader:
        w_i, c_i = data
        w_i = w_i.to(device)
        c_i = c_i.to(device)
        expanded_w_i = torch.unsqueeze(w_i, 1)
        expanded_c_i = torch.unsqueeze(c_i, 1)
        word_topic_likelihood = take_along_axis(topic_mean.reshape((1, *topic_mean.shape)),
                                                array(expanded_w_i),
                                                axis=-1)
        doc_topic_prob = word_topic_likelihood/torch.sum(word_topic_likelihood, dim=1, keepdim=True)
        doc_topic_mix = torch.sum(doc_topic_prob * expanded_c_i, dim=-1)/torch.sum(expanded_c_i, dim=-1) # (D_batch, T)

        test_probs = torch.matmul(doc_topic_mix, topic_mean)

        logp = torch.log(torch.gather(input=test_probs, index=w_i, dim=1))
        sum_logp = torch.sum(c_i * logp, )  # the numerator
        sum_count = torch.sum(c_i) # the denominator

        metric_num += sum_logp
        metric_denom += sum_count

    return -metric_num/metric_denom

def log_probability(test_x, decoded_x):
    # TODO: build log_probability metric, VAE evaluation metric
    pass

class coherence_metric:

    def __init__(self, test_loader, dictionary, coherence_type='u_mass'):
        self.coh = coherence_type
        self.bow = self.build_gensim_bow(test_loader)
        self.dictionary = dictionary

    @staticmethod
    def build_gensim_bow(test_loader):
        res = []
        for data, _ in test_loader:
            w_i, c_i = data

            for w_ij, c_ij in zip(w_i.item(), c_i.item()):
                untruncated = [(w, c) for w,c in zip(w_ij, c_ij)]
                ind = next((i for i, x in enumerate(c_ij) if x), None)
                res.append(untruncated[:ind])
        return res

    def coh_metric(self, topics, ):
        # TODO: figure out what we want here for dictionary
        cm = CoherenceModel(topics=topics, corpus=self.bow, dictionary=self.dictionary, coherence=self.coh)

        return cm.get_coherence()

if __name__ == '__main__':
    pass